from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import csv
from pathlib import Path

import torch


# ============================================================
# Result container
# ============================================================

@dataclass
class ThresholdCalibrationResult:
    """
    Result of validation-set confidence calibration.
    """

    threshold: float

    precision: float
    recall: float
    specificity: float
    f1: float
    youden_j: float

    tp: int
    tn: int
    fp: int
    fn: int

    num_images: int

    mean_positive_score: float
    mean_negative_score: float

    visualization_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Detection Threshold Calibrator
# ============================================================

class DetectionThresholdCalibrator:

    def __init__(
        self,
        model,
        postprocessor,
        device=None,
        criterion="youden",
        visualization_rule=True,
        verbose=True,
    ):
        """
        Calibrate an image-level confidence threshold
        on a validation dataloader.

        The calibration is performed using:

            image score = maximum detection score

        Ground-truth image label:

            positive -> at least one GT box
            negative -> no GT boxes

        The selected threshold maximizes either:

            criterion="youden"
                Youden J = sensitivity + specificity - 1

            criterion="f1"
                F1 score

        IMPORTANT:
            The supplied postprocessor should normally use
            score_threshold=0.0 so that the complete score
            distribution is available for calibration.

            The NMS threshold remains the official NMS threshold.
        """

        if criterion not in {
            "youden",
            "f1",
        }:
            raise ValueError(
                "criterion must be either "
                "'youden' or 'f1'."
            )

        self.model = model
        self.postprocessor = postprocessor

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.criterion = criterion
        self.visualization_rule = (
            visualization_rule
        )
        self.verbose = verbose

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _safe_divide(
        numerator,
        denominator,
    ):
        if denominator == 0:
            return 0.0

        return (
            numerator / denominator
        )

    # ========================================================
    # Collect validation scores
    # ========================================================

    @torch.no_grad()
    def _collect_scores(
        self,
        dataloader,
    ):
        """
        Run one validation pass.

        Returns:

            positive_scores:
                maximum detection score for positive images.

            negative_scores:
                maximum detection score for negative images.

            image_records:
                one record per validation image.
        """

        self.model.eval()

        positive_scores = []
        negative_scores = []

        image_records = []

        total_images = 0
        positive_images = 0
        negative_images = 0

        if self.verbose:
            print()
            print(
                "[CALIBRATION] "
                "Collecting validation scores..."
            )

        total_batches = len(
            dataloader
        )

        for batch_index, (
            images,
            targets,
        ) in enumerate(
            dataloader,
            start=1,
        ):

            images = images.to(
                self.device
            )

            predictions = self.model(
                images
            )

            detections = (
                self.postprocessor(
                    predictions
                )
            )

            for sample_index, detection in enumerate(
                detections
            ):

                target = targets[
                    sample_index
                ]

                gt_boxes = (
                    target["boxes"]
                    .detach()
                    .cpu()
                )

                scores = (
                    detection["scores"]
                    .detach()
                    .cpu()
                )

                # ------------------------------------------------
                # Image-level GT
                # ------------------------------------------------

                is_positive = (
                    len(gt_boxes) > 0
                )

                # ------------------------------------------------
                # Maximum detection score
                # ------------------------------------------------

                if scores.numel() > 0:

                    max_score = float(
                        scores.max().item()
                    )

                else:

                    max_score = 0.0

                record = {
                    "is_positive": (
                        is_positive
                    ),
                    "max_score": max_score,
                    "num_detections": int(
                        scores.numel()
                    ),
                }

                image_records.append(
                    record
                )

                total_images += 1

                if is_positive:

                    positive_images += 1
                    positive_scores.append(
                        max_score
                    )

                else:

                    negative_images += 1
                    negative_scores.append(
                        max_score
                    )

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if (
                self.verbose
                and (
                    batch_index % 100 == 0
                    or
                    batch_index
                    == total_batches
                )
            ):

                progress = (
                    100.0
                    * batch_index
                    / total_batches
                )

                print(
                    "[CALIBRATION] "
                    f"batch={batch_index}/"
                    f"{total_batches} "
                    f"progress={progress:.1f}%"
                )

        if self.verbose:

            print()
            print(
                "[CALIBRATION] "
                "Score collection completed."
            )

            print(
                f"[CALIBRATION] "
                f"Images: {total_images}"
            )

            print(
                f"[CALIBRATION] "
                f"Positive images: "
                f"{positive_images}"
            )

            print(
                f"[CALIBRATION] "
                f"Negative images: "
                f"{negative_images}"
            )

            if positive_scores:

                print(
                    f"[CALIBRATION] "
                    f"Positive score range: "
                    f"{min(positive_scores):.4f} "
                    f"→ "
                    f"{max(positive_scores):.4f}"
                )

                print(
                    f"[CALIBRATION] "
                    f"Positive mean score: "
                    f"{sum(positive_scores) / len(positive_scores):.4f}"
                )

            if negative_scores:

                print(
                    f"[CALIBRATION] "
                    f"Negative score range: "
                    f"{min(negative_scores):.4f} "
                    f"→ "
                    f"{max(negative_scores):.4f}"
                )

                print(
                    f"[CALIBRATION] "
                    f"Negative mean score: "
                    f"{sum(negative_scores) / len(negative_scores):.4f}"
                )

        return (
            positive_scores,
            negative_scores,
            image_records,
        )

    # ========================================================
    # Evaluate one threshold
    # ========================================================

    @classmethod
    def _evaluate_threshold(
        cls,
        image_records,
        threshold,
    ):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for record in image_records:

            actual_positive = (
                record["is_positive"]
            )

            predicted_positive = (
                record["max_score"]
                >= threshold
            )

            if (
                actual_positive
                and predicted_positive
            ):

                tp += 1

            elif (
                not actual_positive
                and not predicted_positive
            ):

                tn += 1

            elif (
                not actual_positive
                and predicted_positive
            ):

                fp += 1

            else:

                fn += 1

        precision = cls._safe_divide(
            tp,
            tp + fp,
        )

        recall = cls._safe_divide(
            tp,
            tp + fn,
        )

        specificity = cls._safe_divide(
            tn,
            tn + fp,
        )

        f1 = cls._safe_divide(
            2.0
            * precision
            * recall,
            precision + recall,
        )

        youden_j = (
            recall
            + specificity
            - 1.0
        )

        return {
            "threshold": float(
                threshold
            ),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "youden_j": youden_j,
        }

    # ========================================================
    # Build threshold candidates
    # ========================================================

    @staticmethod
    def _build_thresholds(
        positive_scores,
        negative_scores,
        num_thresholds=101,
    ):
        """
        Build a deterministic threshold grid.

        We explicitly include:

            0.0
            1.0

        and a uniform grid between them.

        The default 101 values correspond to:

            0.00, 0.01, ..., 1.00
        """

        thresholds = [
            index / (
                num_thresholds - 1
            )
            for index in range(
                num_thresholds
            )
        ]

        return thresholds

    # ========================================================
    # Select best result
    # ========================================================

    def _select_best(
        self,
        rows,
    ):
        if self.criterion == "youden":

            best = max(
                rows,
                key=lambda row: (
                    row["youden_j"],
                    row["f1"],
                    row["threshold"],
                ),
            )

        else:

            best = max(
                rows,
                key=lambda row: (
                    row["f1"],
                    row["youden_j"],
                    row["threshold"],
                ),
            )

        return best

    # ========================================================
    # Save CSV
    # ========================================================

    @staticmethod
    def save_csv(
        rows,
        path,
    ):
        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not rows:
            return

        fieldnames = list(
            rows[0].keys()
        )

        with path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

    # ========================================================
    # Calibration
    # ========================================================

    def calibrate(
        self,
        dataloader,
        num_thresholds=101,
        save_path=None,
    ):
        """
        Calibrate confidence threshold.

        Returns:

            ThresholdCalibrationResult

        Optionally saves the complete threshold sweep
        to a CSV file.
        """

        (
            positive_scores,
            negative_scores,
            image_records,
        ) = self._collect_scores(
            dataloader
        )

        thresholds = (
            self._build_thresholds(
                positive_scores,
                negative_scores,
                num_thresholds=(
                    num_thresholds
                ),
            )
        )

        rows = []

        for threshold in thresholds:

            result = (
                self._evaluate_threshold(
                    image_records,
                    threshold,
                )
            )

            rows.append(
                result
            )

        # ----------------------------------------------------
        # Print detailed threshold analysis
        # ----------------------------------------------------

        if self.verbose:

            print()
            print(
                "=" * 85
            )

            print(
                "[CALIBRATION] "
                "IMAGE-LEVEL THRESHOLD ANALYSIS"
            )

            print(
                "=" * 85
            )

            print(
                f"{'Threshold':>10}"
                f"{'Precision':>12}"
                f"{'Recall':>12}"
                f"{'Specificity':>14}"
                f"{'F1':>12}"
                f"{'Youden':>12}"
            )

            print(
                "-" * 85
            )

            for row in rows:

                # Print every fifth threshold when a very
                # dense grid is used, plus the important
                # default threshold 0.10.
                threshold = row[
                    "threshold"
                ]

                should_print = (
                    num_thresholds <= 21
                    or
                    abs(
                        (
                            threshold
                            * 100
                        )
                        % 5
                    ) < 1e-9
                    or
                    abs(
                        threshold
                        - 0.10
                    ) < 1e-9
                )

                if not should_print:
                    continue

                print(
                    f"{threshold:>10.2f}"
                    f"{row['precision']:>12.4f}"
                    f"{row['recall']:>12.4f}"
                    f"{row['specificity']:>14.4f}"
                    f"{row['f1']:>12.4f}"
                    f"{row['youden_j']:>12.4f}"
                )

        # ----------------------------------------------------
        # Best threshold
        # ----------------------------------------------------

        best = self._select_best(
            rows
        )

        # ----------------------------------------------------
        # Visualization threshold
        # ----------------------------------------------------

        if self.visualization_rule:

            visualization_threshold = max(
                0.5
                * best["threshold"],
                0.10,
            )

        else:

            visualization_threshold = (
                best["threshold"]
            )

        result = ThresholdCalibrationResult(
            threshold=(
                best["threshold"]
            ),
            precision=(
                best["precision"]
            ),
            recall=(
                best["recall"]
            ),
            specificity=(
                best["specificity"]
            ),
            f1=(
                best["f1"]
            ),
            youden_j=(
                best["youden_j"]
            ),
            tp=(
                best["tp"]
            ),
            tn=(
                best["tn"]
            ),
            fp=(
                best["fp"]
            ),
            fn=(
                best["fn"]
            ),
            num_images=len(
                image_records
            ),
            mean_positive_score=(
                sum(
                    positive_scores
                )
                / len(
                    positive_scores
                )
                if positive_scores
                else 0.0
            ),
            mean_negative_score=(
                sum(
                    negative_scores
                )
                / len(
                    negative_scores
                )
                if negative_scores
                else 0.0
            ),
            visualization_threshold=(
                visualization_threshold
            ),
        )

        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        if save_path is not None:

            self.save_csv(
                rows,
                save_path,
            )

        # ----------------------------------------------------
        # Final log
        # ----------------------------------------------------

        if self.verbose:

            print()
            print(
                "=" * 85
            )

            print(
                "[CALIBRATION] RESULT"
            )

            print(
                "=" * 85
            )

            print(
                f"[CALIBRATION] Criterion: "
                f"{self.criterion}"
            )

            print(
                f"[CALIBRATION] "
                f"Optimal threshold τ*: "
                f"{result.threshold:.4f}"
            )

            print(
                f"[CALIBRATION] Precision: "
                f"{result.precision:.4f}"
            )

            print(
                f"[CALIBRATION] Recall: "
                f"{result.recall:.4f}"
            )

            print(
                f"[CALIBRATION] Specificity: "
                f"{result.specificity:.4f}"
            )

            print(
                f"[CALIBRATION] F1: "
                f"{result.f1:.4f}"
            )

            print(
                f"[CALIBRATION] Youden J: "
                f"{result.youden_j:.4f}"
            )

            print(
                f"[CALIBRATION] "
                f"TP={result.tp} "
                f"TN={result.tn} "
                f"FP={result.fp} "
                f"FN={result.fn}"
            )

            print(
                f"[CALIBRATION] "
                f"Mean positive max-score: "
                f"{result.mean_positive_score:.4f}"
            )

            print(
                f"[CALIBRATION] "
                f"Mean negative max-score: "
                f"{result.mean_negative_score:.4f}"
            )

            print(
                f"[CALIBRATION] "
                f"Visualization threshold: "
                f"{result.visualization_threshold:.4f}"
            )

            if (
                self.criterion
                == "youden"
            ):

                print(
                    "[CALIBRATION] "
                    "Visualization rule: "
                    "max(0.5 * τ*, 0.10)"
                )

            print(
                "=" * 85
            )

        return result

    # ========================================================
    # Convenience method
    # ========================================================

    def calibrate_and_return_threshold(
        self,
        dataloader,
        num_thresholds=101,
        save_path=None,
    ):
        """
        Convenience wrapper returning only τ*.
        """

        result = self.calibrate(
            dataloader=dataloader,
            num_thresholds=num_thresholds,
            save_path=save_path,
        )

        return result.threshold