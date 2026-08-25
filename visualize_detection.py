from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
    VAL_RATIO,
    SEED,
    NMS_THRESHOLD,
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_test_transforms,
)

from src.datasets.split import (
    create_train_val_split,
)

from src.models.detector import (
    DetectionFramework,
)

from src.inference import (
    DetectionPostProcessor,
)

from src.calibration import (
    DetectionThresholdCalibrator,
)


# ============================================================
# CONFIGURATION
# ============================================================

IOU_THRESHOLD = 0.50

OFFICIAL_SCORE_THRESHOLD = 0.10

DEFAULT_MAX_DETECTIONS = 10

VISUALIZATION_OVERLAP_THRESHOLD = 0.40

DEFAULT_NUM_FLOW_IMAGES = 6

IMAGE_DPI = 110


# ============================================================
# GEOMETRY
# ============================================================

def compute_box_area(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        *
        max(0.0, y2 - y1)
    )


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    union = (
        compute_box_area(box_a)
        +
        compute_box_area(box_b)
        -
        intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def compute_overlap_over_smaller_area(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = iw * ih

    smaller_area = min(
        compute_box_area(box_a),
        compute_box_area(box_b),
    )

    if smaller_area <= 0.0:
        return 0.0

    return intersection / smaller_area


def best_iou_against_gt(
    prediction_box,
    gt_boxes,
):
    if not gt_boxes:
        return 0.0

    return max(
        compute_iou(
            prediction_box,
            gt_box,
        )
        for gt_box in gt_boxes
    )


# ============================================================
# VISUALIZATION-ONLY REDUNDANCY FILTER
# ============================================================

def suppress_redundant_predictions(
    boxes,
    scores,
    overlap_threshold,
    max_detections,
):
    """
    This is ONLY for visualization.

    If two predictions overlap strongly, the larger-area box
    is retained. Score is used only as a tie-breaker.

    Official AP / AR are not affected.
    """

    if boxes.numel() == 0:
        return boxes, scores

    boxes_list = (
        boxes
        .detach()
        .cpu()
        .tolist()
    )

    scores_list = (
        scores
        .detach()
        .cpu()
        .tolist()
    )

    candidates = []

    for box, score in zip(
        boxes_list,
        scores_list,
    ):
        candidates.append(
            {
                "box": box,
                "score": float(score),
                "area": compute_box_area(box),
            }
        )

    kept = []

    while candidates:

        candidates.sort(
            key=lambda item: (
                item["area"],
                item["score"],
            ),
            reverse=True,
        )

        current = candidates.pop(0)

        kept.append(current)

        if len(kept) >= max_detections:
            break

        remaining = []

        for candidate in candidates:

            overlap = (
                compute_overlap_over_smaller_area(
                    current["box"],
                    candidate["box"],
                )
            )

            if overlap < overlap_threshold:
                remaining.append(candidate)

        candidates = remaining

    kept_boxes = torch.tensor(
        [
            item["box"]
            for item in kept
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    kept_scores = torch.tensor(
        [
            item["score"]
            for item in kept
        ],
        dtype=scores.dtype,
        device=scores.device,
    )

    return kept_boxes, kept_scores


# ============================================================
# BOX-LEVEL MATCHING
# ============================================================

def match_predictions_to_ground_truth(
    gt_boxes,
    pred_boxes,
    iou_threshold=0.50,
):
    """
    Greedy one-to-one matching.

    Returns:
        tp
        fp
        fn
        matched_ious
    """

    gt_boxes = [
        list(box)
        for box in gt_boxes
    ]

    pred_boxes = [
        list(box)
        for box in pred_boxes
    ]

    if not gt_boxes:
        return 0, len(pred_boxes), 0, []

    if not pred_boxes:
        return 0, 0, len(gt_boxes), []

    matches = []

    for pred_index, pred_box in enumerate(
        pred_boxes
    ):
        for gt_index, gt_box in enumerate(
            gt_boxes
        ):
            iou = compute_iou(
                pred_box,
                gt_box,
            )

            matches.append(
                (
                    iou,
                    pred_index,
                    gt_index,
                )
            )

    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_predictions = set()
    matched_gt = set()
    matched_ious = []

    for (
        iou,
        pred_index,
        gt_index,
    ) in matches:

        if iou < iou_threshold:
            break

        if pred_index in matched_predictions:
            continue

        if gt_index in matched_gt:
            continue

        matched_predictions.add(
            pred_index
        )

        matched_gt.add(
            gt_index
        )

        matched_ious.append(
            float(iou)
        )

    tp = len(matched_predictions)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return (
        tp,
        fp,
        fn,
        matched_ious,
    )


# ============================================================
# METRICS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def compute_binary_metrics(
    tp,
    tn,
    fp,
    fn,
):
    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    youden_j = (
        recall
        +
        specificity
        -
        1.0
    )

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "youden_j": float(youden_j),
    }


# ============================================================
# VISUALIZATION CLASS
# ============================================================

class DetectionAnalysisVisualizer:

    def __init__(
        self,
        checkpoint_path,
        backbone,
        device,
        output_dir,
        max_detections=DEFAULT_MAX_DETECTIONS,
        overlap_threshold=VISUALIZATION_OVERLAP_THRESHOLD,
        manual_threshold=None,
        num_flow_images=DEFAULT_NUM_FLOW_IMAGES,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.backbone_type = backbone
        self.device = device

        self.output_dir = Path(
            output_dir
        )

        self.max_detections = (
            max_detections
        )

        self.overlap_threshold = (
            overlap_threshold
        )

        self.manual_threshold = (
            manual_threshold
        )

        self.num_flow_images = (
            num_flow_images
        )

        # -----------------------------------------------------
        # Output structure
        # -----------------------------------------------------

        self.metrics_dir = (
            self.output_dir / "metrics"
        )

        self.tp_dir = (
            self.output_dir / "TP"
        )

        self.fp_dir = (
            self.output_dir / "FP"
        )

        self.tn_dir = (
            self.output_dir / "TN"
        )

        self.fn_dir = (
            self.output_dir / "FN"
        )

        self.feature_flow_dir = (
            self.output_dir
            / "feature_flow"
        )

        # Create everything immediately.
        for directory in (
            self.metrics_dir,
            self.tp_dir,
            self.fp_dir,
            self.tn_dir,
            self.fn_dir,
            self.feature_flow_dir,
        ):
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------

        self.dataset = (
            RSNAPneumoniaDataset(
                dcm_path=TRAIN_DCM_PATH,
                csv_path=CSV_PATH,
                transform=get_test_transforms(
                    IMAGE_SIZE
                ),
            )
        )

        (
            _,
            self.val_indices,
        ) = create_train_val_split(
            self.dataset,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        self.val_loader = (
            self.dataset.get_dataloader(
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=VAL_NUM_WORKERS,
                indices=self.val_indices,
            )
        )

        print(
            f"[LOG] Validation images: "
            f"{len(self.val_indices)}"
        )

        # -----------------------------------------------------
        # Backbone
        # -----------------------------------------------------

        if backbone == "chest_xray":

            backbone_path = (
                RESNET50_CHEST_XRAY_CHECKPOINT
            )

            print(
                "[LOG] Backbone: "
                "Chest-Xray pretrained ResNet-50"
            )

        else:

            backbone_path = None

            print(
                "[LOG] Backbone: "
                "ImageNet pretrained ResNet-50"
            )

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        self.model = (
            DetectionFramework(
                path_model=backbone_path,
            )
            .to(self.device)
        )

        # -----------------------------------------------------
        # Checkpoint
        # -----------------------------------------------------

        print(
            "[LOG] Loading checkpoint:"
        )

        print(
            f"      {self.checkpoint_path}"
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if (
            not isinstance(
                checkpoint,
                dict,
            )
            or
            "model_state_dict"
            not in checkpoint
        ):
            raise RuntimeError(
                "Invalid detector checkpoint."
            )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        del checkpoint

        self.model.eval()

        print(
            "[LOG] Detector checkpoint loaded."
        )

        # -----------------------------------------------------
        # Official postprocessor
        # -----------------------------------------------------

        self.official_postprocessor = (
            DetectionPostProcessor(
                score_threshold=(
                    OFFICIAL_SCORE_THRESHOLD
                ),
                nms_threshold=NMS_THRESHOLD,
            )
        )

        # -----------------------------------------------------
        # Calibration postprocessor
        # -----------------------------------------------------

        self.calibration_postprocessor = (
            DetectionPostProcessor(
                score_threshold=0.0,
                nms_threshold=NMS_THRESHOLD,
            )
        )

        self.tau = None
        self.visualization_threshold = None

    # ========================================================
    # Canvas validation
    # ========================================================

    @staticmethod
    def validate_canvas(
        image,
        gt_boxes,
        pred_boxes,
        expected_size,
        patient_id,
    ):
        if image.ndim == 3:
            height = image.shape[-2]
            width = image.shape[-1]

        elif image.ndim == 2:
            height = image.shape[-2]
            width = image.shape[-1]

        else:
            raise RuntimeError(
                f"[{patient_id}] "
                f"Unexpected image shape: "
                f"{tuple(image.shape)}"
            )

        if (
            width != expected_size
            or height != expected_size
        ):
            raise RuntimeError(
                "\n"
                f"[{patient_id}] IMAGE CANVAS ERROR\n"
                f"Expected: "
                f"{expected_size}x{expected_size}\n"
                f"Actual: "
                f"{width}x{height}\n"
            )

        for box in (
            gt_boxes + pred_boxes
        ):

            x1, y1, x2, y2 = box

            if not (
                0.0 <= x1 <= width
                and
                0.0 <= x2 <= width
                and
                0.0 <= y1 <= height
                and
                0.0 <= y2 <= height
            ):
                raise RuntimeError(
                    "\n"
                    f"[{patient_id}] "
                    f"BOX OUTSIDE CANVAS\n"
                    f"Canvas: "
                    f"{width}x{height}\n"
                    f"Box: "
                    f"{box}\n"
                )

            if x2 < x1 or y2 < y1:
                raise RuntimeError(
                    f"[{patient_id}] "
                    f"Invalid box: {box}"
                )

        return width, height

    # ========================================================
    # Image conversion
    # ========================================================

    @staticmethod
    def image_for_plot(
        image,
    ):
        """
        No resize is performed here.

        The image is already IMAGE_SIZE x IMAGE_SIZE.
        """

        image = (
            image
            .detach()
            .cpu()
        )

        if image.ndim == 3:
            image = image.permute(
                1,
                2,
                0,
            )

        image = image.numpy()

        if image.ndim == 3:
            image = image.mean(
                axis=2
            )

        return image

    # ========================================================
    # Threshold calibration
    # ========================================================

    def calibrate_threshold(
        self,
    ):
        if (
            self.manual_threshold
            is not None
        ):

            self.tau = (
                self.manual_threshold
            )

            self.visualization_threshold = (
                self.manual_threshold
            )

            print(
                "[CALIBRATION] "
                f"Manual threshold = "
                f"{self.visualization_threshold:.4f}"
            )

            return

        print()
        print(
            "=" * 80
        )

        print(
            "CALIBRATING DETECTION THRESHOLD"
        )

        print(
            "=" * 80
        )

        calibrator = (
            DetectionThresholdCalibrator(
                model=self.model,
                postprocessor=(
                    self.calibration_postprocessor
                ),
                device=self.device,
                criterion="youden",
                visualization_rule=False,
                verbose=True,
            )
        )

        result = calibrator.calibrate(
            dataloader=self.val_loader,
            num_thresholds=101,
            save_path=(
                self.metrics_dir
                / "threshold_calibration.csv"
            ),
        )

        self.tau = (
            result.threshold
        )

        # We use tau directly for visualization.
        self.visualization_threshold = (
            self.tau
        )

        print()
        print(
            "[LOG] tau* = "
            f"{self.tau:.4f}"
        )

        print(
            "[LOG] visualization threshold = "
            f"{self.visualization_threshold:.4f}"
        )

    # ========================================================
    # Filter predictions for visualization
    # ========================================================

    def filter_predictions(
        self,
        detections,
    ):
        boxes = detections[
            "boxes"
        ]

        scores = detections[
            "scores"
        ]

        keep = (
            scores
            >= self.visualization_threshold
        )

        boxes = boxes[keep]
        scores = scores[keep]

        (
            boxes,
            scores,
        ) = suppress_redundant_predictions(
            boxes=boxes,
            scores=scores,
            overlap_threshold=(
                self.overlap_threshold
            ),
            max_detections=(
                self.max_detections
            ),
        )

        return {
            "boxes": boxes,
            "scores": scores,
        }

    # ========================================================
    # Image-level category
    # ========================================================

    @staticmethod
    def classify_image(
        gt_boxes,
        detections,
    ):
        gt_positive = (
            len(gt_boxes) > 0
        )

        predicted_positive = (
            len(
                detections["boxes"]
            ) > 0
        )

        if gt_positive and predicted_positive:
            return "TP"

        if gt_positive and not predicted_positive:
            return "FN"

        if not gt_positive and predicted_positive:
            return "FP"

        return "TN"

    # ========================================================
    # FULL VALIDATION PASS
    # ========================================================

    @torch.no_grad()
    def collect_validation_results(
        self,
    ):
        """
        One complete validation pass.

        This phase DOES NOT save images.

        It collects all information required for:
            - metrics
            - confusion matrix
            - per-image CSV
            - later image export
        """

        results = []

        image_tp = 0
        image_tn = 0
        image_fp = 0
        image_fn = 0

        box_tp = 0
        box_fp = 0
        box_fn = 0

        total_images = 0
        total_gt_boxes = 0
        total_pred_boxes = 0

        matched_ious = []

        self.model.eval()

        total_batches = len(
            self.val_loader
        )

        print()
        print(
            "=" * 80
        )

        print(
            "FULL VALIDATION ANALYSIS"
        )

        print(
            "=" * 80
        )

        for batch_index, (
            images,
            targets,
        ) in enumerate(
            self.val_loader,
            start=1,
        ):

            images = images.to(
                self.device
            )

            predictions = (
                self.model(
                    images
                )
            )

            detections = (
                self.official_postprocessor(
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
                    target[
                        "boxes"
                    ]
                    .detach()
                    .cpu()
                    .float()
                )

                position = (
                    (
                        batch_index - 1
                    )
                    * BATCH_SIZE
                    + sample_index
                )

                if (
                    position
                    >= len(
                        self.val_indices
                    )
                ):
                    continue

                dataset_index = (
                    self.val_indices[
                        position
                    ]
                )

                patient_id = (
                    self.dataset
                    .image_paths[
                        dataset_index
                    ]
                    .stem
                )

                image_cpu = (
                    images[
                        sample_index
                    ]
                    .detach()
                    .cpu()
                )

                visual_detections = (
                    self.filter_predictions(
                        detection
                    )
                )

                gt_list = (
                    gt_boxes.tolist()
                )

                pred_boxes = (
                    visual_detections[
                        "boxes"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                pred_scores = (
                    visual_detections[
                        "scores"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                # ------------------------------------------------
                # Image-level
                # ------------------------------------------------

                category = (
                    self.classify_image(
                        gt_boxes,
                        visual_detections,
                    )
                )

                if category == "TP":
                    image_tp += 1

                elif category == "TN":
                    image_tn += 1

                elif category == "FP":
                    image_fp += 1

                elif category == "FN":
                    image_fn += 1

                # ------------------------------------------------
                # Box-level
                # ------------------------------------------------

                (
                    current_tp,
                    current_fp,
                    current_fn,
                    current_ious,
                ) = match_predictions_to_ground_truth(
                    gt_boxes=gt_list,
                    pred_boxes=pred_boxes,
                    iou_threshold=IOU_THRESHOLD,
                )

                box_tp += current_tp
                box_fp += current_fp
                box_fn += current_fn

                matched_ious.extend(
                    current_ious
                )

                max_score = (
                    max(pred_scores)
                    if pred_scores
                    else 0.0
                )

                best_iou = (
                    max(current_ious)
                    if current_ious
                    else 0.0
                )

                results.append(
                    {
                        "dataset_index": int(
                            dataset_index
                        ),
                        "patient_id": str(
                            patient_id
                        ),
                        "category": category,
                        "num_gt_boxes": len(
                            gt_list
                        ),
                        "num_predictions": len(
                            pred_boxes
                        ),
                        "max_score": float(
                            max_score
                        ),
                        "best_matched_iou": float(
                            best_iou
                        ),
                        "box_tp": int(
                            current_tp
                        ),
                        "box_fp": int(
                            current_fp
                        ),
                        "box_fn": int(
                            current_fn
                        ),
                        "image": image_cpu,
                        "gt_boxes": gt_boxes,
                        "detections": visual_detections,
                    }
                )

                total_images += 1

                total_gt_boxes += (
                    len(gt_list)
                )

                total_pred_boxes += (
                    len(pred_boxes)
                )

            if (
                batch_index % 100 == 0
                or batch_index == total_batches
            ):
                print(
                    "[ANALYSIS] "
                    f"batch={batch_index}/"
                    f"{total_batches} "
                    f"progress="
                    f"{100.0 * batch_index / total_batches:.1f}%"
                )

        return {
            "results": results,
            "image_tp": image_tp,
            "image_tn": image_tn,
            "image_fp": image_fp,
            "image_fn": image_fn,
            "box_tp": box_tp,
            "box_fp": box_fp,
            "box_fn": box_fn,
            "total_images": total_images,
            "total_gt_boxes": total_gt_boxes,
            "total_pred_boxes": total_pred_boxes,
            "matched_ious": matched_ious,
        }

    # ========================================================
    # Metrics
    # ========================================================

    def compute_metrics(
        self,
        analysis,
    ):
        box_metrics = (
            compute_binary_metrics(
                tp=analysis[
                    "box_tp"
                ],
                tn=0,
                fp=analysis[
                    "box_fp"
                ],
                fn=analysis[
                    "box_fn"
                ],
            )
        )

        image_metrics = (
            compute_binary_metrics(
                tp=analysis[
                    "image_tp"
                ],
                tn=analysis[
                    "image_tn"
                ],
                fp=analysis[
                    "image_fp"
                ],
                fn=analysis[
                    "image_fn"
                ],
            )
        )

        mean_best_iou = safe_divide(
            sum(
                analysis[
                    "matched_ious"
                ]
            ),
            len(
                analysis[
                    "matched_ious"
                ]
            ),
        )

        mean_predictions = safe_divide(
            analysis[
                "total_pred_boxes"
            ],
            analysis[
                "total_images"
            ],
        )

        return {
            "experiment": (
                self.checkpoint_path
                .parent
                .name
            ),
            "checkpoint": str(
                self.checkpoint_path
            ),
            "backbone": (
                self.backbone_type
            ),
            "image_size": int(
                IMAGE_SIZE
            ),
            "official_score_threshold": float(
                OFFICIAL_SCORE_THRESHOLD
            ),
            "nms_threshold": float(
                NMS_THRESHOLD
            ),
            "visualization_threshold": float(
                self.visualization_threshold
            ),
            "evaluation_iou_threshold": float(
                IOU_THRESHOLD
            ),
            "visualization_overlap_threshold": float(
                self.overlap_threshold
            ),
            "num_validation_images": int(
                analysis[
                    "total_images"
                ]
            ),
            "total_gt_boxes": int(
                analysis[
                    "total_gt_boxes"
                ]
            ),
            "total_visualization_predictions": int(
                analysis[
                    "total_pred_boxes"
                ]
            ),
            "mean_predictions_per_image": float(
                mean_predictions
            ),
            "mean_best_matched_iou": float(
                mean_best_iou
            ),
            "box_level": box_metrics,
            "image_level": image_metrics,
        }

    # ========================================================
    # Save metrics before image export
    # ========================================================

    def save_per_image_results(
        self,
        results,
    ):
        path = (
            self.metrics_dir
            / "per_image_results.csv"
        )

        fieldnames = [
            "dataset_index",
            "patient_id",
            "category",
            "num_gt_boxes",
            "num_predictions",
            "max_score",
            "best_matched_iou",
            "box_tp",
            "box_fp",
            "box_fn",
        ]

        with path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                writer.writerow(
                    {
                        key: result[key]
                        for key in fieldnames
                    }
                )

        print(
            f"[METRICS] Saved: {path}"
        )

    def save_metrics(
        self,
        metrics,
    ):
        json_path = (
            self.metrics_dir
            / "metrics.json"
        )

        csv_path = (
            self.metrics_dir
            / "metrics.csv"
        )

        with json_path.open(
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=2,
            )

        rows = []

        for section_name in (
            "box_level",
            "image_level",
        ):

            section = metrics[
                section_name
            ]

            for key, value in (
                section.items()
            ):
                rows.append(
                    {
                        "section": section_name,
                        "metric": key,
                        "value": value,
                    }
                )

        rows.extend(
            [
                {
                    "section": "global",
                    "metric": "tau",
                    "value": self.tau,
                },
                {
                    "section": "global",
                    "metric": (
                        "visualization_threshold"
                    ),
                    "value": (
                        self.visualization_threshold
                    ),
                },
                {
                    "section": "global",
                    "metric": (
                        "mean_predictions_per_image"
                    ),
                    "value": (
                        metrics[
                            "mean_predictions_per_image"
                        ]
                    ),
                },
                {
                    "section": "global",
                    "metric": (
                        "mean_best_matched_iou"
                    ),
                    "value": (
                        metrics[
                            "mean_best_matched_iou"
                        ]
                    ),
                },
            ]
        )

        with csv_path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "section",
                    "metric",
                    "value",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"[METRICS] Saved: {json_path}"
        )

        print(
            f"[METRICS] Saved: {csv_path}"
        )

    def save_confusion_matrix(
        self,
        metrics,
    ):
        image = metrics[
            "image_level"
        ]

        matrix = [
            [
                image["tn"],
                image["fp"],
            ],
            [
                image["fn"],
                image["tp"],
            ],
        ]

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            matrix
        )

        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )

        ax.set_xticklabels(
            [
                "Predicted Negative",
                "Predicted Positive",
            ]
        )

        ax.set_yticklabels(
            [
                "Actual Negative",
                "Actual Positive",
            ]
        )

        ax.set_xlabel(
            "Prediction"
        )

        ax.set_ylabel(
            "Ground Truth"
        )

        ax.set_title(
            "Image-Level Confusion Matrix"
        )

        labels = [
            [
                "TN",
                "FP",
            ],
            [
                "FN",
                "TP",
            ],
        ]

        values = [
            [
                image["tn"],
                image["fp"],
            ],
            [
                image["fn"],
                image["tp"],
            ],
        ]

        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    (
                        f"{labels[row][col]}\n"
                        f"{values[row][col]}"
                    ),
                    ha="center",
                    va="center",
                    fontsize=13,
                )

        path = (
            self.metrics_dir
            / "confusion_matrix.png"
        )

        fig.tight_layout()

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"[METRICS] Saved: {path}"
        )

    # ========================================================
    # Feature flow
    # ========================================================

    @staticmethod
    def reduce_feature_map(
        feature,
    ):
        return (
            feature
            .detach()
            .float()
            .abs()
            .mean(dim=1)[0]
            .cpu()
        )

    @torch.no_grad()
    def plot_feature_flow(
        self,
        image,
        patient_id,
        output_path,
    ):
        image_np = (
            self.image_for_plot(
                image
            )
        )

        if image_np.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError(
                f"[{patient_id}] "
                f"Feature flow image has shape "
                f"{image_np.shape}"
            )

        image_batch = (
            image
            .unsqueeze(0)
            .to(self.device)
        )

        backbone_output = (
            self.model
            .fpn
            .backbone(
                image_batch
            )
        )

        C2, C3, C4, C5 = (
            backbone_output
        )

        fpn_output = (
            self.model
            .fpn(
                image_batch
            )
        )

        P3, P4, P5, P6, P7 = (
            fpn_output
        )

        features = [
            ("C2", C2),
            ("C3", C3),
            ("C4", C4),
            ("C5", C5),
            ("P3", P3),
            ("P4", P4),
            ("P5", P5),
            ("P6", P6),
            ("P7", P7),
        ]

        count = 1 + len(features)
        columns = 5
        rows = math.ceil(
            count / columns
        )

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(
                4 * columns,
                4 * rows,
            ),
        )

        axes = axes.flatten()

        axes[0].imshow(
            image_np,
            cmap="gray",
        )

        axes[0].set_title(
            f"Input\n{IMAGE_SIZE}x{IMAGE_SIZE}"
        )

        axes[0].axis("off")

        for index, (
            name,
            feature,
        ) in enumerate(
            features,
            start=1,
        ):

            activation = (
                self.reduce_feature_map(
                    feature
                )
            )

            axes[index].imshow(
                activation,
                cmap="magma",
            )

            axes[index].set_title(
                (
                    f"{name}\n"
                    f"{tuple(feature.shape)}"
                )
            )

            axes[index].axis("off")

        for index in range(
            count,
            len(axes),
        ):
            axes[index].axis("off")

        fig.suptitle(
            f"Feature flow: {patient_id}"
        )

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=IMAGE_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    def save_feature_flow_examples(
        self,
        results,
    ):
        tp_results = [
            result
            for result in results
            if result["category"] == "TP"
        ]

        selected = tp_results[
            :self.num_flow_images
        ]

        if len(selected) < (
            self.num_flow_images
        ):
            remaining = [
                result
                for result in results
                if result["category"] != "TP"
            ]

            selected.extend(
                remaining[
                    :(
                        self.num_flow_images
                        - len(selected)
                    )
                ]
            )

        print()
        print(
            "[IMAGES] Saving feature-flow "
            f"examples: {len(selected)}"
        )

        for index, result in enumerate(
            selected,
            start=1,
        ):

            output_path = (
                self.feature_flow_dir
                /
                (
                    f"{index:02d}_"
                    f"{result['category']}_"
                    f"{result['patient_id']}.png"
                )
            )

            self.plot_feature_flow(
                image=result["image"],
                patient_id=(
                    result["patient_id"]
                ),
                output_path=output_path,
            )

    # ========================================================
    # Prediction image
    # ========================================================

    def save_prediction_image(
        self,
        result,
        output_path,
    ):
        """
        GT boxes:
            GREEN

        Prediction boxes:
            RED
        """

        image = result["image"]

        patient_id = result[
            "patient_id"
        ]

        category = result[
            "category"
        ]

        gt_boxes = (
            result["gt_boxes"]
            .detach()
            .cpu()
            .tolist()
        )

        pred_boxes = (
            result["detections"]["boxes"]
            .detach()
            .cpu()
            .tolist()
        )

        pred_scores = (
            result["detections"]["scores"]
            .detach()
            .cpu()
            .tolist()
        )

        image_np = (
            self.image_for_plot(
                image
            )
        )

        width, height = (
            self.validate_canvas(
                image=image,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                expected_size=IMAGE_SIZE,
                patient_id=patient_id,
            )
        )

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            image_np,
            cmap="gray",
        )

        # ====================================================
        # GT = GREEN
        # ====================================================

        for index, box in enumerate(
            gt_boxes,
            start=1,
        ):

            x1, y1, x2, y2 = box

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2.5,
                edgecolor="green",
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                max(
                    4,
                    y1 - 4,
                ),
                f"GT #{index}",
                color="green",
                fontsize=7,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "pad": 1.5,
                },
            )

        # ====================================================
        # PREDICTIONS = RED
        # ====================================================

        for index, (
            box,
            score,
        ) in enumerate(
            zip(
                pred_boxes,
                pred_scores,
            ),
            start=1,
        ):

            x1, y1, x2, y2 = box

            best_iou = (
                best_iou_against_gt(
                    box,
                    gt_boxes,
                )
            )

            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.6,
                edgecolor="red",
            )

            ax.add_patch(rect)

            label = (
                f"P#{index} "
                f"s={score:.2f}"
            )

            if gt_boxes:
                label += (
                    f" IoU={best_iou:.2f}"
                )

            ax.text(
                x1,
                min(
                    height - 4,
                    y2 + 10,
                ),
                label,
                color="red",
                fontsize=6,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.70,
                    "pad": 1,
                },
            )

        ax.set_title(
            (
                f"{patient_id} | {category}\n"
                f"GT={len(gt_boxes)} | "
                f"Pred={len(pred_boxes)} | "
                f"threshold="
                f"{self.visualization_threshold:.3f}"
            ),
            fontsize=9,
        )

        ax.set_xlim(
            0,
            width,
        )

        ax.set_ylim(
            height,
            0,
        )

        ax.axis("off")

        fig.savefig(
            output_path,
            dpi=IMAGE_DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # SAVE ALL IMAGES -- THIS IS LAST
    # ========================================================

    def save_all_prediction_images(
        self,
        results,
    ):
        """
        Potentially long-running phase.

        Metrics have already been saved before this starts.

        This can safely be interrupted.
        """

        print()
        print(
            "=" * 80
        )

        print(
            "SAVING ALL TP / FP / TN / FN IMAGES"
        )

        print(
            "=" * 80
        )

        category_dirs = {
            "TP": self.tp_dir,
            "FP": self.fp_dir,
            "TN": self.tn_dir,
            "FN": self.fn_dir,
        }

        counters = {
            "TP": 0,
            "FP": 0,
            "TN": 0,
            "FN": 0,
        }

        total = len(
            results
        )

        for index, result in enumerate(
            results,
            start=1,
        ):

            category = (
                result["category"]
            )

            counters[
                category
            ] += 1

            output_path = (
                category_dirs[
                    category
                ]
                /
                (
                    f"{counters[category]:05d}_"
                    f"{result['patient_id']}_"
                    f"{category}.jpg"
                )
            )

            self.save_prediction_image(
                result=result,
                output_path=output_path,
            )

            if (
                index % 100 == 0
                or index == total
            ):

                print(
                    "[IMAGES] "
                    f"{index}/{total} "
                    f"saved | "
                    f"TP={counters['TP']} "
                    f"FP={counters['FP']} "
                    f"TN={counters['TN']} "
                    f"FN={counters['FN']}"
                )

        print()
        print(
            "[IMAGES] Image export completed."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def print_summary(
        self,
        metrics,
    ):
        box = metrics[
            "box_level"
        ]

        image = metrics[
            "image_level"
        ]

        print()
        print(
            "=" * 80
        )

        print(
            "METRICS COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"tau*:                   "
            f"{self.tau:.4f}"
        )

        print(
            f"Visualization threshold:"
            f" {self.visualization_threshold:.4f}"
        )

        print()
        print(
            "BOX-LEVEL"
        )

        print(
            f"  TP:         "
            f"{box['tp']}"
        )

        print(
            f"  FP:         "
            f"{box['fp']}"
        )

        print(
            f"  FN:         "
            f"{box['fn']}"
        )

        print(
            f"  Precision:  "
            f"{box['precision']:.6f}"
        )

        print(
            f"  Recall:     "
            f"{box['recall']:.6f}"
        )

        print(
            f"  F1:         "
            f"{box['f1']:.6f}"
        )

        print()
        print(
            "IMAGE-LEVEL"
        )

        print(
            f"  TP:         "
            f"{image['tp']}"
        )

        print(
            f"  TN:         "
            f"{image['tn']}"
        )

        print(
            f"  FP:         "
            f"{image['fp']}"
        )

        print(
            f"  FN:         "
            f"{image['fn']}"
        )

        print(
            f"  Accuracy:   "
            f"{image['accuracy']:.6f}"
        )

        print(
            f"  Precision:  "
            f"{image['precision']:.6f}"
        )

        print(
            f"  Recall:     "
            f"{image['recall']:.6f}"
        )

        print(
            f"  Specificity:"
            f"{image['specificity']:.6f}"
        )

        print(
            f"  F1:         "
            f"{image['f1']:.6f}"
        )

        print()
        print(
            f"Mean predictions/image: "
            f"{metrics['mean_predictions_per_image']:.4f}"
        )

        print(
            f"Mean best matched IoU:   "
            f"{metrics['mean_best_matched_iou']:.6f}"
        )

        print()
        print(
            f"Metrics directory: "
            f"{self.metrics_dir}"
        )

        print(
            "=" * 80
        )

    # ========================================================
    # FULL WORKFLOW
    # ========================================================

    def run(
        self,
    ):
        print()
        print(
            "=" * 80
        )

        print(
            "DETECTOR VALIDATION ANALYSIS"
        )

        print(
            "=" * 80
        )

        # ====================================================
        # PHASE 1
        # Calibration
        # ====================================================

        self.calibrate_threshold()

        # ====================================================
        # PHASE 2
        # Full inference / metrics
        #
        # NO IMAGE FILES ARE WRITTEN HERE.
        # ====================================================

        analysis = (
            self.collect_validation_results()
        )

        # ====================================================
        # PHASE 3
        # Save metrics FIRST
        # ====================================================

        metrics = (
            self.compute_metrics(
                analysis
            )
        )

        self.save_per_image_results(
            analysis["results"]
        )

        self.save_metrics(
            metrics
        )

        self.save_confusion_matrix(
            metrics
        )

        self.print_summary(
            metrics
        )

        print()
        print(
            "[LOG] Metrics are safely saved."
        )

        print(
            "[LOG] The next phase only writes images."
        )

        print(
            "[LOG] You can stop the process during "
            "image export without losing the metrics."
        )

        # ====================================================
        # PHASE 4
        # Feature flow
        # ====================================================

        self.save_feature_flow_examples(
            analysis["results"]
        )

        # ====================================================
        # PHASE 5
        # ALL TP / FP / TN / FN IMAGES
        #
        # This is intentionally LAST.
        # ====================================================

        self.save_all_prediction_images(
            analysis["results"]
        )

        print()
        print(
            "=" * 80
        )

        print(
            "FULL ANALYSIS COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            f"Output: "
            f"{self.output_dir}"
        )

        print(
            "=" * 80
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Complete detector validation analysis."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--backbone",
        required=True,
        choices=[
            "chest_xray",
            "imagenet",
        ],
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
    )

    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=(
            VISUALIZATION_OVERLAP_THRESHOLD
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Manual visualization threshold. "
            "If omitted, Youden calibration is used."
        ),
    )

    parser.add_argument(
        "--num-flow-images",
        type=int,
        default=DEFAULT_NUM_FLOW_IMAGES,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Automatic output directory
    # --------------------------------------------------------

    if args.output is None:

        checkpoint_parts = (
            Path(
                args.checkpoint
            ).parts
        )

        experiment_name = (
            "experiment"
        )

        if (
            "checkpoints"
            in checkpoint_parts
        ):

            index = (
                checkpoint_parts.index(
                    "checkpoints"
                )
            )

            if (
                index + 1
                < len(checkpoint_parts)
            ):
                experiment_name = (
                    checkpoint_parts[
                        index + 1
                    ]
                )

        output_dir = (
            Path("visualization")
            / experiment_name
        )

    else:

        output_dir = Path(
            args.output
        )

    visualizer = (
        DetectionAnalysisVisualizer(
            checkpoint_path=(
                args.checkpoint
            ),
            backbone=(
                args.backbone
            ),
            device=device,
            output_dir=output_dir,
            max_detections=(
                args.max_detections
            ),
            overlap_threshold=(
                args.overlap_threshold
            ),
            manual_threshold=(
                args.threshold
            ),
            num_flow_images=(
                args.num_flow_images
            ),
        )
    )

    visualizer.run()


if __name__ == "__main__":
    main()