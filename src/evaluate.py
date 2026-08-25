import time

import torch

from src.metrics import compute_metrics


class DetectionEvaluator:

    def __init__(
        self,
        model,
        postprocessor,
        device,
    ):
        self.model = model
        self.postprocessor = postprocessor
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        max_batches=None,
    ):
        """
        Run detection evaluation.

        Args:
            dataloader:
                Validation DataLoader.

            max_batches:
                Maximum number of batches to evaluate.
                None -> evaluate the complete validation set.

        Returns:
            Dataset-level detection metrics.
        """

        self.model.eval()

        all_predictions = []
        all_targets = []

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        total_images = 0
        images_with_detections = 0
        total_detections = 0
        max_detection_score = 0.0

        start_time = time.perf_counter()

        total_batches = (
            max_batches
            if max_batches is not None
            else len(dataloader)
        )

        print()
        print("=" * 70)
        print("[VAL] Starting detection evaluation")
        print("=" * 70)

        print(
            f"[VAL] Batches:         {total_batches}"
        )

        print(
            f"[VAL] Max batches:     {max_batches}"
        )

        print(
            f"[VAL] Device:          {self.device}"
        )

        # ---------------------------------------------------------
        # Validation loop
        # ---------------------------------------------------------

        for batch_idx, (
            images,
            targets,
        ) in enumerate(
            dataloader,
            start=1,
        ):

            # Optional quick validation.
            if (
                max_batches is not None
                and batch_idx > max_batches
            ):
                break

            images = images.to(
                self.device
            )

            # -----------------------------------------------------
            # Forward
            # -----------------------------------------------------

            predictions = self.model(
                images
            )

            # -----------------------------------------------------
            # Post-processing
            # -----------------------------------------------------

            detections = self.postprocessor(
                predictions
            )

            # -----------------------------------------------------
            # Store predictions
            # -----------------------------------------------------

            all_predictions.extend(
                detections
            )

            # -----------------------------------------------------
            # Diagnostics
            # -----------------------------------------------------

            for detection in detections:

                total_images += 1

                num_detections = (
                    detection["boxes"].shape[0]
                )

                total_detections += (
                    num_detections
                )

                if num_detections > 0:

                    images_with_detections += 1

                    batch_max_score = (
                        detection["scores"]
                        .max()
                        .item()
                    )

                    max_detection_score = max(
                        max_detection_score,
                        batch_max_score,
                    )

            # -----------------------------------------------------
            # Store GT targets
            # -----------------------------------------------------

            for target in targets:

                all_targets.append(
                    {
                        "boxes": (
                            target["boxes"]
                            .cpu()
                        ),
                        "labels": (
                            target["labels"]
                            .cpu()
                        ),
                    }
                )

            # -----------------------------------------------------
            # Progress logging
            # -----------------------------------------------------

            if (
                batch_idx % 100 == 0
                or batch_idx == total_batches
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                progress = (
                    100.0
                    * batch_idx
                    / total_batches
                )

                current_avg = (
                    total_detections
                    / total_images
                    if total_images > 0
                    else 0.0
                )

                print(
                    f"[VAL] "
                    f"batch={batch_idx}/"
                    f"{total_batches} "
                    f"progress={progress:.1f}% "
                    f"time={elapsed / 60.0:.2f} min "
                    f"detections/img={current_avg:.2f}"
                )

        # ---------------------------------------------------------
        # Diagnostics summary
        # ---------------------------------------------------------

        if total_images > 0:

            average_detections = (
                total_detections
                / total_images
            )

            detection_image_ratio = (
                images_with_detections
                / total_images
            )

        else:

            average_detections = 0.0
            detection_image_ratio = 0.0

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print()
        print(
            "=" * 70
        )

        print(
            "[VAL] Inference + post-processing completed"
        )

        print(
            "=" * 70
        )

        print(
            f"[VAL] Images evaluated:       "
            f"{total_images}"
        )

        print(
            f"[VAL] Images with detections: "
            f"{images_with_detections} "
            f"({100.0 * detection_image_ratio:.2f}%)"
        )

        print(
            f"[VAL] Total detections:       "
            f"{total_detections}"
        )

        print(
            f"[VAL] Average detections/img: "
            f"{average_detections:.4f}"
        )

        print(
            f"[VAL] Maximum detection score:"
            f" {max_detection_score:.6f}"
        )

        print(
            f"[VAL] Inference time:          "
            f"{elapsed / 60.0:.2f} min"
        )

        print(
            "-" * 70
        )

        # ---------------------------------------------------------
        # Dataset-level metrics
        # ---------------------------------------------------------

        print(
            "[VAL] Starting metric computation..."
        )

        metrics_start = (
            time.perf_counter()
        )

        metrics = compute_metrics(
            all_predictions,
            all_targets,
        )

        metrics_elapsed = (
            time.perf_counter()
            - metrics_start
        )

        print(
            "[VAL] Metric computation completed"
        )

        print(
            f"[VAL] Metrics time:            "
            f"{metrics_elapsed / 60.0:.2f} min"
        )

        print(
            "-" * 70
        )

        if isinstance(
            metrics,
            dict,
        ):

            for (
                name,
                value,
            ) in metrics.items():

                if isinstance(
                    value,
                    (int, float),
                ):
                    print(
                        f"[VAL] "
                        f"{name:<10}: "
                        f"{value:.6f}"
                    )

        total_elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[VAL] Total evaluation time: "
            f"{total_elapsed / 60.0:.2f} min"
        )

        print(
            "=" * 70
        )

        return metrics