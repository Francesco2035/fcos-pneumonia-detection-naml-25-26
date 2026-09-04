from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

from src.config import IMAGE_SIZE

from src.analysis.geometry import (
    best_iou_against_gt,
)


class DetectionVisualizer:
    """
    Generates visual outputs for detector analysis.

    This class is responsible only for visualization and image export.
    Quantitative analysis and prediction matching are handled separately
    by DetectionAnalyzer.
    """

    def __init__(
        self,
        output_dir,
        num_flow_images=6,
        image_dpi=110,
    ):
        self.output_dir = Path(
            output_dir
        )

        self.num_flow_images = (
            num_flow_images
        )

        self.image_dpi = image_dpi

        # -----------------------------------------------------
        # Output directories
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
            self.output_dir / "feature_flow"
        )

        # Create directories used by the visualizer.
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

    # =========================================================
    # Image conversion
    # =========================================================

    @staticmethod
    def image_for_plot(
        image,
    ):
        """
        Convert a tensor image into a NumPy array for plotting.

        No resizing is performed because the image is already
        expected to have IMAGE_SIZE x IMAGE_SIZE resolution.
        """

        image = (
            image
            .detach()
            .cpu()
        )

        # Convert CHW -> HWC when needed.
        if image.ndim == 3:
            image = image.permute(
                1,
                2,
                0,
            )

        image = image.numpy()

        # Convert multi-channel input into grayscale.
        if image.ndim == 3:
            image = image.mean(
                axis=2
            )

        return image

    # =========================================================
    # Canvas validation
    # =========================================================

    @staticmethod
    def validate_canvas(
        image,
        gt_boxes,
        pred_boxes,
        expected_size,
        patient_id,
    ):
        """
        Validate image and bounding-box coordinates before plotting.
        """

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

        # Ensure the image uses the expected canvas size.
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

        # Ensure all boxes are inside the image.
        for box in (
            gt_boxes + pred_boxes
        ):
            x1, y1, x2, y2 = box

            if not (
                0.0 <= x1 <= width
                and 0.0 <= x2 <= width
                and 0.0 <= y1 <= height
                and 0.0 <= y2 <= height
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

            # Check coordinate ordering.
            if x2 < x1 or y2 < y1:
                raise RuntimeError(
                    f"[{patient_id}] "
                    f"Invalid box: {box}"
                )

        return width, height

    # =========================================================
    # Confusion matrix
    # =========================================================

    def save_confusion_matrix(
        self,
        metrics,
    ):
        """
        Save the image-level confusion matrix as a PNG.
        """

        image_metrics = metrics[
            "image_level"
        ]

        matrix = [
            [
                image_metrics["tn"],
                image_metrics["fp"],
            ],
            [
                image_metrics["fn"],
                image_metrics["tp"],
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
                image_metrics["tn"],
                image_metrics["fp"],
            ],
            [
                image_metrics["fn"],
                image_metrics["tp"],
            ],
        ]

        # Write the count inside each matrix cell.
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

    # =========================================================
    # Feature flow
    # =========================================================

    @staticmethod
    def reduce_feature_map(
        feature,
    ):
        """
        Reduce a feature tensor to one 2D activation map for visualization.

        Absolute activation values are averaged across channels.
        """

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
        model,
        image,
        patient_id,
        output_path,
    ):
        """
        Visualize feature maps produced by the backbone and FPN.
        """

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
            .to(
                next(
                    model.parameters()
                ).device
            )
        )

        # Extract backbone features.
        backbone_output = (
            model.fpn.backbone(
                image_batch
            )
        )

        C2, C3, C4, C5 = (
            backbone_output
        )

        # Extract FPN features.
        fpn_output = (
            model.fpn(
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

        count = (
            1 + len(features)
        )

        columns = 5

        rows = (
            math.ceil(
                count / columns
            )
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

        # Original input image.
        axes[0].imshow(
            image_np,
            cmap="gray",
        )

        axes[0].set_title(
            f"Input\n{IMAGE_SIZE}x{IMAGE_SIZE}"
        )

        axes[0].axis("off")

        # Feature maps.
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

        # Hide unused subplot slots.
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
            dpi=self.image_dpi,
            bbox_inches="tight",
        )

        plt.close(fig)

    def save_feature_flow_examples(
        self,
        model,
        results,
    ):
        """
        Save feature-flow visualizations for a small set of validation images.

        True positives are preferred when available.
        """

        tp_results = [
            result
            for result in results
            if result["category"] == "TP"
        ]

        selected = tp_results[
            :self.num_flow_images
        ]

        # Fill remaining slots with other examples.
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
                model=model,
                image=result["image"],
                patient_id=(
                    result["patient_id"]
                ),
                output_path=output_path,
            )

    # =========================================================
    # Plotting-safe box clipping
    # =========================================================

    @staticmethod
    def clip_boxes_to_canvas(
        boxes,
        width,
        height,
        patient_id,
        box_type,
    ):
        """
        Clip box coordinates to the image canvas for visualization.

        Detection/metric computation is unaffected: clipping is performed
        only on the local coordinates used for drawing. Boxes that collapse
        to zero/negative area after clipping are skipped.
        """
        clipped_boxes = []
        changed = False

        for box in boxes:
            x1, y1, x2, y2 = [
                float(value)
                for value in box
            ]

            original = (
                x1,
                y1,
                x2,
                y2,
            )

            # Normalize coordinate ordering before clipping.
            if x2 < x1:
                x1, x2 = x2, x1

            if y2 < y1:
                y1, y2 = y2, y1

            x1 = min(
                max(x1, 0.0),
                float(width),
            )
            x2 = min(
                max(x2, 0.0),
                float(width),
            )
            y1 = min(
                max(y1, 0.0),
                float(height),
            )
            y2 = min(
                max(y2, 0.0),
                float(height),
            )

            clipped = (
                x1,
                y1,
                x2,
                y2,
            )

            if clipped != original:
                changed = True
                print(
                    "[VISUALIZER] "
                    f"{patient_id}: clipped {box_type} box "
                    f"{original} -> {clipped}"
                )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                print(
                    "[VISUALIZER] "
                    f"{patient_id}: skipped degenerate "
                    f"{box_type} box after clipping: "
                    f"{clipped}"
                )
                continue

            clipped_boxes.append(
                clipped
            )

        return (
            clipped_boxes,
            changed,
        )

    # =========================================================
    # Prediction image
    # =========================================================

    def save_prediction_image(
        self,
        result,
        output_path,
        threshold,
    ):
        """
        Save an image with ground-truth and predicted bounding boxes.

        Ground-truth boxes are shown in green and predictions in red.
        """

        image = result["image"]

        patient_id = (
            result["patient_id"]
        )

        category = (
            result["category"]
        )

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

        # Validate the image itself. Bounding-box coordinates are made
        # plotting-safe below rather than aborting the whole export.
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
            width != IMAGE_SIZE
            or height != IMAGE_SIZE
        ):
            raise RuntimeError(
                "\n"
                f"[{patient_id}] IMAGE CANVAS ERROR\n"
                f"Expected: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
                f"Actual: {width}x{height}\n"
            )

        gt_boxes, gt_changed = (
            self.clip_boxes_to_canvas(
                boxes=gt_boxes,
                width=width,
                height=height,
                patient_id=patient_id,
                box_type="ground-truth",
            )
        )

        pred_boxes, pred_changed = (
            self.clip_boxes_to_canvas(
                boxes=pred_boxes,
                width=width,
                height=height,
                patient_id=patient_id,
                box_type="prediction",
            )
        )

        if gt_changed or pred_changed:
            print(
                "[VISUALIZER] "
                f"{patient_id}: box coordinates adjusted "
                "for plotting only."
            )

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            image_np,
            cmap="gray",
        )

        # -----------------------------------------------------
        # Ground-truth boxes
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Predictions
        # -----------------------------------------------------

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
                f"threshold={threshold:.3f}"
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
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

    # =========================================================
    # Save all prediction images
    # =========================================================

    def save_all_prediction_images(
        self,
        results,
        threshold,
    ):
        """
        Export all validation images grouped by TP, FP, TN and FN.

        Metrics have already been computed before this function is called.
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
                threshold=threshold,
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