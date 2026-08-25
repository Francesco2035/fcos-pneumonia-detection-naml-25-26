from pathlib import Path
import argparse
import csv
import json
import time

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle


from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
    VAL_RATIO,
    SEED,
    SCORE_THRESHOLD,
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


# ============================================================
# Default configuration
# ============================================================

DEFAULT_CHECKPOINT = (
    "checkpoints/exp8/best.pt"
)

DEFAULT_EXPERIMENT = "exp8"

OUTPUT_ROOT = Path(
    "visualization"
)

IOU_THRESHOLD = 0.50

# ------------------------------------------------------------
# Visualization filtering only
#
# IMPORTANT:
# Official inference remains:
#
#     SCORE_THRESHOLD = 0.10
#     NMS_THRESHOLD   = 0.50
#
# This threshold is only used to make saved figures readable.
# ------------------------------------------------------------

VISUALIZATION_THRESHOLD = 0.15

VISUALIZATION_OVERLAP_THRESHOLD = 0.40

MAX_VISUALIZATIONS = 10

# ------------------------------------------------------------
# Number of example images saved per category.
#
# Set to None to save every image.
# ------------------------------------------------------------

DEFAULT_MAX_IMAGES_PER_CATEGORY = 20


# ============================================================
# IoU
# ============================================================

def compute_iou(
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

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return (
        intersection
        / union
    )


# ============================================================
# Overlap relative to smaller box
# ============================================================

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

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    smaller_area = min(
        area_a,
        area_b,
    )

    if smaller_area <= 0.0:
        return 0.0

    return (
        intersection
        / smaller_area
    )


# ============================================================
# Visualization-only redundant box suppression
# ============================================================

def suppress_redundant_predictions(
    boxes,
    scores,
    overlap_threshold=0.40,
    max_detections=10,
):
    """
    Visualization-only suppression.

    Predictions are sorted by score.
    A lower-score prediction is removed when the intersection
    covers >= overlap_threshold of the smaller box.

    This DOES NOT affect official evaluation.
    """

    if boxes.numel() == 0:
        return (
            boxes,
            scores,
        )

    order = torch.argsort(
        scores,
        descending=True,
    )

    boxes_sorted = boxes[
        order
    ]

    scores_sorted = scores[
        order
    ]

    boxes_cpu = (
        boxes_sorted
        .detach()
        .cpu()
        .tolist()
    )

    kept_indices = []

    for candidate_index in range(
        len(boxes_cpu)
    ):

        candidate = boxes_cpu[
            candidate_index
        ]

        redundant = False

        for kept_index in kept_indices:

            kept_box = boxes_cpu[
                kept_index
            ]

            overlap = (
                compute_overlap_over_smaller_area(
                    candidate,
                    kept_box,
                )
            )

            if overlap >= overlap_threshold:
                redundant = True
                break

        if not redundant:

            kept_indices.append(
                candidate_index
            )

        if len(kept_indices) >= max_detections:
            break

    keep = torch.tensor(
        kept_indices,
        dtype=torch.long,
        device=boxes.device,
    )

    return (
        boxes_sorted[keep],
        scores_sorted[keep],
    )


# ============================================================
# Matching predictions to GT
# ============================================================

def match_predictions_to_ground_truth(
    prediction_boxes,
    prediction_scores,
    ground_truth_boxes,
    iou_threshold=0.50,
):
    """
    Greedy score-ordered matching.

    A prediction is TP when it matches one previously unmatched
    GT box with IoU >= threshold.

    Remaining predictions are FP.
    Remaining GT boxes are FN.

    Returns:
        matches
        prediction_is_tp
        num_tp
        num_fp
        num_fn
    """

    prediction_boxes = list(
        prediction_boxes
    )

    prediction_scores = list(
        prediction_scores
    )

    ground_truth_boxes = list(
        ground_truth_boxes
    )

    num_predictions = len(
        prediction_boxes
    )

    num_gt = len(
        ground_truth_boxes
    )

    if num_predictions == 0:

        return (
            [],
            [],
            0,
            0,
            num_gt,
        )

    if num_gt == 0:

        return (
            [
                None
                for _ in prediction_boxes
            ],
            [
                False
                for _ in prediction_boxes
            ],
            0,
            num_predictions,
            0,
        )

    # --------------------------------------------------------
    # Process predictions from highest score to lowest.
    # --------------------------------------------------------

    order = sorted(
        range(num_predictions),
        key=lambda index: (
            prediction_scores[index]
        ),
        reverse=True,
    )

    matched_gt = set()

    matches = [
        None
        for _ in prediction_boxes
    ]

    prediction_is_tp = [
        False
        for _ in prediction_boxes
    ]

    for prediction_index in order:

        prediction_box = (
            prediction_boxes[
                prediction_index
            ]
        )

        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt_box in enumerate(
            ground_truth_boxes
        ):

            if gt_index in matched_gt:
                continue

            iou = compute_iou(
                prediction_box,
                gt_box,
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            matched_gt.add(
                best_gt_index
            )

            matches[
                prediction_index
            ] = {
                "gt_index": best_gt_index,
                "iou": best_iou,
            }

            prediction_is_tp[
                prediction_index
            ] = True

    num_tp = sum(
        prediction_is_tp
    )

    num_fp = (
        num_predictions
        - num_tp
    )

    num_fn = (
        num_gt
        - num_tp
    )

    return (
        matches,
        prediction_is_tp,
        num_tp,
        num_fp,
        num_fn,
    )


# ============================================================
# Image-level classification
# ============================================================

def classify_image(
    ground_truth_boxes,
    prediction_boxes,
):
    """
    Image-level binary classification:

        GT positive:
            at least one GT box

        Prediction positive:
            at least one detection

    Returns:
        "TP", "TN", "FP", or "FN"
    """

    gt_positive = (
        len(ground_truth_boxes) > 0
    )

    prediction_positive = (
        len(prediction_boxes) > 0
    )

    if gt_positive and prediction_positive:
        return "TP"

    if not gt_positive and not prediction_positive:
        return "TN"

    if not gt_positive and prediction_positive:
        return "FP"

    return "FN"


# ============================================================
# Filter for visualization
# ============================================================

def prepare_visualization_detections(
    detections,
):
    """
    Keep official inference untouched.

    For qualitative figures only:
        score >= VISUALIZATION_THRESHOLD
        redundant-box suppression
        maximum MAX_VISUALIZATIONS
    """

    boxes = detections[
        "boxes"
    ]

    scores = detections[
        "scores"
    ]

    labels = detections[
        "labels"
    ]

    keep = (
        scores
        >= VISUALIZATION_THRESHOLD
    )

    boxes = boxes[
        keep
    ]

    scores = scores[
        keep
    ]

    labels = labels[
        keep
    ]

    (
        boxes,
        scores,
    ) = suppress_redundant_predictions(
        boxes=boxes,
        scores=scores,
        overlap_threshold=(
            VISUALIZATION_OVERLAP_THRESHOLD
        ),
        max_detections=MAX_VISUALIZATIONS,
    )

    labels = torch.ones(
        boxes.shape[0],
        dtype=torch.long,
        device=boxes.device,
    )

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


# ============================================================
# Load model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):
    print(
        "[LOG] Loading detector..."
    )

    model = DetectionFramework(
        path_model=(
            RESNET50_CHEST_XRAY_CHECKPOINT
        ),
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
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

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    del checkpoint

    model.eval()

    print(
        "[LOG] Detector checkpoint loaded:"
    )

    print(
        f"      {checkpoint_path}"
    )

    return model


# ============================================================
# Plot image
# ============================================================

def plot_detection_image(
    image,
    patient_id,
    ground_truth_boxes,
    detections,
    output_path,
    image_category,
):
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

    gt_boxes = (
        ground_truth_boxes
        .tolist()
    )

    pred_boxes = (
        detections["boxes"]
        .detach()
        .cpu()
        .tolist()
    )

    pred_scores = (
        detections["scores"]
        .detach()
        .cpu()
        .tolist()
    )

    height, width = (
        image.shape[:2]
    )

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.imshow(
        image,
        cmap="gray",
    )

    # ========================================================
    # Ground truth
    # ========================================================

    for index, box in enumerate(
        gt_boxes,
        start=1,
    ):

        x1, y1, x2, y2 = box

        rect = Rectangle(
            (
                x1,
                y1,
            ),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=3,
            edgecolor="lime",
        )

        ax.add_patch(
            rect
        )

        ax.text(
            x1,
            max(
                5,
                y1 - 5,
            ),
            f"GT #{index}",
            color="lime",
            fontsize=9,
            bbox=dict(
                facecolor="black",
                alpha=0.7,
                pad=2,
            ),
        )

    # ========================================================
    # Predictions
    # ========================================================

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

        best_iou = 0.0

        if len(gt_boxes) > 0:

            best_iou = max(
                compute_iou(
                    box,
                    gt_box,
                )
                for gt_box in gt_boxes
            )

        status = (
            "TP"
            if best_iou >= IOU_THRESHOLD
            else "FP"
        )

        rect = Rectangle(
            (
                x1,
                y1,
            ),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor="red",
        )

        ax.add_patch(
            rect
        )

        ax.text(
            x1,
            min(
                height - 5,
                y2 + 12,
            ),
            (
                f"P#{index} "
                f"s={score:.2f} "
                f"IoU={best_iou:.2f} "
                f"{status}"
            ),
            color="red",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                pad=2,
            ),
        )

    # ========================================================
    # Title
    # ========================================================

    ax.set_title(
        (
            f"{patient_id}\n"
            f"Category: {image_category}\n"
            f"GT={len(gt_boxes)} | "
            f"Pred={len(pred_boxes)}"
        ),
        fontsize=12,
    )

    ax.axis(
        "off"
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Confusion matrix
# ============================================================

def save_confusion_matrix(
    tp,
    fp,
    fn,
    tn,
    output_path,
):
    matrix = [
        [tn, fp],
        [fn, tp],
    ]

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        matrix,
        cmap="Blues",
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

    for row in range(2):

        for col in range(2):

            ax.text(
                col,
                row,
                str(
                    matrix[row][col]
                ),
                ha="center",
                va="center",
                fontsize=16,
            )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Safe metric
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full validation analysis and "
            "qualitative visualization."
        )
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--max-images-per-category",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_CATEGORY,
        help=(
            "Maximum number of saved examples per "
            "TP/FP/TN/FN category. "
            "Use -1 to save all."
        ),
    )

    args = parser.parse_args()

    if args.max_images_per_category == -1:

        max_images_per_category = None

    else:

        max_images_per_category = (
            args.max_images_per_category
        )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # Output directories
    # ========================================================

    experiment_dir = (
        OUTPUT_ROOT
        / args.experiment
    )

    metrics_dir = (
        experiment_dir
        / "metrics"
    )

    category_dirs = {
        "TP": experiment_dir / "TP",
        "FP": experiment_dir / "FP",
        "TN": experiment_dir / "TN",
        "FN": experiment_dir / "FN",
    }

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in (
        category_dirs.values()
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

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

    print(
        f"Experiment:                 "
        f"{args.experiment}"
    )

    print(
        f"Checkpoint:                 "
        f"{args.checkpoint}"
    )

    print(
        f"Image size:                 "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Batch size:                 "
        f"{BATCH_SIZE}"
    )

    print(
        f"Official score threshold:   "
        f"{SCORE_THRESHOLD}"
    )

    print(
        f"Official NMS threshold:     "
        f"{NMS_THRESHOLD}"
    )

    print(
        f"Evaluation IoU threshold:  "
        f"{IOU_THRESHOLD}"
    )

    print(
        f"Visualization threshold:    "
        f"{VISUALIZATION_THRESHOLD}"
    )

    print(
        f"Visualization overlap:      "
        f"{VISUALIZATION_OVERLAP_THRESHOLD}"
    )

    print(
        f"Validation ratio:            "
        f"{VAL_RATIO}"
    )

    print(
        f"Seed:                       "
        f"{SEED}"
    )

    print(
        f"Device:                     "
        f"{device}"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "[LOG] Creating dataset..."
    )

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    # ========================================================
    # Validation split
    # ========================================================

    print(
        "[LOG] Creating validation split..."
    )

    _, val_indices = (
        create_train_val_split(
            dataset,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )
    )

    print(
        f"[LOG] Validation images: "
        f"{len(val_indices)}"
    )

    val_loader = (
        dataset.get_dataloader(
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=VAL_NUM_WORKERS,
            indices=val_indices,
        )
    )

    # ========================================================
    # Model
    # ========================================================

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    # ========================================================
    # Official postprocessor
    # ========================================================

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=(
                SCORE_THRESHOLD
            ),
            nms_threshold=(
                NMS_THRESHOLD
            ),
        )
    )

    # ========================================================
    # Output containers
    # ========================================================

    per_image_rows = []

    total_box_tp = 0
    total_box_fp = 0
    total_box_fn = 0

    image_tp = 0
    image_fp = 0
    image_tn = 0
    image_fn = 0

    total_predictions = 0
    total_gt_boxes = 0

    best_ious = []

    category_counts = {
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
    }

    start_time = time.perf_counter()

    # ========================================================
    # Validation loop
    # ========================================================

    for batch_index, (
        images,
        targets,
    ) in enumerate(
        val_loader,
        start=1,
    ):

        images = images.to(
            device
        )

        predictions = model(
            images
        )

        detections = (
            postprocessor(
                predictions
            )
        )

        # ----------------------------------------------------
        # Process every image in the batch
        # ----------------------------------------------------

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
                .float()
            )

            pred_boxes = (
                detection["boxes"]
                .detach()
                .cpu()
                .float()
            )

            pred_scores = (
                detection["scores"]
                .detach()
                .cpu()
                .float()
            )

            patient_index = (
                (
                    batch_index - 1
                )
                * BATCH_SIZE
                + sample_index
            )

            dataset_index = val_indices[
                patient_index
            ]

            patient_id = (
                dataset
                .image_paths[
                    dataset_index
                ]
                .stem
            )

            gt_list = (
                gt_boxes.tolist()
            )

            pred_list = (
                pred_boxes.tolist()
            )

            score_list = (
                pred_scores.tolist()
            )

            # ------------------------------------------------
            # Box-level matching
            # ------------------------------------------------

            (
                matches,
                prediction_is_tp,
                num_tp,
                num_fp,
                num_fn,
            ) = match_predictions_to_ground_truth(
                prediction_boxes=pred_list,
                prediction_scores=score_list,
                ground_truth_boxes=gt_list,
                iou_threshold=IOU_THRESHOLD,
            )

            total_box_tp += num_tp
            total_box_fp += num_fp
            total_box_fn += num_fn

            total_predictions += len(
                pred_list
            )

            total_gt_boxes += len(
                gt_list
            )

            # ------------------------------------------------
            # Best IoU for each GT
            # ------------------------------------------------

            image_gt_best_ious = []

            for gt_box in gt_list:

                if len(pred_list) == 0:

                    best_iou = 0.0

                else:

                    best_iou = max(
                        compute_iou(
                            gt_box,
                            prediction_box,
                        )
                        for prediction_box
                        in pred_list
                    )

                image_gt_best_ious.append(
                    best_iou
                )

                best_ious.append(
                    best_iou
                )

            # ------------------------------------------------
            # Image-level classification
            # ------------------------------------------------

            image_category = (
                classify_image(
                    ground_truth_boxes=gt_list,
                    prediction_boxes=pred_list,
                )
            )

            if image_category == "TP":
                image_tp += 1

            elif image_category == "TN":
                image_tn += 1

            elif image_category == "FP":
                image_fp += 1

            elif image_category == "FN":
                image_fn += 1

            category_counts[
                image_category
            ] += 1

            # ------------------------------------------------
            # Per-image metrics
            # ------------------------------------------------

            row = {
                "patient_id": patient_id,
                "gt_boxes": len(gt_list),
                "predictions": len(pred_list),
                "box_tp": num_tp,
                "box_fp": num_fp,
                "box_fn": num_fn,
                "image_category": image_category,
                "max_score": (
                    max(score_list)
                    if score_list
                    else 0.0
                ),
                "mean_score": (
                    sum(score_list)
                    / len(score_list)
                    if score_list
                    else 0.0
                ),
                "best_iou": (
                    max(
                        image_gt_best_ious
                    )
                    if image_gt_best_ious
                    else 0.0
                ),
                "mean_gt_best_iou": (
                    sum(
                        image_gt_best_ious
                    )
                    / len(
                        image_gt_best_ious
                    )
                    if image_gt_best_ious
                    else 0.0
                ),
            }

            per_image_rows.append(
                row
            )

            # ------------------------------------------------
            # Visualization detections
            # ------------------------------------------------

            detections_for_plot = (
                prepare_visualization_detections(
                    detection
                )
            )

            # ------------------------------------------------
            # Save example image only if category quota allows
            # ------------------------------------------------

            should_save = (
                max_images_per_category
                is None
                or
                category_counts[
                    image_category
                ]
                <= max_images_per_category
            )

            if should_save:

                image_tensor = (
                    images[
                        sample_index
                    ]
                    .detach()
                    .cpu()
                )

                output_path = (
                    category_dirs[
                        image_category
                    ]
                    /
                    (
                        f"{patient_id}.png"
                    )
                )

                plot_detection_image(
                    image=image_tensor,
                    patient_id=patient_id,
                    ground_truth_boxes=gt_boxes,
                    detections=detections_for_plot,
                    output_path=output_path,
                    image_category=image_category,
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index % 100 == 0
            or
            batch_index == len(
                val_loader
            )
        ):

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"[VAL ANALYSIS] "
                f"batch={batch_index}/"
                f"{len(val_loader)} "
                f"progress="
                f"{100.0 * batch_index / len(val_loader):.1f}% "
                f"time="
                f"{elapsed / 60.0:.2f} min"
            )

    # ========================================================
    # Detection metrics
    # ========================================================

    box_precision = safe_divide(
        total_box_tp,
        total_box_tp + total_box_fp,
    )

    box_recall = safe_divide(
        total_box_tp,
        total_box_tp + total_box_fn,
    )

    box_f1 = safe_divide(
        2.0
        * box_precision
        * box_recall,
        box_precision
        + box_recall,
    )

    # ========================================================
    # Image-level metrics
    # ========================================================

    total_images = (
        image_tp
        + image_fp
        + image_tn
        + image_fn
    )

    image_accuracy = safe_divide(
        image_tp + image_tn,
        total_images,
    )

    image_precision = safe_divide(
        image_tp,
        image_tp + image_fp,
    )

    image_recall = safe_divide(
        image_tp,
        image_tp + image_fn,
    )

    image_specificity = safe_divide(
        image_tn,
        image_tn + image_fp,
    )

    image_f1 = safe_divide(
        2.0
        * image_precision
        * image_recall,
        image_precision
        + image_recall,
    )

    # ========================================================
    # Mean IoU
    # ========================================================

    mean_best_iou = (
        sum(best_ious)
        /
        len(best_ious)
        if best_ious
        else 0.0
    )

    mean_predictions_per_image = (
        total_predictions
        /
        total_images
        if total_images > 0
        else 0.0
    )

    # ========================================================
    # Save per-image CSV
    # ========================================================

    csv_path = (
        metrics_dir
        / "per_image.csv"
    )

    fieldnames = [
        "patient_id",
        "gt_boxes",
        "predictions",
        "box_tp",
        "box_fp",
        "box_fn",
        "image_category",
        "max_score",
        "mean_score",
        "best_iou",
        "mean_gt_best_iou",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            per_image_rows
        )

    # ========================================================
    # Summary dictionary
    # ========================================================

    summary = {
        "experiment": args.experiment,
        "checkpoint": args.checkpoint,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "score_threshold": SCORE_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "validation_ratio": VAL_RATIO,
        "seed": SEED,
        "num_images": total_images,
        "total_gt_boxes": total_gt_boxes,
        "total_predictions": total_predictions,
        "mean_predictions_per_image": (
            mean_predictions_per_image
        ),
        "mean_best_gt_iou": mean_best_iou,
        "box_metrics": {
            "tp": total_box_tp,
            "fp": total_box_fp,
            "fn": total_box_fn,
            "precision": box_precision,
            "recall": box_recall,
            "f1": box_f1,
        },
        "image_metrics": {
            "tp": image_tp,
            "tn": image_tn,
            "fp": image_fp,
            "fn": image_fn,
            "accuracy": image_accuracy,
            "precision": image_precision,
            "recall": image_recall,
            "specificity": image_specificity,
            "f1": image_f1,
        },
    }

    json_path = (
        metrics_dir
        / "summary.json"
    )

    with json_path.open(
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # Save confusion matrix
    # ========================================================

    confusion_matrix_path = (
        metrics_dir
        / "confusion_matrix.png"
    )

    save_confusion_matrix(
        tp=image_tp,
        fp=image_fp,
        fn=image_fn,
        tn=image_tn,
        output_path=(
            confusion_matrix_path
        ),
    )

    # ========================================================
    # Save human-readable summary
    # ========================================================

    summary_path = (
        metrics_dir
        / "summary.txt"
    )

    with summary_path.open(
        "w"
    ) as file:

        file.write(
            "DETECTOR VALIDATION ANALYSIS\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Experiment: {args.experiment}\n"
        )

        file.write(
            f"Checkpoint: {args.checkpoint}\n"
        )

        file.write(
            f"Images: {total_images}\n"
        )

        file.write(
            f"GT boxes: {total_gt_boxes}\n"
        )

        file.write(
            f"Predictions: {total_predictions}\n"
        )

        file.write(
            f"Predictions/image: "
            f"{mean_predictions_per_image:.4f}\n"
        )

        file.write(
            f"Mean best GT IoU: "
            f"{mean_best_iou:.4f}\n\n"
        )

        file.write(
            "BOX-LEVEL METRICS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"TP: {total_box_tp}\n"
        )

        file.write(
            f"FP: {total_box_fp}\n"
        )

        file.write(
            f"FN: {total_box_fn}\n"
        )

        file.write(
            f"Precision: {box_precision:.6f}\n"
        )

        file.write(
            f"Recall: {box_recall:.6f}\n"
        )

        file.write(
            f"F1: {box_f1:.6f}\n\n"
        )

        file.write(
            "IMAGE-LEVEL METRICS\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"TP: {image_tp}\n"
        )

        file.write(
            f"TN: {image_tn}\n"
        )

        file.write(
            f"FP: {image_fp}\n"
        )

        file.write(
            f"FN: {image_fn}\n"
        )

        file.write(
            f"Accuracy: {image_accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {image_precision:.6f}\n"
        )

        file.write(
            f"Recall: {image_recall:.6f}\n"
        )

        file.write(
            f"Specificity: {image_specificity:.6f}\n"
        )

        file.write(
            f"F1: {image_f1:.6f}\n\n"
        )

        file.write(
            "VISUALIZATION\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Threshold: "
            f"{VISUALIZATION_THRESHOLD:.3f}\n"
        )

        file.write(
            f"Overlap threshold: "
            f"{VISUALIZATION_OVERLAP_THRESHOLD:.3f}\n"
        )

        file.write(
            f"Max detections: "
            f"{MAX_VISUALIZATIONS}\n"
        )

    # ========================================================
    # Final console output
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "VALIDATION ANALYSIS COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        "BOX-LEVEL"
    )

    print(
        f"  TP:         {total_box_tp}"
    )

    print(
        f"  FP:         {total_box_fp}"
    )

    print(
        f"  FN:         {total_box_fn}"
    )

    print(
        f"  Precision:  {box_precision:.6f}"
    )

    print(
        f"  Recall:     {box_recall:.6f}"
    )

    print(
        f"  F1:         {box_f1:.6f}"
    )

    print()
    print(
        "IMAGE-LEVEL"
    )

    print(
        f"  TP:         {image_tp}"
    )

    print(
        f"  TN:         {image_tn}"
    )

    print(
        f"  FP:         {image_fp}"
    )

    print(
        f"  FN:         {image_fn}"
    )

    print(
        f"  Accuracy:   {image_accuracy:.6f}"
    )

    print(
        f"  Precision:  {image_precision:.6f}"
    )

    print(
        f"  Recall:     {image_recall:.6f}"
    )

    print(
        f"  Specificity:{image_specificity:.6f}"
    )

    print(
        f"  F1:         {image_f1:.6f}"
    )

    print()
    print(
        f"Mean best GT IoU: "
        f"{mean_best_iou:.6f}"
    )

    print()
    print(
        f"Output: "
        f"{experiment_dir}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
