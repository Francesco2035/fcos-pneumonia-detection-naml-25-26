from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle


from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
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
# Configuration
# ============================================================

CHECKPOINT = (
    "checkpoints/exp8/best.pt"
)

OUTPUT_DIR = Path(
    "visualizations_exp8_val"
)

NUM_SAMPLES = 20


# ============================================================
# Official evaluation configuration
# ============================================================

# DO NOT CHANGE THESE.
#
# These are the parameters used by the official validation
# and therefore by the reported AP / AR metrics.

OFFICIAL_SCORE_THRESHOLD = SCORE_THRESHOLD
OFFICIAL_NMS_THRESHOLD = NMS_THRESHOLD


# ============================================================
# Visualization-only configuration
# ============================================================

# Lower than 0.25 because redundant predictions will now be
# removed separately.
VISUALIZATION_THRESHOLD = 0.2

# If the intersection covers at least this fraction of the
# smaller box, the two predictions are considered redundant.
VISUALIZATION_OVERLAP_THRESHOLD = 0.27

# Final maximum number of predictions shown in one image.
MAX_VISUALIZATIONS = 10


# ============================================================
# IoU
# ============================================================

def compute_iou(
    box_a,
    box_b,
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

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
        intersection / union
    )


# ============================================================
# Overlap relative to smaller box
# ============================================================

def compute_overlap_over_smaller_area(
    box_a,
    box_b,
):
    """
    Measure how much of the smaller box is covered by
    the intersection.

    Example:

        small box completely inside large box
            -> overlap = 1.0

        60% of the small box covered
            -> overlap = 0.60
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )

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
# Best IoU between prediction and GT
# ============================================================

def best_iou_against_gt(
    prediction_box,
    gt_boxes,
):
    if len(gt_boxes) == 0:
        return 0.0

    return max(
        compute_iou(
            prediction_box,
            gt_box,
        )
        for gt_box in gt_boxes
    )


# ============================================================
# Best prediction for a GT
# ============================================================

def best_prediction_for_gt(
    gt_box,
    prediction_boxes,
    prediction_scores,
):
    if len(prediction_boxes) == 0:
        return None

    best_index = -1
    best_iou = 0.0

    for index, pred_box in enumerate(
        prediction_boxes
    ):

        iou = compute_iou(
            gt_box,
            pred_box,
        )

        if iou > best_iou:
            best_iou = iou
            best_index = index

    if best_index < 0:
        return None

    return {
        "index": best_index,
        "iou": best_iou,
        "score": float(
            prediction_scores[
                best_index
            ]
        ),
    }


# ============================================================
# Visualization-only redundant-box suppression
# ============================================================

def suppress_redundant_predictions(
    boxes,
    scores,
    overlap_threshold,
    max_detections,
):
    """
    Keep high-confidence predictions and remove boxes that
    are strongly redundant with an already-kept box.

    Processing is done in descending score order.

    If the intersection covers >= overlap_threshold of the
    smaller box, the lower-scoring box is discarded.

    IMPORTANT:
        This is ONLY for visualization.
        It does not affect official metrics.
    """

    if boxes.numel() == 0:
        return (
            boxes,
            scores,
        )

    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Greedy selection
    # --------------------------------------------------------

    kept_indices = []

    for candidate_index in range(
        len(boxes_cpu)
    ):

        candidate_box = (
            boxes_cpu[
                candidate_index
            ]
        )

        is_redundant = False

        for kept_index in kept_indices:

            kept_box = (
                boxes_cpu[
                    kept_index
                ]
            )

            overlap = (
                compute_overlap_over_smaller_area(
                    candidate_box,
                    kept_box,
                )
            )

            if (
                overlap
                >= overlap_threshold
            ):

                is_redundant = True
                break

        if not is_redundant:

            kept_indices.append(
                candidate_index
            )

        if (
            len(kept_indices)
            >= max_detections
        ):
            break

    # --------------------------------------------------------
    # Map back to tensors
    # --------------------------------------------------------

    kept_indices = torch.tensor(
        kept_indices,
        dtype=torch.long,
        device=boxes.device,
    )

    return (
        boxes_sorted[
            kept_indices
        ],
        scores_sorted[
            kept_indices
        ],
    )


# ============================================================
# Filter predictions ONLY for visualization
# ============================================================

def filter_for_visualization(
    detections,
):
    """
    Official inference:

        score >= 0.10
        NMS = 0.50

    Visualization only:

        score >= 0.15
        redundant-box suppression
        overlap threshold = 0.60
        maximum 10 boxes
    """

    scores = detections[
        "scores"
    ]

    # --------------------------------------------------------
    # 1. Visualization confidence threshold
    # --------------------------------------------------------

    keep = (
        scores
        >= VISUALIZATION_THRESHOLD
    )

    boxes = detections[
        "boxes"
    ][keep]

    scores = scores[
        keep
    ]

    labels = detections[
        "labels"
    ][keep]

    # --------------------------------------------------------
    # 2. Remove redundant nested / heavily overlapping boxes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

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
        CHECKPOINT,
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
        "[LOG] Checkpoint loaded:"
    )

    print(
        f"      {CHECKPOINT}"
    )

    return model


# ============================================================
# Plot one image
# ============================================================

def plot_sample(
    image,
    patient_id,
    gt_boxes,
    detections,
    output_path,
):
    image = (
        image
        .detach()
        .cpu()
    )

    # --------------------------------------------------------
    # [C,H,W] -> [H,W,C]
    # --------------------------------------------------------

    if image.ndim == 3:

        image = image.permute(
            1,
            2,
            0,
        )

    image = image.numpy()

    # --------------------------------------------------------
    # Convert RGB-like image to grayscale
    # --------------------------------------------------------

    if image.ndim == 3:

        image = image.mean(
            axis=2
        )

    gt_boxes = (
        gt_boxes
        .tolist()
    )

    pred_boxes = (
        detections[
            "boxes"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    pred_scores = (
        detections[
            "scores"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 10)
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
            fontsize=10,
            bbox=dict(
                facecolor="black",
                alpha=0.7,
                pad=2,
            ),
        )

    # ========================================================
    # Predictions
    # ========================================================

    tp_count = 0
    fp_count = 0

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

        is_tp = (
            best_iou >= 0.5
        )

        if is_tp:
            tp_count += 1
        else:
            fp_count += 1

        status = (
            "TP"
            if is_tp
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
            f"GT={len(gt_boxes)} | "
            f"Shown={len(pred_boxes)} | "
            f"TP={tp_count} | "
            f"FP={fp_count}\n"
            f"Official threshold="
            f"{OFFICIAL_SCORE_THRESHOLD:.2f} | "
            f"Visualization threshold="
            f"{VISUALIZATION_THRESHOLD:.2f} | "
            f"Overlap="
            f"{VISUALIZATION_OVERLAP_THRESHOLD:.2f}"
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
# Main
# ============================================================

@torch.no_grad()
def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 75
    )
    print(
        "EXP8 VALIDATION VISUALIZATION"
    )
    print(
        "=" * 75
    )

    print(
        f"Checkpoint:              "
        f"{CHECKPOINT}"
    )

    print(
        f"Image size:              "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Official score threshold:"
        f" {OFFICIAL_SCORE_THRESHOLD:.2f}"
    )

    print(
        f"Official NMS threshold:  "
        f"{OFFICIAL_NMS_THRESHOLD:.2f}"
    )

    print(
        f"Visualization threshold: "
        f"{VISUALIZATION_THRESHOLD:.2f}"
    )

    print(
        f"Redundancy overlap:      "
        f"{VISUALIZATION_OVERLAP_THRESHOLD:.2f}"
    )

    print(
        f"Maximum plotted boxes:   "
        f"{MAX_VISUALIZATIONS}"
    )

    print(
        f"Validation split:        "
        f"{VAL_RATIO}"
    )

    print(
        f"Seed:                    "
        f"{SEED}"
    )

    print(
        f"Device:                  "
        f"{device}"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
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

    # ========================================================
    # Model
    # ========================================================

    model = load_model(
        device
    )

    # ========================================================
    # Official postprocessor
    # ========================================================

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=(
                OFFICIAL_SCORE_THRESHOLD
            ),
            nms_threshold=(
                OFFICIAL_NMS_THRESHOLD
            ),
        )
    )

    # ========================================================
    # Select deterministic samples
    # ========================================================

    positive_indices = []
    negative_indices = []

    for dataset_index in val_indices:

        image, target = dataset[
            dataset_index
        ]

        if len(
            target["boxes"]
        ) > 0:

            positive_indices.append(
                dataset_index
            )

        else:

            negative_indices.append(
                dataset_index
            )

        if (
            len(
                positive_indices
            ) >= 10
            and
            len(
                negative_indices
            ) >= 10
        ):
            break

    selected_indices = (
        positive_indices[:10]
        +
        negative_indices[:10]
    )

    print()
    print(
        "[LOG] Selected:"
    )

    print(
        f"      positives = "
        f"{len(positive_indices[:10])}"
    )

    print(
        f"      negatives = "
        f"{len(negative_indices[:10])}"
    )

    # ========================================================
    # Statistics
    # ========================================================

    total_gt = 0
    total_predictions = 0
    total_tp = 0
    total_fp = 0

    best_gt_ious = []

    # ========================================================
    # Inference
    # ========================================================

    for sample_number, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):

        image, target = dataset[
            dataset_index
        ]

        image_batch = (
            image
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Model forward
        # ----------------------------------------------------

        predictions = model(
            image_batch
        )

        # ----------------------------------------------------
        # Official post-processing
        #
        # This remains:
        #     score threshold = 0.10
        #     NMS             = 0.50
        #
        # Exactly as used for evaluation.
        # ----------------------------------------------------

        detections = (
            postprocessor(
                predictions
            )[0]
        )

        official_count = len(
            detections["boxes"]
        )

        # ----------------------------------------------------
        # Visualization-only filtering
        # ----------------------------------------------------

        detections_for_plot = (
            filter_for_visualization(
                detections
            )
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_boxes = (
            target["boxes"]
            .detach()
            .cpu()
        )

        patient_id = (
            dataset.image_paths[
                dataset_index
            ].stem
        )

        gt_list = (
            gt_boxes.tolist()
        )

        plot_boxes = (
            detections_for_plot[
                "boxes"
            ]
            .detach()
            .cpu()
            .tolist()
        )

        plot_scores = (
            detections_for_plot[
                "scores"
            ]
            .detach()
            .cpu()
            .tolist()
        )

        total_gt += len(
            gt_list
        )

        total_predictions += len(
            plot_boxes
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print()
        print(
            "-" * 75
        )

        print(
            f"[{sample_number}/"
            f"{len(selected_indices)}] "
            f"{patient_id}"
        )

        print(
            f"GT boxes:                  "
            f"{len(gt_list)}"
        )

        print(
            f"Official detections:       "
            f"{official_count}"
        )

        print(
            f"Shown after visual filter: "
            f"{len(plot_boxes)}"
        )

        # ----------------------------------------------------
        # Best prediction for each GT
        # ----------------------------------------------------

        for gt_index, gt_box in enumerate(
            gt_list,
            start=1,
        ):

            result = (
                best_prediction_for_gt(
                    gt_box,
                    plot_boxes,
                    plot_scores,
                )
            )

            if result is None:

                print(
                    f"  GT #{gt_index}: "
                    f"NO SHOWN PREDICTION"
                )

                continue

            best_gt_ious.append(
                result["iou"]
            )

            print(
                f"  GT #{gt_index}: "
                f"best IoU="
                f"{result['iou']:.3f}, "
                f"score="
                f"{result['score']:.3f}"
            )

        # ----------------------------------------------------
        # TP / FP for displayed detections
        # ----------------------------------------------------

        image_tp = 0
        image_fp = 0

        for pred_box in plot_boxes:

            best_iou = (
                best_iou_against_gt(
                    pred_box,
                    gt_list,
                )
            )

            if best_iou >= 0.5:

                image_tp += 1

            else:

                image_fp += 1

        total_tp += image_tp
        total_fp += image_fp

        print(
            f"Shown TP @ IoU 0.5: "
            f"{image_tp}"
        )

        print(
            f"Shown FP @ IoU 0.5: "
            f"{image_fp}"
        )

        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            /
            (
                f"{sample_number:02d}_"
                f"{patient_id}.png"
            )
        )

        plot_sample(
            image=image,
            patient_id=patient_id,
            gt_boxes=gt_boxes,
            detections=detections_for_plot,
            output_path=output_path,
        )

    # ========================================================
    # Summary
    # ========================================================

    mean_best_iou = (
        sum(best_gt_ious)
        /
        max(
            len(best_gt_ious),
            1,
        )
    )

    print()
    print(
        "=" * 75
    )
    print(
        "VISUALIZATION SUMMARY"
    )
    print(
        "=" * 75
    )

    print(
        f"Images:                 "
        f"{len(selected_indices)}"
    )

    print(
        f"GT boxes:               "
        f"{total_gt}"
    )

    print(
        f"Shown predictions:      "
        f"{total_predictions}"
    )

    print(
        f"Shown predictions/img:  "
        f"{total_predictions / max(len(selected_indices), 1):.2f}"
    )

    print(
        f"Shown TP @ IoU 0.5:     "
        f"{total_tp}"
    )

    print(
        f"Shown FP @ IoU 0.5:     "
        f"{total_fp}"
    )

    print(
        f"Mean best GT IoU:       "
        f"{mean_best_iou:.4f}"
    )

    print(
        f"Output directory:       "
        f"{OUTPUT_DIR}"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()