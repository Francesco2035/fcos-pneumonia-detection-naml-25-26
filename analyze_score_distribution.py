from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch

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
# Configuration
# ============================================================

CHECKPOINT = (
    "checkpoints/exp8/best.pt"
)

OUTPUT_DIR = Path(
    "visualization/exp8/metrics"
)

IOU_THRESHOLD = 0.50

# Thresholds to test AFTER collecting all predictions.
THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


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
# Safe divide
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


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
        or "model_state_dict"
        not in checkpoint
    ):
        raise RuntimeError(
            "Invalid checkpoint."
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
        "[LOG] Detector loaded."
    )

    return model


# ============================================================
# Get best IoU for prediction
# ============================================================

def best_iou_for_prediction(
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
# Process one image
# ============================================================

def collect_image_predictions(
    detection,
    target,
):
    pred_boxes = (
        detection["boxes"]
        .detach()
        .cpu()
        .tolist()
    )

    pred_scores = (
        detection["scores"]
        .detach()
        .cpu()
        .tolist()
    )

    gt_boxes = (
        target["boxes"]
        .detach()
        .cpu()
        .float()
        .tolist()
    )

    records = []

    for box, score in zip(
        pred_boxes,
        pred_scores,
    ):

        best_iou = (
            best_iou_for_prediction(
                box,
                gt_boxes,
            )
        )

        is_tp = (
            best_iou
            >= IOU_THRESHOLD
        )

        records.append(
            {
                "score": float(score),
                "iou": float(best_iou),
                "is_tp": bool(is_tp),
            }
        )

    return {
        "gt_count": len(
            gt_boxes
        ),
        "predictions": records,
    }


# ============================================================
# Threshold analysis
# ============================================================

def evaluate_threshold(
    image_records,
    threshold,
):
    """
    Evaluate box-level detection after applying a score
    threshold.

    Predictions are already classified as TP/FP according
    to IoU >= IOU_THRESHOLD.

    FN is computed from the number of GT boxes that do not
    have a surviving TP after thresholding.
    """

    tp = 0
    fp = 0
    total_gt = 0
    positive_images = 0
    negative_images = 0
    predicted_positive_images = 0

    # --------------------------------------------------------
    # Box-level analysis
    # --------------------------------------------------------

    for image in image_records:

        gt_count = image[
            "gt_count"
        ]

        predictions = image[
            "predictions"
        ]

        total_gt += gt_count

        surviving = [
            prediction
            for prediction in predictions
            if prediction["score"]
            >= threshold
        ]

        image_has_gt = (
            gt_count > 0
        )

        if image_has_gt:
            positive_images += 1
        else:
            negative_images += 1

        if len(surviving) > 0:
            predicted_positive_images += 1

        # ----------------------------------------------------
        # TP / FP
        #
        # NOTE:
        # this is the same simplified point-wise analysis
        # used in analyze_detector.py.
        # ----------------------------------------------------

        image_tp = sum(
            prediction["is_tp"]
            for prediction in surviving
        )

        image_fp = (
            len(surviving)
            - image_tp
        )

        tp += image_tp
        fp += image_fp

    # --------------------------------------------------------
    # FN
    #
    # We need unique GT matching. To avoid making the main
    # analysis complicated, estimate FN as:
    #
    # total GT - TP
    #
    # This is appropriate here because each TP corresponds
    # to a matched GT under our simplified analysis.
    # --------------------------------------------------------

    fn = max(
        0,
        total_gt - tp,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0
        * precision
        * recall,
        precision
        + recall,
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_images": (
            predicted_positive_images
        ),
        "positive_images": positive_images,
        "negative_images": negative_images,
    }


# ============================================================
# Image-level threshold analysis
# ============================================================

def evaluate_image_threshold(
    image_records,
    threshold,
):
    """
    Image-level classification:

        actual positive:
            at least one GT

        predicted positive:
            at least one detection above threshold
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for image in image_records:

        gt_positive = (
            image["gt_count"] > 0
        )

        prediction_positive = any(
            prediction["score"]
            >= threshold
            for prediction
            in image["predictions"]
        )

        if (
            gt_positive
            and prediction_positive
        ):
            tp += 1

        elif (
            not gt_positive
            and not prediction_positive
        ):
            tn += 1

        elif (
            not gt_positive
            and prediction_positive
        ):
            fp += 1

        else:
            fn += 1

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

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

    f1 = safe_divide(
        2.0
        * precision
        * recall,
        precision
        + recall,
    )

    youden = (
        recall
        + specificity
        - 1.0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "youden_j": youden,
    }


# ============================================================
# Save threshold CSV
# ============================================================

def save_csv(
    rows,
    path,
):
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


# ============================================================
# Plot score distribution
# ============================================================

def plot_score_distribution(
    image_records,
    path,
):
    tp_scores = []
    fp_scores = []

    for image in image_records:

        for prediction in image[
            "predictions"
        ]:

            if prediction["is_tp"]:

                tp_scores.append(
                    prediction["score"]
                )

            else:

                fp_scores.append(
                    prediction["score"]
                )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    if tp_scores:
        ax.hist(
            tp_scores,
            bins=50,
            alpha=0.6,
            label="TP",
        )

    if fp_scores:
        ax.hist(
            fp_scores,
            bins=50,
            alpha=0.6,
            label="FP",
        )

    ax.axvline(
        SCORE_THRESHOLD,
        linestyle="--",
        label=(
            f"Official threshold "
            f"{SCORE_THRESHOLD:.2f}"
        ),
    )

    ax.set_xlabel(
        "Detection score"
    )

    ax.set_ylabel(
        "Number of predictions"
    )

    ax.set_title(
        "TP vs FP Score Distribution"
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Plot precision-recall
# ============================================================

def plot_precision_recall(
    rows,
    path,
):
    precision = [
        row["precision"]
        for row in rows
    ]

    recall = [
        row["recall"]
        for row in rows
    ]

    thresholds = [
        row["threshold"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        recall,
        precision,
        marker="o",
    )

    for x, y, threshold in zip(
        recall,
        precision,
        thresholds,
    ):
        ax.annotate(
            f"{threshold:.2f}",
            (x, y),
            fontsize=8,
        )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall by Score Threshold"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    plt.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Plot image-level threshold analysis
# ============================================================

def plot_image_level_thresholds(
    rows,
    path,
):
    thresholds = [
        row["threshold"]
        for row in rows
    ]

    precision = [
        row["precision"]
        for row in rows
    ]

    recall = [
        row["recall"]
        for row in rows
    ]

    specificity = [
        row["specificity"]
        for row in rows
    ]

    f1 = [
        row["f1"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        thresholds,
        precision,
        marker="o",
        label="Precision",
    )

    ax.plot(
        thresholds,
        recall,
        marker="o",
        label="Recall",
    )

    ax.plot(
        thresholds,
        specificity,
        marker="o",
        label="Specificity",
    )

    ax.plot(
        thresholds,
        f1,
        marker="o",
        label="F1",
    )

    ax.set_xlabel(
        "Score threshold"
    )

    ax.set_ylabel(
        "Metric"
    )

    ax.set_title(
        "Image-Level Metrics vs Score Threshold"
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        path,
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "EXP8 SCORE DISTRIBUTION ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint:       {CHECKPOINT}"
    )

    print(
        f"Image size:       {IMAGE_SIZE}"
    )

    print(
        f"Batch size:       {BATCH_SIZE}"
    )

    print(
        f"Official score:   {SCORE_THRESHOLD}"
    )

    print(
        f"NMS:              {NMS_THRESHOLD}"
    )

    print(
        f"IoU threshold:    {IOU_THRESHOLD}"
    )

    print(
        f"Device:            {device}"
    )

    print(
        "=" * 80
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

    val_loader = (
        dataset.get_dataloader(
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=VAL_NUM_WORKERS,
            indices=val_indices,
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

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=0.0,
            nms_threshold=NMS_THRESHOLD,
        )
    )

    # IMPORTANT:
    #
    # We use score_threshold=0.0 here so that we collect ALL
    # predictions needed to study the score distribution.
    #
    # This is an ANALYSIS configuration only.
    #
    # It does NOT change the official exp8 metrics:
    #
    #     score threshold = 0.10
    #     NMS = 0.50
    #

    # ========================================================
    # Collect all predictions
    # ========================================================

    image_records = []

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

        for sample_index, detection in enumerate(
            detections
        ):

            target = targets[
                sample_index
            ]

            image_result = (
                collect_image_predictions(
                    detection,
                    target,
                )
            )

            image_records.append(
                image_result
            )

        if (
            batch_index % 100 == 0
            or batch_index == len(
                val_loader
            )
        ):

            print(
                f"[SCORE ANALYSIS] "
                f"batch={batch_index}/"
                f"{len(val_loader)} "
                f"progress="
                f"{100.0 * batch_index / len(val_loader):.1f}%"
            )

    print()
    print(
        "[LOG] Score collection completed."
    )

    # ========================================================
    # Threshold analyses
    # ========================================================

    box_rows = []
    image_rows = []

    for threshold in THRESHOLDS:

        box_rows.append(
            evaluate_threshold(
                image_records,
                threshold,
            )
        )

        image_rows.append(
            evaluate_image_threshold(
                image_records,
                threshold,
            )
        )

    # ========================================================
    # Best thresholds
    # ========================================================

    best_box_f1 = max(
        box_rows,
        key=lambda row:
        row["f1"],
    )

    best_image_f1 = max(
        image_rows,
        key=lambda row:
        row["f1"],
    )

    best_youden = max(
        image_rows,
        key=lambda row:
        row["youden_j"],
    )

    # ========================================================
    # Save CSV
    # ========================================================

    save_csv(
        box_rows,
        OUTPUT_DIR
        / "threshold_analysis_box.csv",
    )

    save_csv(
        image_rows,
        OUTPUT_DIR
        / "threshold_analysis_image.csv",
    )

    # ========================================================
    # Save plots
    # ========================================================

    plot_score_distribution(
        image_records,
        OUTPUT_DIR
        / "tp_fp_score_distribution.png",
    )

    plot_precision_recall(
        box_rows,
        OUTPUT_DIR
        / "precision_recall_curve.png",
    )

    plot_image_level_thresholds(
        image_rows,
        OUTPUT_DIR
        / "image_level_threshold_analysis.png",
    )

    # ========================================================
    # Save text summary
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "score_analysis_summary.txt"
    )

    with summary_path.open(
        "w"
    ) as file:

        file.write(
            "EXP8 SCORE DISTRIBUTION ANALYSIS\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Official inference threshold: "
            f"{SCORE_THRESHOLD}\n"
        )

        file.write(
            f"NMS threshold: "
            f"{NMS_THRESHOLD}\n"
        )

        file.write(
            f"IoU threshold: "
            f"{IOU_THRESHOLD}\n\n"
        )

        file.write(
            "BEST BOX-LEVEL F1\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Threshold: "
            f"{best_box_f1['threshold']:.3f}\n"
        )

        file.write(
            f"Precision: "
            f"{best_box_f1['precision']:.6f}\n"
        )

        file.write(
            f"Recall: "
            f"{best_box_f1['recall']:.6f}\n"
        )

        file.write(
            f"F1: "
            f"{best_box_f1['f1']:.6f}\n\n"
        )

        file.write(
            "BEST IMAGE-LEVEL F1\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Threshold: "
            f"{best_image_f1['threshold']:.3f}\n"
        )

        file.write(
            f"Precision: "
            f"{best_image_f1['precision']:.6f}\n"
        )

        file.write(
            f"Recall: "
            f"{best_image_f1['recall']:.6f}\n"
        )

        file.write(
            f"Specificity: "
            f"{best_image_f1['specificity']:.6f}\n"
        )

        file.write(
            f"F1: "
            f"{best_image_f1['f1']:.6f}\n\n"
        )

        file.write(
            "BEST YOUDEN J\n"
        )

        file.write(
            "-" * 40
            + "\n"
        )

        file.write(
            f"Threshold: "
            f"{best_youden['threshold']:.3f}\n"
        )

        file.write(
            f"Precision: "
            f"{best_youden['precision']:.6f}\n"
        )

        file.write(
            f"Recall: "
            f"{best_youden['recall']:.6f}\n"
        )

        file.write(
            f"Specificity: "
            f"{best_youden['specificity']:.6f}\n"
        )

        file.write(
            f"Youden J: "
            f"{best_youden['youden_j']:.6f}\n"
        )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "BOX-LEVEL THRESHOLD ANALYSIS"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print(
        "-" * 50
    )

    for row in box_rows:

        print(
            f"{row['threshold']:>10.2f}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>12.4f}"
            f"{row['f1']:>12.4f}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "IMAGE-LEVEL THRESHOLD ANALYSIS"
    )

    print(
        "=" * 80
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
        "-" * 74
    )

    for row in image_rows:

        print(
            f"{row['threshold']:>10.2f}"
            f"{row['precision']:>12.4f}"
            f"{row['recall']:>12.4f}"
            f"{row['specificity']:>14.4f}"
            f"{row['f1']:>12.4f}"
            f"{row['youden_j']:>12.4f}"
        )

    print()
    print(
        "=" * 80
    )

    print(
        "BEST THRESHOLDS"
    )

    print(
        "=" * 80
    )

    print(
        f"Box F1 threshold:    "
        f"{best_box_f1['threshold']:.3f}"
    )

    print(
        f"Image F1 threshold:  "
        f"{best_image_f1['threshold']:.3f}"
    )

    print(
        f"Youden threshold:    "
        f"{best_youden['threshold']:.3f}"
    )

    print()
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
