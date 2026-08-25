import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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

from src.metrics import (
    compute_metrics,
)


# ============================================================
# Utilities
# ============================================================

def load_model(
    checkpoint_path,
    backbone,
    device,
):
    """
    Build the same detector architecture used by training
    and load ONLY model weights.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Backbone
    # --------------------------------------------------------

    if backbone == "chest_xray":

        path_model = (
            RESNET50_CHEST_XRAY_CHECKPOINT
        )

        if not Path(
            path_model
        ).is_file():

            raise FileNotFoundError(
                "Chest-Xray backbone checkpoint "
                f"not found:\n{path_model}"
            )

        print(
            "[LOG] Backbone: "
            "Chest-Xray pretrained ResNet-50"
        )

    elif backbone == "imagenet":

        path_model = None

        print(
            "[LOG] Backbone: "
            "ImageNet pretrained ResNet-50"
        )

    else:

        raise ValueError(
            "backbone must be "
            "'chest_xray' or 'imagenet'"
        )

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    model = DetectionFramework(
        path_model=path_model,
    ).to(device)

    print(
        "[LOG] Loading checkpoint:"
    )

    print(
        f"      {checkpoint_path}"
    )

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
    ):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
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
        "[LOG] Model weights loaded."
    )

    return model


# ============================================================
# Target preparation
# ============================================================

def normalize_target(
    target,
):
    """
    Make targets compatible with the metrics implementation.

    The current dataset returns boxes + labels. The metrics
    code also uses area / iscrowd for size-specific metrics.
    """

    boxes = (
        target["boxes"]
        .detach()
        .cpu()
        .float()
    )

    labels = (
        target["labels"]
        .detach()
        .cpu()
        .long()
    )

    if len(boxes) > 0:

        areas = (
            (
                boxes[:, 2]
                - boxes[:, 0]
            ).clamp(min=0)
            *
            (
                boxes[:, 3]
                - boxes[:, 1]
            ).clamp(min=0)
        )

    else:

        areas = torch.zeros(
            (0,),
            dtype=torch.float32,
        )

    iscrowd = torch.zeros(
        (len(boxes),),
        dtype=torch.int64,
    )

    return {
        "boxes": boxes,
        "labels": labels,
        "area": areas,
        "iscrowd": iscrowd,
    }


# ============================================================
# Standard processor
# ============================================================

def run_standard(
    processor,
    predictions,
):
    """
    A:
        current project behavior
    """

    return processor(
        predictions
    )


# ============================================================
# Range-only processor
# ============================================================

@torch.no_grad()
def run_range_only(
    processor,
    predictions,
):
    """
    B:

        P3..P7
          ↓
        decode
          ↓
        regression-range filtering
          ↓
        concatenate
          ↓
        NMS

    Uses the exact _process_level() and
    _filter_by_regression_range() methods
    from the current DetectionPostProcessor.
    """

    batch_size = (
        predictions["P3"]
        ["classification"]
        .shape[0]
    )

    outputs = []

    for batch_index in range(
        batch_size
    ):

        all_boxes = []
        all_scores = []

        # ----------------------------------------------------
        # Levels
        # ----------------------------------------------------

        for level_name, stride in zip(
            processor.levels,
            processor.strides,
        ):

            pred_level = (
                predictions[
                    level_name
                ]
            )

            classification = (
                pred_level[
                    "classification"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            regression = (
                pred_level[
                    "regression"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            centerness = (
                pred_level[
                    "centerness"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            (
                boxes,
                scores,
                anchors,
                distances,
            ) = processor._process_level(
                classification,
                regression,
                centerness,
                stride,
            )

            (
                boxes,
                scores,
                anchors,
                distances,
            ) = processor._filter_by_regression_range(
                boxes,
                scores,
                anchors,
                distances,
                stride,
            )

            if boxes.numel() == 0:
                continue

            all_boxes.append(
                boxes
            )

            all_scores.append(
                scores
            )

        # ----------------------------------------------------
        # Empty
        # ----------------------------------------------------

        if len(all_boxes) == 0:

            device = (
                predictions["P3"]
                ["classification"]
                .device
            )

            final_boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=device,
            )

            final_scores = torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            )

        else:

            final_boxes = torch.cat(
                all_boxes,
                dim=0,
            )

            final_scores = torch.cat(
                all_scores,
                dim=0,
            )

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        if final_boxes.shape[0] > 0:

            keep = torch.ops.torchvision.nms(
                final_boxes,
                final_scores,
                processor.nms_th,
            )

            final_boxes = (
                final_boxes[keep]
            )

            final_scores = (
                final_scores[keep]
            )

        final_labels = torch.ones(
            final_boxes.shape[0],
            dtype=torch.long,
            device=final_boxes.device,
        )

        outputs.append(
            {
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            }
        )

    return outputs


# ============================================================
# Paper-style processor
# ============================================================

@torch.no_grad()
def run_paper(
    processor,
    predictions,
):
    """
    C:

        P3..P7
          ↓
        decode
          ↓
        regression-range filtering
          ↓
        cross-level selection
          ↓
        NMS

    Uses the selection implementation contained in the
    current DetectionPostProcessor.
    """

    batch_size = (
        predictions["P3"]
        ["classification"]
        .shape[0]
    )

    outputs = []

    for batch_index in range(
        batch_size
    ):

        all_boxes = []
        all_scores = []
        all_levels = []

        # ----------------------------------------------------
        # Levels
        # ----------------------------------------------------

        for level_name, stride in zip(
            processor.levels,
            processor.strides,
        ):

            pred_level = (
                predictions[
                    level_name
                ]
            )

            classification = (
                pred_level[
                    "classification"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            regression = (
                pred_level[
                    "regression"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            centerness = (
                pred_level[
                    "centerness"
                ][
                    batch_index:
                    batch_index + 1
                ]
            )

            (
                boxes,
                scores,
                anchors,
                distances,
            ) = processor._process_level(
                classification,
                regression,
                centerness,
                stride,
            )

            # ------------------------------------------------
            # Scale filtering
            # ------------------------------------------------

            (
                boxes,
                scores,
                anchors,
                distances,
            ) = processor._filter_by_regression_range(
                boxes,
                scores,
                anchors,
                distances,
                stride,
            )

            if boxes.numel() == 0:
                continue

            all_boxes.append(
                boxes
            )

            all_scores.append(
                scores
            )

            all_levels.append(
                torch.full(
                    (
                        boxes.shape[0],
                    ),
                    int(stride),
                    dtype=torch.long,
                    device=boxes.device,
                )
            )

        # ----------------------------------------------------
        # Empty
        # ----------------------------------------------------

        if len(all_boxes) == 0:

            device = (
                predictions["P3"]
                ["classification"]
                .device
            )

            final_boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=device,
            )

            final_scores = torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            )

            final_levels = torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )

        else:

            final_boxes = torch.cat(
                all_boxes,
                dim=0,
            )

            final_scores = torch.cat(
                all_scores,
                dim=0,
            )

            final_levels = torch.cat(
                all_levels,
                dim=0,
            )

        # ----------------------------------------------------
        # Cross-level selection
        # ----------------------------------------------------

        if (
            final_boxes.shape[0] > 0
        ):

            (
                final_boxes,
                final_scores,
                final_levels,
            ) = (
                processor._paper_select_levels(
                    final_boxes,
                    final_scores,
                    final_levels,
                )
            )

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        if final_boxes.shape[0] > 0:

            keep = torch.ops.torchvision.nms(
                final_boxes,
                final_scores,
                processor.nms_th,
            )

            final_boxes = (
                final_boxes[keep]
            )

            final_scores = (
                final_scores[keep]
            )

        final_labels = torch.ones(
            final_boxes.shape[0],
            dtype=torch.long,
            device=final_boxes.device,
        )

        outputs.append(
            {
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels,
            }
        )

    return outputs


# ============================================================
# Metrics
# ============================================================

def compute_variant_metrics(
    predictions,
    targets,
):
    """
    Compute the exact project metrics.
    """

    metrics = compute_metrics(
        predictions,
        targets,
    )

    return metrics


# ============================================================
# Diagnostics
# ============================================================

def summarize_predictions(
    name,
    predictions,
):
    """
    Print detection-count diagnostics.
    """

    total_images = 0
    images_with_detections = 0
    total_detections = 0
    max_score = 0.0

    for detection in predictions:

        total_images += 1

        number = (
            detection["boxes"]
            .shape[0]
        )

        total_detections += number

        if number > 0:

            images_with_detections += 1

            max_score = max(
                max_score,
                float(
                    detection[
                        "scores"
                    ].max()
                    .item()
                ),
            )

    average = (
        total_detections
        / max(total_images, 1)
    )

    ratio = (
        images_with_detections
        / max(total_images, 1)
    )

    print()
    print(
        f"[{name}] diagnostics"
    )

    print(
        f"  images:              "
        f"{total_images}"
    )

    print(
        f"  images with dets:    "
        f"{images_with_detections} "
        f"({100.0 * ratio:.2f}%)"
    )

    print(
        f"  total detections:    "
        f"{total_detections}"
    )

    print(
        f"  average/img:         "
        f"{average:.4f}"
    )

    print(
        f"  max score:           "
        f"{max_score:.6f}"
    )


# ============================================================
# Pretty metrics
# ============================================================

def print_metrics(
    name,
    metrics,
):
    """
    Print the metrics relevant to the report.
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"{name}"
    )

    print(
        "=" * 70
    )

    keys = [
        "AP",
        "AP@0.5",
        "AP@0.5:0.95",
        "AP_M",
        "AP_L",
        "AR@10",
        "AR_M",
        "AR_L",
    ]

    for key in keys:

        if key in metrics:

            value = metrics[key]

            if (
                isinstance(
                    value,
                    (float, int),
                )
            ):

                print(
                    f"{key:12s}: "
                    f"{float(value):.6f}"
                )

    print(
        "=" * 70
    )


# ============================================================
# Main evaluation
# ============================================================

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Post-processing ablation on a fixed "
            "detector checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Detector checkpoint, e.g. "
            "checkpoints/exp8/best.pt"
        ),
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
        "--num-workers",
        type=int,
        default=VAL_NUM_WORKERS,
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "POST-PROCESSING ABLATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Checkpoint:   {args.checkpoint}"
    )

    print(
        f"Backbone:     {args.backbone}"
    )

    print(
        f"Image size:   {IMAGE_SIZE}"
    )

    print(
        f"Val ratio:    {VAL_RATIO}"
    )

    print(
        f"Seed:         {SEED}"
    )

    print(
        f"Batch size:   {BATCH_SIZE}"
    )

    print(
        f"Device:       {device}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "[LOG] Creating validation dataset..."
    )

    val_dataset = (
        RSNAPneumoniaDataset(
            dcm_path=TRAIN_DCM_PATH,
            csv_path=CSV_PATH,
            transform=get_test_transforms(
                IMAGE_SIZE
            ),
        )
    )

    # ========================================================
    # EXACT SAME SPLIT
    # ========================================================

    print(
        "[LOG] Recreating validation split..."
    )

    train_indices, val_indices = (
        create_train_val_split(
            val_dataset,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )
    )

    print(
        f"[LOG] Validation samples: "
        f"{len(val_indices)}"
    )

    validation_subset = (
        Subset(
            val_dataset,
            val_indices,
        )
    )

    val_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=(
            val_dataset.__class__
            .__module__
            and None
        ),
    )

    # --------------------------------------------------------
    # The dataset provides its own collate function.
    # Re-create the loader using the project's method to avoid
    # changing target formatting.
    # --------------------------------------------------------

    val_loader = (
        val_dataset.get_dataloader(
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            indices=val_indices,
        )
    )

    # ========================================================
    # Model
    # ========================================================

    model = load_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        device=device,
    )

    # ========================================================
    # Processors
    # ========================================================

    # --------------------------------------------------------
    # All processors use score_threshold=0.0 so that AP/AR
    # sees the complete prediction ranking.
    # --------------------------------------------------------

    standard_processor = (
        DetectionPostProcessor(
            score_threshold=0.0,
            nms_threshold=NMS_THRESHOLD,
            selection_mode="standard",
        )
    )

    paper_processor = (
        DetectionPostProcessor(
            score_threshold=0.0,
            nms_threshold=NMS_THRESHOLD,
            selection_mode="paper",
        )
    )

    # ========================================================
    # Accumulators
    # ========================================================

    predictions_standard = []
    predictions_range = []
    predictions_paper = []

    targets_all = []

    # ========================================================
    # Validation inference
    # ========================================================

    model.eval()

    start = time.perf_counter()

    for batch_idx, (
        images,
        batch_targets,
    ) in enumerate(
        val_loader,
        start=1,
    ):

        images = images.to(
            device
        )

        # ----------------------------------------------------
        # ONE model forward
        # ----------------------------------------------------

        raw_predictions = (
            model(images)
        )

        # ----------------------------------------------------
        # A: standard
        # ----------------------------------------------------

        standard = run_standard(
            standard_processor,
            raw_predictions,
        )

        # ----------------------------------------------------
        # B: range only
        # ----------------------------------------------------

        range_only = run_range_only(
            standard_processor,
            raw_predictions,
        )

        # ----------------------------------------------------
        # C: paper-style
        # ----------------------------------------------------

        paper = run_paper(
            paper_processor,
            raw_predictions,
        )

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        predictions_standard.extend(
            standard
        )

        predictions_range.extend(
            range_only
        )

        predictions_paper.extend(
            paper
        )

        for target in batch_targets:

            targets_all.append(
                normalize_target(
                    target
                )
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx % 100 == 0
            or batch_idx
            == len(val_loader)
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            progress = (
                100.0
                * batch_idx
                / len(val_loader)
            )

            print(
                f"[ABLATION] "
                f"batch={batch_idx}/"
                f"{len(val_loader)} "
                f"progress={progress:.1f}% "
                f"time={elapsed / 60.0:.2f} min"
            )

    # ========================================================
    # Sanity check
    # ========================================================

    number_targets = len(
        targets_all
    )

    number_standard = len(
        predictions_standard
    )

    number_range = len(
        predictions_range
    )

    number_paper = len(
        predictions_paper
    )

    print()
    print(
        "[LOG] Dataset sizes:"
    )

    print(
        f"  targets:   {number_targets}"
    )

    print(
        f"  standard:  {number_standard}"
    )

    print(
        f"  range:     {number_range}"
    )

    print(
        f"  paper:     {number_paper}"
    )

    if not (
        number_targets
        == number_standard
        == number_range
        == number_paper
    ):

        raise RuntimeError(
            "Prediction/target list lengths differ."
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    summarize_predictions(
        "A — STANDARD",
        predictions_standard,
    )

    summarize_predictions(
        "B — RANGE FILTER",
        predictions_range,
    )

    summarize_predictions(
        "C — PAPER STYLE",
        predictions_paper,
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics_standard = (
        compute_variant_metrics(
            predictions_standard,
            targets_all,
        )
    )

    metrics_range = (
        compute_variant_metrics(
            predictions_range,
            targets_all,
        )
    )

    metrics_paper = (
        compute_variant_metrics(
            predictions_paper,
            targets_all,
        )
    )

    print_metrics(
        "A — STANDARD",
        metrics_standard,
    )

    print_metrics(
        "B — RANGE FILTER",
        metrics_range,
    )

    print_metrics(
        "C — PAPER STYLE",
        metrics_paper,
    )

    # ========================================================
    # Final comparison
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 80
    )

    comparison = {
        "STANDARD": metrics_standard,
        "RANGE": metrics_range,
        "PAPER": metrics_paper,
    }

    print(
        f"{'Mode':<15}"
        f"{'AP':>12}"
        f"{'AP_M':>12}"
        f"{'AP_L':>12}"
        f"{'AR@10':>12}"
    )

    print(
        "-" * 63
    )

    for name, metrics in (
        comparison.items()
    ):

        ap = float(
            metrics.get(
                "AP",
                metrics.get(
                    "AP@0.5",
                    0.0,
                ),
            )
        )

        ap_m = float(
            metrics.get(
                "AP_M",
                0.0,
            )
        )

        ap_l = float(
            metrics.get(
                "AP_L",
                0.0,
            )
        )

        ar = float(
            metrics.get(
                "AR@10",
                0.0,
            )
        )

        print(
            f"{name:<15}"
            f"{ap:>12.6f}"
            f"{ap_m:>12.6f}"
            f"{ap_l:>12.6f}"
            f"{ar:>12.6f}"
        )

    print(
        "=" * 80
    )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()