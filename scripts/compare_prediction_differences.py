#!/usr/bin/env python3

"""
CSV-first qualitative comparison of the four final detector models.

Workflow
--------
1. Read the four existing per_image_results.csv files.
2. Screen all common patient IDs using cheap scalar information:
      - TP/TN/FP/FN disagreement
      - prediction-count gap
      - best-IoU gap
3. Keep only the most interesting cases.
4. Reconstruct each model with the architecture expected by its checkpoint.
5. Run inference only on the selected validation images.
6. Compare the actual retained bounding boxes across models.
7. Save 2x2 composite figures only for verified disagreements.

The stored Youden thresholds are read from each model's metrics.json.
No threshold calibration and no AP/AR recomputation are performed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

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
)
from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import get_test_transforms
from src.datasets.split import create_train_val_split
from src.models.detector import DetectionFramework
from src.inference import DetectionPostProcessor
from src.analysis.geometry import suppress_redundant_predictions


PROJECT_DIR = Path(
    "/home/legion/shared/Projects/NAML_25-26"
)

VISUALIZATION_ROOT = PROJECT_DIR / "visualization"

def log(message):
    print(f"[COMPARE-CSV] {message}", flush=True)


MODEL_SPECS = [
    {
        "name": "resnet101_imagenet_long_ft",
        "checkpoint": (
            PROJECT_DIR
            / "checkpoints"
            / "resnet101_imagenet_long_ft"
            / "best.pt"
        ),
        "backbone": "imagenet",
        "depth": 101,
        "csv": (
            VISUALIZATION_ROOT
            / "resnet101_imagenet_long_ft"
            / "metrics"
            / "per_image_results.csv"
        ),
    },
    {
        "name": "resnet50_imagenet_ft",
        "checkpoint": (
            PROJECT_DIR
            / "checkpoints"
            / "resnet50_imagenet_ft"
            / "best.pt"
        ),
        "backbone": "imagenet",
        "depth": 50,
        "csv": (
            VISUALIZATION_ROOT
            / "resnet50_imagenet_ft"
            / "metrics"
            / "per_image_results.csv"
        ),
    },
    {
        "name": "resnet101_chestxray_long",
        "checkpoint": (
            PROJECT_DIR
            / "checkpoints"
            / "resnet101_chestxray_long"
            / "best.pt"
        ),
        "backbone": "chest_xray",
        "depth": 101,
        "csv": (
            VISUALIZATION_ROOT
            / "resnet101_chestxray_long"
            / "metrics"
            / "per_image_results.csv"
        ),
    },
    {
        "name": "resnet50_chestxray_ft",
        "checkpoint": (
            PROJECT_DIR
            / "checkpoints"
            / "resnet50_chestxray_ft"
            / "best.pt"
        ),
        "backbone": "chest_xray",
        "depth": 50,
        "csv": (
            VISUALIZATION_ROOT
            / "resnet50_chestxray_ft"
            / "metrics"
            / "per_image_results.csv"
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Screen the four existing per-image CSVs first, then verify "
            "separate category and pure-localization disagreement cases."
        )
    )

    parser.add_argument(
        "--category-cases",
        type=int,
        default=30,
        help="Maximum category-disagreement cases to verify. Default: 30.",
    )

    parser.add_argument(
        "--localization-cases",
        type=int,
        default=30,
        help=(
            "Maximum pure-localization disagreement cases to verify. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--prediction-iou-threshold",
        type=float,
        default=0.50,
        help=(
            "IoU threshold for considering two predicted boxes "
            "corresponding. Default: 0.50."
        ),
    )

    parser.add_argument(
        "--localization-iou-gap",
        type=float,
        default=0.10,
        help="Minimum best-IoU gap for localization screening. Default: 0.10.",
    )

    parser.add_argument(
        "--prediction-count-gap",
        type=int,
        default=1,
        help="Minimum prediction-count gap for localization screening. Default: 1.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            VISUALIZATION_ROOT
            / "differences_csv_first"
        ),
        help=(
            "Output directory. Default: "
            "visualization/differences_csv_first"
        ),
    )

    args = parser.parse_args()

    log(
        "Arguments: "
        f"category_cases={args.category_cases}, "
        f"localization_cases={args.localization_cases}, "
        f"prediction_iou_threshold={args.prediction_iou_threshold}, "
        f"localization_iou_gap={args.localization_iou_gap}, "
        f"prediction_count_gap={args.prediction_count_gap}, "
        f"output={args.output}"
    )

    return args


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_tau(model_spec: dict) -> float:
    metrics_path = (
        VISUALIZATION_ROOT
        / model_spec["name"]
        / "metrics"
        / "metrics.json"
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Missing metrics.json:\n{metrics_path}"
        )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    if "visualization_threshold" in metrics:
        return float(metrics["visualization_threshold"])

    if "tau_star" in metrics:
        return float(metrics["tau_star"])

    raise KeyError(
        f"No stored Youden threshold found in {metrics_path}"
    )


def read_per_image_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing per-image CSV:\n{path}"
        )

    results = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "dataset_index",
            "patient_id",
            "category",
            "num_predictions",
            "best_matched_iou",
        }

        missing = required.difference(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            patient_id = row["patient_id"]

            results[patient_id] = {
                "dataset_index": int(row["dataset_index"]),
                "patient_id": patient_id,
                "category": row["category"],
                "num_predictions": int(
                    row["num_predictions"]
                ),
                "best_matched_iou": float(
                    row["best_matched_iou"]
                ),
                "num_gt_boxes": int(
                    row["num_gt_boxes"]
                ),
            }

    return results


def infer_checkpoint_architecture(
    checkpoint_path: Path,
) -> str:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
    ):
        raise RuntimeError(
            f"Invalid detector checkpoint:\n{checkpoint_path}"
        )

    keys = list(
        checkpoint["model_state_dict"].keys()
    )

    if any(
        "identity_downsample" in key
        for key in keys
    ):
        architecture = "chest_xray"

    elif any(
        ".downsample." in key
        for key in keys
    ):
        architecture = "imagenet"

    else:
        raise RuntimeError(
            "Could not infer checkpoint architecture:\n"
            f"{checkpoint_path}"
        )

    del checkpoint
    return architecture


def get_chest_xray_pretrain_path(depth: int) -> Path:
    if depth == 50:
        return (
            PROJECT_DIR
            / "checkpoints"
            / "pretrain"
            / "chest_xray_50"
            / "best.pt"
        )

    if depth == 101:
        return (
            PROJECT_DIR
            / "checkpoints"
            / "pretrain"
            / "chest_xray_101"
            / "best.pt"
        )

    raise ValueError(
        f"Unsupported ResNet depth: {depth}"
    )


def build_model(
    spec: dict,
    device,
):
    checkpoint_architecture = (
        infer_checkpoint_architecture(
            spec["checkpoint"]
        )
    )

    if checkpoint_architecture == "chest_xray":
        backbone_path = str(
            get_chest_xray_pretrain_path(
                spec["depth"]
            )
        )
    else:
        backbone_path = None

    print()
    print("=" * 80)
    print(
        f"[VERIFY-MODEL] {spec['name']}"
    )
    print("=" * 80)
    print(
        "[VERIFY-MODEL] Checkpoint architecture: "
        f"{checkpoint_architecture}"
    )
    print(
        "[VERIFY-MODEL] Stored tau*: "
        f"{load_tau(spec):.3f}"
    )

    if backbone_path is None:
        print(
            "[VERIFY-MODEL] Architecture initialization: ImageNet"
        )
    else:
        print(
            "[VERIFY-MODEL] Architecture initialization: "
            f"Chest-Xray ResNet-{spec['depth']}"
        )
        print(
            f"[VERIFY-MODEL] Backbone checkpoint: {backbone_path}"
        )

    model = DetectionFramework(
        path_model=backbone_path,
        resnet_depth=spec["depth"],
    ).to(device)

    checkpoint = torch.load(
        spec["checkpoint"],
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    del checkpoint

    model.eval()

    return (
        model,
        checkpoint_architecture,
    )


def box_iou(
    a,
    b,
) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

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

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def prediction_set_difference(
    per_model: dict[str, dict],
    model_names: list[str],
    iou_threshold: float,
):
    pair_differences = []

    for i, model_a in enumerate(model_names):
        boxes_a = per_model[
            model_a
        ]["boxes"]

        for model_b in model_names[i + 1:]:
            boxes_b = per_model[
                model_b
            ]["boxes"]

            unmatched_a = 0
            unmatched_b = 0

            for box_a in boxes_a:
                best = max(
                    (
                        box_iou(box_a, box_b)
                        for box_b in boxes_b
                    ),
                    default=0.0,
                )

                if best < iou_threshold:
                    unmatched_a += 1

            for box_b in boxes_b:
                best = max(
                    (
                        box_iou(box_b, box_a)
                        for box_a in boxes_a
                    ),
                    default=0.0,
                )

                if best < iou_threshold:
                    unmatched_b += 1

            if (
                len(boxes_a) != len(boxes_b)
                or unmatched_a > 0
                or unmatched_b > 0
            ):
                pair_differences.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "count_a": len(boxes_a),
                        "count_b": len(boxes_b),
                        "unmatched_a": unmatched_a,
                        "unmatched_b": unmatched_b,
                    }
                )

    return pair_differences


def image_to_numpy(image):
    image = (
        image.detach()
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
        image = image.mean(axis=2)

    return image


def draw_panel(
    ax,
    image,
    gt_boxes,
    boxes,
    scores,
    model_name,
    tau,
    category,
):
    ax.imshow(
        image_to_numpy(image),
        cmap="gray",
    )

    for index, box in enumerate(
        gt_boxes,
        start=1,
    ):
        x1, y1, x2, y2 = box

        x1 = max(
            0.0,
            min(float(IMAGE_SIZE), float(x1)),
        )
        y1 = max(
            0.0,
            min(float(IMAGE_SIZE), float(y1)),
        )
        x2 = max(
            0.0,
            min(float(IMAGE_SIZE), float(x2)),
        )
        y2 = max(
            0.0,
            min(float(IMAGE_SIZE), float(y2)),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2.0,
                edgecolor="green",
            )
        )

        ax.text(
            x1,
            max(4.0, y1 - 4.0),
            f"GT #{index}",
            color="green",
            fontsize=6,
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "pad": 1.5,
            },
        )

    for index, (box, score) in enumerate(
        zip(boxes, scores),
        start=1,
    ):
        x1, y1, x2, y2 = box

        x1 = max(
            0.0,
            min(float(IMAGE_SIZE), float(x1)),
        )
        y1 = max(
            0.0,
            min(float(IMAGE_SIZE), float(y1)),
        )
        x2 = max(
            0.0,
            min(float(IMAGE_SIZE), float(x2)),
        )
        y2 = max(
            0.0,
            min(float(IMAGE_SIZE), float(y2)),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=1.4,
                edgecolor="red",
            )
        )

        ax.text(
            x1,
            min(
                float(IMAGE_SIZE - 4),
                y2 + 10,
            ),
            f"P{index} s={score:.2f}",
            color="red",
            fontsize=5.5,
            bbox={
                "facecolor": "white",
                "alpha": 0.70,
                "pad": 1,
            },
        )

    ax.set_title(
        (
            f"{model_name}\n"
            f"{category} | Pred={len(boxes)} | "
            f"$\\tau$={tau:.2f}"
        ),
        fontsize=8,
    )

    ax.set_xlim(
        0,
        IMAGE_SIZE,
    )
    ax.set_ylim(
        IMAGE_SIZE,
        0,
    )
    ax.axis("off")


def save_composite(
    patient_id,
    image,
    gt_boxes,
    model_predictions,
    model_taus,
    output_path,
    reason,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
    )

    axes = axes.flatten()

    for index, model_name in enumerate(
        model_predictions
    ):
        data = model_predictions[
            model_name
        ]

        draw_panel(
            ax=axes[index],
            image=image,
            gt_boxes=gt_boxes,
            boxes=data["boxes"],
            scores=data["scores"],
            model_name=model_name,
            tau=model_taus[model_name],
            category=data["category"],
        )

    fig.suptitle(
        (
            f"Qualitative prediction disagreement: {patient_id}\n"
            f"{reason}"
        ),
        fontsize=12,
    )

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def collect_selected_case(
    model,
    loader,
    selected_indices: set[int],
    tau: float,
    device,
):
    """
    Run only the selected dataset indices for one model.

    Returns:
        dict indexed by dataset_index with:
            image, gt_boxes, boxes, scores, category
    """

    postprocessor = DetectionPostProcessor(
        score_threshold=tau,
        nms_threshold=NMS_THRESHOLD,
    )

    results = {}

    for images, targets, dataset_indices in loader:
        images = images.to(device)

        with torch.no_grad():
            predictions = model(images)

        detections = postprocessor(
            predictions
        )

        for sample_index, detection in enumerate(
            detections
        ):
            dataset_index = int(
                dataset_indices[sample_index]
            )

            if dataset_index not in selected_indices:
                continue

            gt_boxes = (
                targets[sample_index]["boxes"]
                .detach()
                .cpu()
                .float()
            )

            boxes = (
                detection["boxes"]
                .detach()
                .cpu()
            )

            scores = (
                detection["scores"]
                .detach()
                .cpu()
            )

            (
                boxes,
                scores,
            ) = suppress_redundant_predictions(
                boxes=boxes,
                scores=scores,
                overlap_threshold=0.40,
                max_detections=10,
            )

            gt_positive = len(gt_boxes) > 0
            predicted_positive = len(boxes) > 0

            if (
                gt_positive
                and predicted_positive
            ):
                category = "TP"
            elif (
                gt_positive
                and not predicted_positive
            ):
                category = "FN"
            elif (
                not gt_positive
                and predicted_positive
            ):
                category = "FP"
            else:
                category = "TN"

            results[dataset_index] = {
                "image": (
                    images[
                        sample_index
                    ].detach().cpu()
                ),
                "gt_boxes": gt_boxes,
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "category": category,
            }

    return results


def make_selected_loader(
    dataset,
    dataset_indices,
):
    """
    Build a validation-style loader over only selected indices.

    The custom dataset dataloader returns only images and targets, so we
    wrap the returned batch with the corresponding dataset indices.
    """

    base_loader = dataset.get_dataloader(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        indices=dataset_indices,
    )

    class IndexedLoader:
        def __iter__(self):
            offset = 0

            for images, targets in base_loader:
                batch_size = len(images)

                current_indices = dataset_indices[
                    offset : offset + batch_size
                ]

                offset += batch_size

                yield (
                    images,
                    targets,
                    current_indices,
                )

        def __len__(self):
            return len(base_loader)

    return IndexedLoader()


def score_candidate(
    patient_id,
    rows_by_model,
    model_names,
    args,
):
    categories = {
        name: rows_by_model[name]["category"]
        for name in model_names
    }

    counts = {
        name: rows_by_model[name]["num_predictions"]
        for name in model_names
    }

    best_ious = {
        name: rows_by_model[name]["best_matched_iou"]
        for name in model_names
    }

    category_disagreement = (
        len(set(categories.values())) > 1
    )

    count_gap = (
        max(counts.values())
        - min(counts.values())
    )

    iou_gap = (
        max(best_ious.values())
        - min(best_ious.values())
    )

    score = 0

    if category_disagreement:
        score += 1000

    if count_gap >= args.prediction_count_gap:
        score += 100

    if iou_gap >= args.localization_iou_gap:
        score += 100

    # Prefer larger disagreements.
    score += 10 * count_gap
    score += 10 * iou_gap

    return {
        "patient_id": patient_id,
        "categories": categories,
        "counts": counts,
        "best_ious": best_ious,
        "category_disagreement": category_disagreement,
        "count_gap": count_gap,
        "iou_gap": iou_gap,
        "score": score,
    }


def write_screening_csv(
    path,
    category_candidates,
    localization_candidates,
):
    model_names = [
        "resnet101_imagenet_long_ft",
        "resnet50_imagenet_ft",
        "resnet101_chestxray_long",
        "resnet50_chestxray_ft",
    ]

    fieldnames = [
        "pool",
        "patient_id",
        "category_disagreement",
        "prediction_count_gap",
        "best_iou_gap",
    ]

    for name in model_names:
        fieldnames.extend(
            [
                f"{name}_category",
                f"{name}_num_predictions",
                f"{name}_best_matched_iou",
            ]
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for pool_name, candidates in [
            ("category_disagreement", category_candidates),
            ("pure_localization_disagreement", localization_candidates),
        ]:
            for candidate in candidates:
                row = {
                    "pool": pool_name,
                    "patient_id": candidate["patient_id"],
                    "category_disagreement": (
                        candidate["category_disagreement"]
                    ),
                    "prediction_count_gap": (
                        candidate["count_gap"]
                    ),
                    "best_iou_gap": (
                        f"{candidate['iou_gap']:.4f}"
                    ),
                }

                for name in model_names:
                    row[f"{name}_category"] = (
                        candidate["categories"][name]
                    )
                    row[f"{name}_num_predictions"] = (
                        candidate["counts"][name]
                    )
                    row[f"{name}_best_matched_iou"] = (
                        f"{candidate['best_ious'][name]:.4f}"
                    )

                writer.writerow(row)



def main():
    log("PROGRAM STARTED")
    log(f"Running file: {Path(__file__).resolve()}")

    args = parse_args()

    if args.category_cases < 0:
        raise ValueError("--category-cases must be >= 0.")

    if args.localization_cases < 0:
        raise ValueError("--localization-cases must be >= 0.")

    if not 0.0 <= args.prediction_iou_threshold <= 1.0:
        raise ValueError(
            "--prediction-iou-threshold must be in [0, 1]."
        )

    if not 0.0 <= args.localization_iou_gap <= 1.0:
        raise ValueError(
            "--localization-iou-gap must be in [0, 1]."
        )

    if args.prediction_count_gap < 1:
        raise ValueError(
            "--prediction-count-gap must be >= 1."
        )

    log("Validating arguments and preparing output directories.")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    category_dir = (
        args.output / "category_disagreements"
    )
    localization_dir = (
        args.output / "prediction_disagreements"
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    localization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = [
        spec["name"]
        for spec in MODEL_SPECS
    ]

    print("=" * 80)
    print("CSV-FIRST QUALITATIVE COMPARISON")
    print("=" * 80)

    # ---------------------------------------------------------
    # Phase 1: screen the already-generated CSVs.
    # ---------------------------------------------------------

    csv_results = {}

    for spec in MODEL_SPECS:
        model_name = spec["name"]
        print(f"[CSV] Reading {model_name}")

        csv_results[model_name] = read_per_image_csv(
            spec["csv"]
        )

        print(
            "[CSV] Rows: "
            f"{len(csv_results[model_name])}"
        )

    common_ids = set(
        csv_results[model_names[0]].keys()
    )

    for model_name in model_names[1:]:
        common_ids &= set(
            csv_results[model_name].keys()
        )

    log(f"Common patient IDs across all four models: {len(common_ids)}")

    all_candidates = []

    for patient_id in sorted(common_ids):
        rows_by_model = {
            model_name: csv_results[model_name][patient_id]
            for model_name in model_names
        }

        all_candidates.append(
            score_candidate(
                patient_id,
                rows_by_model,
                model_names,
                args,
            )
        )

    # Category disagreement pool.
    category_candidates = [
        candidate
        for candidate in all_candidates
        if candidate["category_disagreement"]
    ]

    # Pure localization pool: all models have the same image-level
    # category, but the scalar CSV outputs suggest different localization
    # quality or a different number of retained predictions.
    localization_candidates = [
        candidate
        for candidate in all_candidates
        if (
            not candidate["category_disagreement"]
            and (
                candidate["count_gap"]
                >= args.prediction_count_gap
                or candidate["iou_gap"]
                >= args.localization_iou_gap
            )
        )
    ]

    category_candidates.sort(
        key=lambda candidate: (
            -candidate["count_gap"],
            -candidate["iou_gap"],
        )
    )

    localization_candidates.sort(
        key=lambda candidate: (
            -candidate["iou_gap"],
            -candidate["count_gap"],
        )
    )

    category_candidates = category_candidates[
        :args.category_cases
    ]

    localization_candidates = localization_candidates[
        :args.localization_cases
    ]

    screening_csv = (
        args.output / "candidate_screening.csv"
    )

    write_screening_csv(
        screening_csv,
        category_candidates,
        localization_candidates,
    )

    print(
        "[CSV] Category candidates selected: "
        f"{len(category_candidates)}"
    )
    print(
        "[CSV] Pure localization candidates selected: "
        f"{len(localization_candidates)}"
    )
    print(
        "[CSV] Screening CSV: "
        f"{screening_csv}"
    )

    selected_candidates = (
        category_candidates
        + localization_candidates
    )

    if not selected_candidates:
        print("[CSV] No candidates selected.")
        return

    # ---------------------------------------------------------
    # Phase 2: recover the exact validation split.
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(IMAGE_SIZE),
    )

    (
        _,
        val_indices,
    ) = create_train_val_split(
        dataset,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    patient_to_dataset_index = {}

    for dataset_index in val_indices:
        patient_id = (
            dataset.image_paths[
                dataset_index
            ].stem
        )
        patient_to_dataset_index[
            patient_id
        ] = int(dataset_index)

    selected_dataset_indices = []

    for candidate in selected_candidates:
        patient_id = candidate["patient_id"]
        dataset_index = patient_to_dataset_index.get(
            patient_id
        )

        if dataset_index is None:
            print(
                "[WARN] Patient not found in validation split: "
                f"{patient_id}"
            )
            continue

        candidate["dataset_index"] = dataset_index
        selected_dataset_indices.append(dataset_index)

    selected_dataset_indices = sorted(
        set(selected_dataset_indices)
    )

    print(
        "[VERIFY] Selected validation images: "
        f"{len(selected_dataset_indices)}"
    )

    if not selected_dataset_indices:
        return

    selected_set = set(
        selected_dataset_indices
    )

    loader = make_selected_loader(
        dataset,
        selected_dataset_indices,
    )

    # ---------------------------------------------------------
    # Phase 3: targeted inference, one model at a time.
    # ---------------------------------------------------------

    device = get_device()

    print(f"[VERIFY] Device: {device}")

    model_results = {}
    model_taus = {}

    for model_number, spec in enumerate(MODEL_SPECS, start=1):
        model_name = spec["name"]
        tau = load_tau(spec)

        model_taus[model_name] = tau

        model, architecture = build_model(
            spec,
            device,
        )

        print(
            "[VERIFY] Running selected inference for "
            f"{model_name} "
            f"(architecture={architecture}, tau={tau:.3f})"
        )

        model_results[model_name] = collect_selected_case(
            model=model,
            loader=loader,
            selected_indices=selected_set,
            tau=tau,
            device=device,
        )

        log(
            f"[MODEL {model_number}/4] Completed {model_name}: "
            f"{len(model_results[model_name])} targeted cases."
        )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Phase 4: verify box-level differences and save 2x2 figures.
    # ---------------------------------------------------------

    verified_rows = []

    saved_category = 0
    saved_localization = 0

    category_ids = {
        candidate["patient_id"]
        for candidate in category_candidates
    }

    localization_ids = {
        candidate["patient_id"]
        for candidate in localization_candidates
    }

    for candidate in selected_candidates:
        patient_id = candidate["patient_id"]
        dataset_index = candidate.get(
            "dataset_index"
        )

        if dataset_index is None:
            continue

        if any(
            dataset_index not in model_results[model_name]
            for model_name in model_names
        ):
            print(
                "[WARN] Missing targeted result for "
                f"{patient_id}"
            )
            continue

        per_model = {
            model_name: model_results[
                model_name
            ][dataset_index]
            for model_name in model_names
        }

        categories = {
            model_name: per_model[
                model_name
            ]["category"]
            for model_name in model_names
        }

        actual_category_disagreement = (
            len(set(categories.values())) > 1
        )

        pair_differences = prediction_set_difference(
            per_model=per_model,
            model_names=model_names,
            iou_threshold=(
                args.prediction_iou_threshold
            ),
        )

        actual_prediction_disagreement = (
            len(pair_differences) > 0
        )

        image = per_model[
            model_names[0]
        ]["image"]

        gt_boxes = per_model[
            model_names[0]
        ]["gt_boxes"].tolist()

        if patient_id in category_ids:
            if not actual_category_disagreement:
                continue

            output_path = (
                category_dir
                / (
                    f"{saved_category + 1:03d}_"
                    f"{patient_id}_"
                    "category_difference.png"
                )
            )

            category_text = ", ".join(
                f"{model_name}={categories[model_name]}"
                for model_name in model_names
            )

            save_composite(
                patient_id=patient_id,
                image=image,
                gt_boxes=gt_boxes,
                model_predictions=per_model,
                model_taus=model_taus,
                output_path=output_path,
                reason=(
                    "Image-level disagreement: "
                    f"{category_text}"
                ),
            )

            saved_category += 1

            verified_rows.append(
                {
                    "patient_id": patient_id,
                    "difference_type": "category",
                    "categories": json.dumps(
                        categories,
                        sort_keys=True,
                    ),
                    "prediction_counts": json.dumps(
                        {
                            model_name: len(
                                per_model[
                                    model_name
                                ]["boxes"]
                            )
                            for model_name in model_names
                        },
                        sort_keys=True,
                    ),
                    "actual_prediction_disagreement": (
                        actual_prediction_disagreement
                    ),
                    "pair_details": json.dumps(
                        pair_differences,
                        sort_keys=True,
                    ),
                }
            )

        elif patient_id in localization_ids:
            # This branch is intentionally restricted to cases where all
            # models retain the same image-level category.
            if actual_category_disagreement:
                continue

            if not actual_prediction_disagreement:
                continue

            common_category = next(
                iter(categories.values())
            )

            output_path = (
                localization_dir
                / (
                    f"{saved_localization + 1:03d}_"
                    f"{patient_id}_"
                    "localization_difference.png"
                )
            )

            save_composite(
                patient_id=patient_id,
                image=image,
                gt_boxes=gt_boxes,
                model_predictions=per_model,
                model_taus=model_taus,
                output_path=output_path,
                reason=(
                    "Pure localization disagreement: "
                    f"all models={common_category}; "
                    f"CSV IoU gap={candidate['iou_gap']:.3f}; "
                    f"CSV count gap={candidate['count_gap']}"
                ),
            )

            saved_localization += 1

            verified_rows.append(
                {
                    "patient_id": patient_id,
                    "difference_type": "localization",
                    "categories": json.dumps(
                        categories,
                        sort_keys=True,
                    ),
                    "prediction_counts": json.dumps(
                        {
                            model_name: len(
                                per_model[
                                    model_name
                                ]["boxes"]
                            )
                            for model_name in model_names
                        },
                        sort_keys=True,
                    ),
                    "actual_prediction_disagreement": (
                        actual_prediction_disagreement
                    ),
                    "pair_details": json.dumps(
                        pair_differences,
                        sort_keys=True,
                    ),
                }
            )

    verified_csv = (
        args.output
        / "verified_disagreements.csv"
    )

    with verified_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id",
                "difference_type",
                "categories",
                "prediction_counts",
                "actual_prediction_disagreement",
                "pair_details",
            ],
        )

        writer.writeheader()
        writer.writerows(verified_rows)

    print()
    print("=" * 80)
    print(
        "CSV-FIRST QUALITATIVE COMPARISON COMPLETED"
    )
    print("=" * 80)
    print(
        "[COMPARE-CSV] Category candidates: "
        f"{len(category_candidates)}"
    )
    print(
        "[COMPARE-CSV] Pure localization candidates: "
        f"{len(localization_candidates)}"
    )
    print(
        "[COMPARE-CSV] Category figures saved: "
        f"{saved_category}"
    )
    print(
        "[COMPARE-CSV] Localization figures saved: "
        f"{saved_localization}"
    )
    print(
        "[COMPARE-CSV] Screening CSV: "
        f"{screening_csv}"
    )
    print(
        "[COMPARE-CSV] Verified CSV: "
        f"{verified_csv}"
    )
    print(
        "[COMPARE-CSV] Output directory: "
        f"{args.output}"
    )
    print("=" * 80)



if __name__ == "__main__":
    main()
