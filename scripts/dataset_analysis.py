#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Generic helpers
# ============================================================

def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower]
        + fraction * (values[upper] - values[lower])
    )


def print_section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# Chest-Xray classification dataset
# ============================================================

def analyze_classification_dataset(root_dir):
    """
    Count images in train/val/test and NORMAL/PNEUMONIA.
    """

    root_dir = Path(root_dir)

    results = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        split_result = {
            "NORMAL": 0,
            "PNEUMONIA": 0,
            "OTHER": 0,
        }

        split_dir = root_dir / split

        if not split_dir.is_dir():
            results[split] = split_result
            continue

        for class_dir in split_dir.iterdir():

            if not class_dir.is_dir():
                continue

            class_name = (
                class_dir.name
                .strip()
                .upper()
            )

            count = sum(
                1
                for path in class_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

            if class_name in (
                "NORMAL",
                "PNEUMONIA",
            ):
                split_result[
                    class_name
                ] += count
            else:
                split_result[
                    "OTHER"
                ] += count

        results[split] = split_result

    return results


def print_classification_results(results):

    print_section(
        "CHEST-XRAY CLASSIFICATION DATASET"
    )

    print(
        f"{'Split':<10}"
        f"{'NORMAL':>12}"
        f"{'PNEUMONIA':>14}"
        f"{'OTHER':>10}"
        f"{'TOTAL':>12}"
        f"{'PNEUMONIA %':>14}"
    )

    print("-" * 72)

    total_normal = 0
    total_pneumonia = 0
    total_other = 0

    for split, counts in results.items():

        normal = counts["NORMAL"]
        pneumonia = counts["PNEUMONIA"]
        other = counts["OTHER"]

        total = normal + pneumonia + other

        print(
            f"{split:<10}"
            f"{normal:>12}"
            f"{pneumonia:>14}"
            f"{other:>10}"
            f"{total:>12}"
            f"{100.0 * safe_ratio(pneumonia, total):>13.2f}%"
        )

        total_normal += normal
        total_pneumonia += pneumonia
        total_other += other

    total = (
        total_normal
        + total_pneumonia
        + total_other
    )

    print("-" * 72)

    print(
        f"{'TOTAL':<10}"
        f"{total_normal:>12}"
        f"{total_pneumonia:>14}"
        f"{total_other:>10}"
        f"{total:>12}"
        f"{100.0 * safe_ratio(total_pneumonia, total):>13.2f}%"
    )

    print()

    print(
        "NORMAL : PNEUMONIA = "
        f"{total_normal} : {total_pneumonia}"
    )

    print(
        "PNEUMONIA / NORMAL = "
        f"{safe_ratio(total_pneumonia, total_normal):.4f}"
    )


def save_classification_plot(
    results,
    output_path,
):

    splits = list(results.keys())
    x = range(len(splits))
    width = 0.35

    normal = [
        results[split]["NORMAL"]
        for split in splits
    ]

    pneumonia = [
        results[split]["PNEUMONIA"]
        for split in splits
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        [i - width / 2 for i in x],
        normal,
        width=width,
        label="NORMAL",
    )

    ax.bar(
        [i + width / 2 for i in x],
        pneumonia,
        width=width,
        label="PNEUMONIA",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of images")
    ax.set_title(
        "Chest-Xray Classification Dataset Balance"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# RSNA annotations
# ============================================================

def analyze_rsna_annotations(
    labels_path,
):
    """
    Parse the RSNA CSV using the same schema as RSNAPneumoniaDataset.

    A patient is positive if at least one row has Target == 1.
    For positive rows, x/y/width/height are stored for box analysis.
    """

    labels_path = Path(labels_path)

    if not labels_path.is_file():
        raise FileNotFoundError(
            f"RSNA labels file not found:\n{labels_path}"
        )

    image_stats = defaultdict(
        lambda: {
            "positive": False,
            "boxes": [],
        }
    )

    annotation_rows = 0

    with labels_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "RSNA CSV has no header."
            )

        required = {
            "patientId",
            "x",
            "y",
            "width",
            "height",
            "Target",
        }

        missing = required - set(
            reader.fieldnames
        )

        if missing:
            raise RuntimeError(
                "Missing RSNA CSV columns: "
                + ", ".join(sorted(missing))
                + "\nFound columns: "
                + ", ".join(reader.fieldnames)
            )

        for row in reader:

            patient_id = row[
                "patientId"
            ].strip()

            if not patient_id:
                continue

            # Create the patient entry also for Target == 0.
            patient = image_stats[
                patient_id
            ]

            try:
                target = int(
                    float(row["Target"])
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if target != 1:
                continue

            patient["positive"] = True

            try:
                x = float(row["x"])
                y = float(row["y"])
                width = float(row["width"])
                height = float(row["height"])
            except (
                TypeError,
                ValueError,
            ):
                continue

            patient["boxes"].append(
                (
                    x,
                    y,
                    width,
                    height,
                )
            )

            annotation_rows += 1

    return image_stats, annotation_rows


# ============================================================
# RSNA DICOM metadata
# ============================================================

def analyze_rsna_dicoms(
    image_dir,
):
    """
    Read DICOM metadata with pydicom.

    Pixel arrays are not loaded, which keeps this analysis fast
    and avoids unnecessarily loading the full dataset into RAM.
    """

    try:
        import pydicom
    except ImportError as error:
        raise RuntimeError(
            "pydicom is required. "
            "Add it to the uv environment."
        ) from error

    image_dir = Path(image_dir)

    dicom_paths = sorted(
        image_dir.glob("*.dcm")
    )

    if not dicom_paths:
        raise FileNotFoundError(
            f"No DICOM files found in:\n{image_dir}"
        )

    dimensions = Counter()
    photometric = Counter()

    for path in dicom_paths:

        try:
            dicom = pydicom.dcmread(
                path,
                stop_before_pixels=True,
            )

            rows = int(
                getattr(
                    dicom,
                    "Rows",
                    0,
                )
            )

            columns = int(
                getattr(
                    dicom,
                    "Columns",
                    0,
                )
            )

            if rows > 0 and columns > 0:
                dimensions[
                    (rows, columns)
                ] += 1

            photometric[
                getattr(
                    dicom,
                    "PhotometricInterpretation",
                    "UNKNOWN",
                )
            ] += 1

        except Exception as error:

            print(
                "[WARN] Failed to read "
                f"{path.name}: {error}"
            )

    return {
        "num_dicoms": len(dicom_paths),
        "dimensions": dimensions,
        "photometric": photometric,
    }


def combine_rsna_statistics(
    annotation_stats,
    dicom_stats,
):
    """
    Combine CSV annotations with the complete DICOM image set.

    The RSNA labels contain one or more rows per patient.
    Target == 1 identifies a positive image.
    """

    total_images = dicom_stats[
        "num_dicoms"
    ]

    csv_patients = len(
        annotation_stats
    )

    positive_images = sum(
        1
        for value in annotation_stats.values()
        if value["positive"]
    )

    negative_images = max(
        0,
        csv_patients - positive_images,
    )

    all_boxes = []

    boxes_per_positive_image = []

    for value in annotation_stats.values():

        boxes = value["boxes"]

        if boxes:
            boxes_per_positive_image.append(
                len(boxes)
            )

        all_boxes.extend(boxes)

    widths = [
        box[2]
        for box in all_boxes
    ]

    heights = [
        box[3]
        for box in all_boxes
    ]

    areas = [
        box[2] * box[3]
        for box in all_boxes
    ]

    return {
        "total_images": total_images,
        "csv_patients": csv_patients,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "total_boxes": len(all_boxes),
        "boxes_per_positive_image": (
            boxes_per_positive_image
        ),
        "widths": widths,
        "heights": heights,
        "areas": areas,
        "dimensions": dicom_stats[
            "dimensions"
        ],
        "photometric": dicom_stats[
            "photometric"
        ],
    }


def print_rsna_results(
    stats,
):
    """Print RSNA statistics."""

    print_section(
        "RSNA PNEUMONIA DETECTION DATASET"
    )

    total = stats[
        "total_images"
    ]

    positive = stats[
        "positive_images"
    ]

    negative = stats[
        "negative_images"
    ]

    csv_patients = stats[
        "csv_patients"
    ]

    print(
        f"Total DICOM images: "
        f"{total}"
    )

    print(
        f"Patient IDs in CSV: "
        f"{csv_patients}"
    )

    if total != csv_patients:
        print(
            "[WARNING] DICOM count and CSV patient count "
            "do not match."
        )

    print()

    print(
        f"Positive images: "
        f"{positive} "
        f"({100.0 * safe_ratio(positive, csv_patients):.2f}% of CSV patients)"
    )

    print(
        f"Negative images: "
        f"{negative} "
        f"({100.0 * safe_ratio(negative, csv_patients):.2f}% of CSV patients)"
    )

    print()

    print(
        f"Total pneumonia boxes: "
        f"{stats['total_boxes']}"
    )

    if stats[
        "boxes_per_positive_image"
    ]:

        values = stats[
            "boxes_per_positive_image"
        ]

        print(
            "Boxes per positive image: "
            f"mean={statistics.mean(values):.2f}, "
            f"median={statistics.median(values):.2f}, "
            f"max={max(values)}"
        )

    if stats["widths"]:

        print()

        print(
            "Bounding-box dimensions:"
        )

        print(
            f"  width : "
            f"mean={statistics.mean(stats['widths']):.2f}, "
            f"median={statistics.median(stats['widths']):.2f}, "
            f"p95={percentile(stats['widths'], 95):.2f}"
        )

        print(
            f"  height: "
            f"mean={statistics.mean(stats['heights']):.2f}, "
            f"median={statistics.median(stats['heights']):.2f}, "
            f"p95={percentile(stats['heights'], 95):.2f}"
        )

        print(
            f"  area  : "
            f"mean={statistics.mean(stats['areas']):.2f}, "
            f"median={statistics.median(stats['areas']):.2f}, "
            f"p95={percentile(stats['areas'], 95):.2f}"
        )

    print()

    print(
        "DICOM dimensions:"
    )

    for (
        dimension,
        count,
    ) in stats[
        "dimensions"
    ].most_common(10):

        print(
            f"  {dimension[0]} x "
            f"{dimension[1]}: "
            f"{count}"
        )

    print()

    print(
        "Photometric interpretation:"
    )

    for (
        interpretation,
        count,
    ) in stats[
        "photometric"
    ].most_common():

        print(
            f"  {interpretation}: {count}"
        )


def save_rsna_balance_plot(
    stats,
    output_path,
):

    labels = [
        "Negative",
        "Positive",
    ]

    values = [
        stats["negative_images"],
        stats["positive_images"],
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.set_title(
        "RSNA Detection Dataset: Image-Level Balance"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_bbox_distribution_plot(
    stats,
    output_path,
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    if stats["widths"]:

        ax.hist(
            stats["widths"],
            bins=40,
            alpha=0.7,
            label="Width",
        )

        ax.hist(
            stats["heights"],
            bins=40,
            alpha=0.7,
            label="Height",
        )

        ax.set_xlabel(
            "Pixels"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "RSNA Bounding-Box Dimensions"
        )

        ax.legend()

    else:

        ax.text(
            0.5,
            0.5,
            "No bounding boxes found",
            ha="center",
            va="center",
        )

        ax.set_axis_off()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze Chest-Xray and RSNA datasets."
        )
    )

    parser.add_argument(
        "--chest-xray-dir",
        type=Path,
        default=Path(
            "data/chest_xray"
        ),
    )

    parser.add_argument(
        "--rsna-dir",
        type=Path,
        default=Path(
            "data/rsna-pneumonia-detection-challenge"
        ),
    )

    parser.add_argument(
        "--rsna-labels",
        type=Path,
        default=None,
        help=(
            "Path to stage_2_train_labels.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "dataset_analysis"
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Classification dataset
    # ---------------------------------------------------------

    classification_results = (
        analyze_classification_dataset(
            args.chest_xray_dir
        )
    )

    print_classification_results(
        classification_results
    )

    save_classification_plot(
        classification_results,
        (
            args.output_dir
            / "chest_xray_class_balance.png"
        ),
    )

    # ---------------------------------------------------------
    # RSNA dataset
    # ---------------------------------------------------------

    rsna_image_dir = (
        args.rsna_dir
        / "stage_2_train_images"
    )

    if args.rsna_labels is None:

        rsna_labels_path = (
            args.rsna_dir
            / "stage_2_train_labels.csv"
        )

    else:

        rsna_labels_path = (
            args.rsna_labels
        )

    print_section(
        "READING RSNA ANNOTATIONS"
    )

    print(
        f"Images: {rsna_image_dir}"
    )

    print(
        f"Labels: {rsna_labels_path}"
    )

    (
        annotation_stats,
        annotation_rows,
    ) = analyze_rsna_annotations(
        rsna_labels_path
    )

    print(
        f"Positive annotation rows: "
        f"{annotation_rows}"
    )

    print_section(
        "READING DICOM METADATA"
    )

    dicom_stats = analyze_rsna_dicoms(
        rsna_image_dir
    )

    stats = combine_rsna_statistics(
        annotation_stats,
        dicom_stats,
    )

    print_rsna_results(
        stats
    )

    save_rsna_balance_plot(
        stats,
        (
            args.output_dir
            / "rsna_image_balance.png"
        ),
    )

    save_bbox_distribution_plot(
        stats,
        (
            args.output_dir
            / "rsna_bbox_dimensions.png"
        ),
    )

    print_section(
        "ANALYSIS COMPLETED"
    )

    print(
        f"Plots saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()