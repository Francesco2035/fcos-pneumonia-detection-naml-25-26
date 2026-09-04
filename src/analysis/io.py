import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def save_per_image_results(
    results,
    output_dir,
):
    """
    Save per-image analysis results to a CSV file.
    """

    metrics_dir = (
        Path(output_dir)
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        metrics_dir
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
    metrics,
    tau,
    visualization_threshold,
    output_dir,
):
    """
    Save analysis metrics in JSON and CSV formats.

    The JSON file preserves the complete nested structure.
    The CSV file stores the main metric sections in tabular form.
    """

    metrics_dir = (
        Path(output_dir)
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # JSON
    # ---------------------------------------------------------

    json_path = (
        metrics_dir
        / "metrics.json"
    )

    with json_path.open(
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    csv_path = (
        metrics_dir
        / "metrics.csv"
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

    # Additional global analysis values.
    rows.extend(
        [
            {
                "section": "global",
                "metric": "tau",
                "value": tau,
            },
            {
                "section": "global",
                "metric": (
                    "visualization_threshold"
                ),
                "value": (
                    visualization_threshold
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

    # Include PR/AP summary in the CSV when available.
    if "precision_recall" in metrics:

        pr_metrics = metrics[
            "precision_recall"
        ]

        for key in (
            "AP",
            "num_gt",
            "num_pred",
        ):

            if key in pr_metrics:
                rows.append(
                    {
                        "section": "precision_recall",
                        "metric": key,
                        "value": pr_metrics[key],
                    }
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


def save_precision_recall_curve(
    precisions,
    recalls,
    ap,
    output_dir,
):
    """
    Save the precision-recall data to CSV and the corresponding curve to PNG.

    The curve is produced from the precision and recall values already
    computed by the project's AP implementation. No additional matching or
    metric definition is introduced here.
    """

    metrics_dir = (
        Path(output_dir)
        / "metrics"
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------

    csv_path = (
        metrics_dir
        / "precision_recall_curve.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "recall",
                "precision",
            ],
        )

        writer.writeheader()

        for recall, precision in zip(
            recalls,
            precisions,
        ):

            writer.writerow(
                {
                    "recall": float(recall),
                    "precision": float(precision),
                }
            )

    # ---------------------------------------------------------
    # PNG
    # ---------------------------------------------------------

    png_path = (
        metrics_dir
        / "precision_recall_curve.png"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    if precisions and recalls:

        ax.plot(
            recalls,
            precisions,
            linewidth=2.0,
        )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        f"Precision-Recall Curve (AP@0.50 = {float(ap):.4f})"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[METRICS] Saved: {csv_path}"
    )

    print(
        f"[METRICS] Saved: {png_path}"
    )
