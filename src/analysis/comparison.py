from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def discover_models(analysis_dir: Path) -> list[Path]:
    """Discover analysis folders containing a saved PR curve."""
    if not analysis_dir.is_dir():
        raise FileNotFoundError(
            f"Analysis directory not found: {analysis_dir}"
        )

    return [
        p for p in sorted(
            analysis_dir.iterdir(),
            key=lambda x: x.name.lower(),
        )
        if p.is_dir()
        and (
            p / "metrics" / "precision_recall_curve.csv"
        ).is_file()
    ]


def load_pr_curve(model_dir: Path) -> tuple[list[float], list[float]]:
    """Load recall and precision values from precision_recall_curve.csv."""
    path = model_dir / "metrics" / "precision_recall_curve.csv"

    recalls = []
    precisions = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"recall", "precision"}
        if not required.issubset(reader.fieldnames or set()):
            raise RuntimeError(
                f"Invalid PR curve file: {path}. "
                "Expected columns: recall, precision."
            )

        for row in reader:
            recalls.append(float(row["recall"]))
            precisions.append(float(row["precision"]))

    return recalls, precisions


def load_metrics(model_dir: Path) -> dict:
    """Load metrics.json for one analyzed model."""
    path = model_dir / "metrics" / "metrics.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_summary(model_dir: Path, metrics: dict) -> dict:
    """
    Extract metrics needed for the final model comparison.

    tau_star is included as a descriptive operating-point quantity, not as
    a performance metric.
    """
    detection = metrics.get("detection_metrics", {})
    image = metrics.get("image_level", {})
    box = metrics.get("box_level", {})

    return {
        "model": model_dir.name,
        "backbone": metrics.get("backbone", ""),
        "resnet_depth": metrics.get("resnet_depth", ""),
        "AP": detection.get(
            "AP",
            metrics.get("precision_recall", {}).get("AP", ""),
        ),
        "AP_M": detection.get("AP_M", ""),
        "AP_L": detection.get("AP_L", ""),
        "AR@10": detection.get("AR@10", ""),
        "AR_M": detection.get("AR_M", ""),
        "AR_L": detection.get("AR_L", ""),
        "mean_best_matched_iou": metrics.get(
            "mean_best_matched_iou", ""
        ),
        "mean_predictions_per_image": metrics.get(
            "mean_predictions_per_image", ""
        ),
        "tau_star": metrics.get(
            "visualization_threshold", ""
        ),
        "youden_J": image.get("youden_j", ""),
        "image_precision": image.get("precision", ""),
        "image_recall": image.get("recall", ""),
        "image_specificity": image.get("specificity", ""),
        "image_accuracy": image.get("accuracy", ""),
        "image_f1": image.get("f1", ""),
        "image_tp": image.get("tp", ""),
        "image_tn": image.get("tn", ""),
        "image_fp": image.get("fp", ""),
        "image_fn": image.get("fn", ""),
        "box_precision": box.get("precision", ""),
        "box_recall": box.get("recall", ""),
        "box_f1": box.get("f1", ""),
    }


def save_summary_csv(rows: list[dict], output_dir: Path) -> Path:
    """Save the final comparison table as CSV."""
    path = output_dir / "metrics_comparison.csv"

    fieldnames = [
        "model",
        "backbone",
        "resnet_depth",
        "AP",
        "AP_M",
        "AP_L",
        "AR@10",
        "AR_M",
        "AR_L",
        "mean_best_matched_iou",
        "mean_predictions_per_image",
        "tau_star",
        "youden_J",
        "image_precision",
        "image_recall",
        "image_specificity",
        "image_accuracy",
        "image_f1",
        "image_tp",
        "image_tn",
        "image_fp",
        "image_fn",
        "box_precision",
        "box_recall",
        "box_f1",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_combined_pr_csv(
    curves: list[tuple[str, list[float], list[float]]],
    output_dir: Path,
) -> Path:
    """Save all PR curves in one long-format CSV."""
    path = output_dir / "precision_recall_comparison.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "recall", "precision"],
        )
        writer.writeheader()

        for model, recalls, precisions in curves:
            for recall, precision in zip(recalls, precisions):
                writer.writerow(
                    {
                        "model": model,
                        "recall": recall,
                        "precision": precision,
                    }
                )

    return path



def _save_figure(
    fig,
    path: Path,
) -> Path:
    """Save a Matplotlib figure in PNG format."""
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return path


def plot_precision_recall(
    curves: list[tuple[str, list[float], list[float], float | None]],
    output_dir: Path,
) -> Path:
    """Create a single PR comparison plot."""
    path = output_dir / "precision_recall_comparison.png"

    fig, ax = plt.subplots(
        figsize=(8, 7),
    )

    for (
        model,
        recalls,
        precisions,
        ap,
    ) in curves:

        label = (
            f"{model} (AP={ap:.4f})"
            if ap is not None
            else model
        )

        ax.plot(
            recalls,
            precisions,
            linewidth=2.0,
            label=label,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Comparison")
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    return _save_figure(fig, path)


def plot_grouped_metrics(
    rows: list[dict],
    metrics: list[str],
    labels: list[str],
    title: str,
    filename: str,
    output_dir: Path,
    ylabel: str = "Score",
    ylim: tuple[float, float] = (0.0, 1.0),
) -> Path:
    """
    Create a grouped bar chart for scalar model metrics.

    Values are printed above the bars. For ordinary performance metrics in
    [0, 1], the y-axis is automatically zoomed around the observed values so
    small differences between models remain visible. Explicit limits, such
    as (-1, 1) for Youden's J, are preserved.
    """
    path = output_dir / filename

    x_positions = list(range(len(rows)))
    n_metrics = len(metrics)

    width = (
        0.55
        if n_metrics == 1
        else 0.8 / n_metrics
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    all_values = []

    for index, metric in enumerate(metrics):
        values = [
            float(row[metric])
            if row.get(metric, "") != ""
            else 0.0
            for row in rows
        ]

        all_values.extend(values)

        offsets = [
            x - 0.4
            + width / 2
            + index * width
            for x in x_positions
        ]

        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=labels[index],
        )

        for bar, value in zip(
            bars,
            values,
        ):
            ax.annotate(
                f"{value:.3f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(
        x_positions
    )

    ax.set_xticklabels(
        [row["model"] for row in rows],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(ylabel)

    if ylim == (0.0, 1.0) and all_values:
        minimum = min(all_values)
        maximum = max(all_values)

        if maximum > minimum:
            margin = max(
                0.02,
                0.08 * (
                    maximum - minimum
                ),
            )

            lower = max(
                0.0,
                minimum - margin,
            )

            upper = min(
                1.0,
                maximum + margin,
            )

            # Avoid excessive zoom when values are almost identical.
            if upper - lower < 0.08:
                center = (
                    minimum + maximum
                ) / 2.0

                half_range = 0.04

                lower = max(
                    0.0,
                    center - half_range,
                )

                upper = min(
                    1.0,
                    center + half_range,
                )

            ax.set_ylim(
                lower,
                upper,
            )

        else:
            ax.set_ylim(
                max(
                    0.0,
                    minimum - 0.05,
                ),
                min(
                    1.0,
                    maximum + 0.05,
                ),
            )

    else:
        ax.set_ylim(
            ylim[0],
            ylim[1],
        )

    ax.set_title(title)

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    if n_metrics > 1:
        ax.legend()

    return _save_figure(
        fig,
        path,
    )



def plot_ap_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare AP, AP_M and AP_L."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "AP",
            "AP_M",
            "AP_L",
        ],
        labels=[
            "AP",
            "AP_M",
            "AP_L",
        ],
        title="Average Precision Comparison",
        filename="ap_comparison.png",
        output_dir=output_dir,
    )


def plot_ar_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare AR@10, AR_M and AR_L."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "AR@10",
            "AR_M",
            "AR_L",
        ],
        labels=[
            "AR@10",
            "AR_M",
            "AR_L",
        ],
        title="Average Recall Comparison",
        filename="ar_comparison.png",
        output_dir=output_dir,
    )


def plot_operating_point_metrics(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare image-level metrics measured at each model's tau*."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "image_precision",
            "image_recall",
            "image_specificity",
            "image_f1",
        ],
        labels=[
            "Precision",
            "Recall",
            "Specificity",
            "F1",
        ],
        title="Image-Level Performance at Youden Operating Point",
        filename="operating_point_comparison.png",
        output_dir=output_dir,
    )


def plot_localization_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare mean matched IoU and box-level F1."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "mean_best_matched_iou",
            "box_f1",
        ],
        labels=[
            "Mean matched IoU",
            "Box F1",
        ],
        title="Localization and Box-Level Performance",
        filename="localization_comparison.png",
        output_dir=output_dir,
    )


def plot_detection_density(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare the average number of post-threshold predictions per image."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "mean_predictions_per_image",
        ],
        labels=[
            "Predictions / image",
        ],
        title="Average Number of Predictions per Image",
        filename="predictions_per_image_comparison.png",
        output_dir=output_dir,
        ylabel="Predictions per image",
        ylim=(
            0.0,
            max(
                1.0,
                max(
                    float(row["mean_predictions_per_image"])
                    if row.get("mean_predictions_per_image", "") != ""
                    else 0.0
                    for row in rows
                )
                * 1.15,
            ),
        ),
    )


def plot_tau_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """
    Compare the Youden-optimal thresholds.

    tau* is an operating-point descriptor, not a direct performance ranking.
    """
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "tau_star",
        ],
        labels=[
            r"$\tau^*$",
        ],
        title="Youden-Optimal Detection Threshold",
        filename="youden_threshold_comparison.png",
        output_dir=output_dir,
        ylabel=r"$\tau^*$",
    )


def plot_youden_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """Compare Youden's J at each model's operating point."""
    return plot_grouped_metrics(
        rows=rows,
        metrics=[
            "youden_J",
        ],
        labels=[
            "Youden J",
        ],
        title="Youden's J at Selected Operating Point",
        filename="youden_j_comparison.png",
        output_dir=output_dir,
        ylabel="Youden J",
        ylim=(-1.0, 1.0),
    )


def save_confusion_matrix_comparison(
    rows: list[dict],
    output_dir: Path,
) -> Path:
    """
    Save image-level confusion counts in one comparison CSV.

    Individual confusion-matrix figures remain produced by the analyzer/
    visualizer for each model. This file is intended for the report table or
    further plotting, rather than replacing those matrices.
    """
    path = output_dir / "confusion_matrix_comparison.csv"

    fieldnames = [
        "model",
        "TP",
        "TN",
        "FP",
        "FN",
    ]

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

        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "TP": row.get("image_tp", ""),
                    "TN": row.get("image_tn", ""),
                    "FP": row.get("image_fp", ""),
                    "FN": row.get("image_fn", ""),
                }
            )

    return path

def save_summary_json(rows: list[dict], output_dir: Path) -> Path:
    """Save comparison results as JSON."""
    path = output_dir / "comparison_summary.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare post-training results for analyzed detector models."
        )
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("visualization"),
        help="Directory containing model analysis folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for comparison outputs. "
            "Default: <analysis-dir>/comparison"
        ),
    )

    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Optional model folder to include. Repeat for multiple models. "
            "If omitted, all discovered models are included."
        ),
    )

    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else analysis_dir / "comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_models(analysis_dir)

    if not discovered:
        raise RuntimeError(
            f"No analyzed models found in {analysis_dir}."
        )

    if args.model:
        by_name = {p.name: p for p in discovered}
        missing = [name for name in args.model if name not in by_name]

        if missing:
            raise RuntimeError(
                "Requested models not found:\n"
                + "\n".join(f"  - {name}" for name in missing)
            )

        model_dirs = [by_name[name] for name in args.model]
    else:
        model_dirs = discovered

    print("=" * 80)
    print("POST-TRAINING MODEL COMPARISON")
    print("=" * 80)
    print(f"[COMPARE] Models selected: {len(model_dirs)}")

    curves_for_plot = []
    curves_for_csv = []
    rows = []

    for model_dir in model_dirs:
        print(f"[COMPARE] Reading: {model_dir.name}")

        recalls, precisions = load_pr_curve(model_dir)
        metrics = load_metrics(model_dir)
        summary = extract_summary(model_dir, metrics)
        rows.append(summary)

        ap = summary["AP"]
        ap_value = float(ap) if ap != "" else None

        curves_for_plot.append(
            (
                model_dir.name,
                recalls,
                precisions,
                ap_value,
            )
        )

        curves_for_csv.append(
            (
                model_dir.name,
                recalls,
                precisions,
            )
        )

    rows.sort(
        key=lambda row: (
            float(row["AP"])
            if row["AP"] != ""
            else float("-inf")
        ),
        reverse=True,
    )

    summary_csv = save_summary_csv(rows, output_dir)
    pr_csv = save_combined_pr_csv(curves_for_csv, output_dir)
    summary_json = save_summary_json(rows, output_dir)

    pr_png = plot_precision_recall(
        curves_for_plot,
        output_dir,
    )

    ap_png = plot_ap_comparison(
        rows,
        output_dir,
    )

    ar_png = plot_ar_comparison(
        rows,
        output_dir,
    )

    operating_point_png = plot_operating_point_metrics(
        rows,
        output_dir,
    )

    localization_png = plot_localization_comparison(
        rows,
        output_dir,
    )

    predictions_per_image_png = plot_detection_density(
        rows,
        output_dir,
    )

    tau_png = plot_tau_comparison(
        rows,
        output_dir,
    )

    youden_png = plot_youden_comparison(
        rows,
        output_dir,
    )

    confusion_csv = save_confusion_matrix_comparison(
        rows,
        output_dir,
    )

    print()
    print("-" * 80)
    print("RANKING BY AP")
    print("-" * 80)

    for index, row in enumerate(rows, start=1):
        ap = row["AP"]
        ap_text = (
            f"{float(ap):.6f}"
            if ap != ""
            else "N/A"
        )
        print(
            f"{index:>2}. {row['model']:<35} AP={ap_text}"
        )

    print()
    print("-" * 80)
    print("YOUDEN OPERATING POINT")
    print("-" * 80)

    for row in rows:
        tau = row["tau_star"]
        tau_text = (
            f"{float(tau):.4f}"
            if tau != ""
            else "N/A"
        )
        print(
            f"{row['model']:<35} tau*={tau_text}"
        )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"[COMPARE] {summary_csv}")
    print(f"[COMPARE] {pr_csv}")
    print(f"[COMPARE] {summary_json}")
    print(f"[COMPARE] {pr_png}")
    print(f"[COMPARE] {ap_png}")
    print(f"[COMPARE] {ar_png}")
    print(f"[COMPARE] {operating_point_png}")
    print(f"[COMPARE] {localization_png}")
    print(f"[COMPARE] {predictions_per_image_png}")
    print(f"[COMPARE] {tau_png}")
    print(f"[COMPARE] {youden_png}")
    print(f"[COMPARE] {confusion_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()