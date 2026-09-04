"""
Plot exported TensorBoard CSV curves for the NAML project.

Usage example:

    uv run python plot_tensorboard_csv.py \
        --metric "Validation AP" \
        --curve "ResNet-50 ImageNet:checkpoints/resnet50_imagenet/ap.csv" \
        --curve "ResNet-101 ImageNet:checkpoints/resnet101_imagenet/ap.csv"

Repeat --curve to compare several models.

Outputs:
    plots/<metric>.png
    plots/<metric>.svg
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_curve_argument(value: str) -> tuple[str, Path]:
    """Parse MODEL_NAME:CSV_PATH."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Each --curve must have the form 'MODEL NAME:CSV_PATH'."
        )

    label, path_text = value.split(":", 1)
    label = label.strip()
    path_text = path_text.strip()

    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty.")

    if not path_text:
        raise argparse.ArgumentTypeError("CSV path cannot be empty.")

    return label, Path(path_text)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def find_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    normalized = {
        normalize_column_name(name): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


def load_curve(
    path: Path,
) -> tuple[list[float], list[float]]:
    """
    Read common TensorBoard CSV formats.

    Supported columns include:
        Step,Value
        step,value
        Epoch,Value
        Wall time,Step,Value

    If no step/epoch column exists, row index is used.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        x_column = find_column(
            reader.fieldnames,
            ("step", "epoch", "epochs"),
        )

        y_column = find_column(
            reader.fieldnames,
            ("value", "values", "scalar"),
        )

        if y_column is None:
            raise ValueError(
                f"Could not find a value column in {path}. "
                f"Available columns: {reader.fieldnames}"
            )

        xs: list[float] = []
        ys: list[float] = []

        for row_index, row in enumerate(reader, start=1):
            raw_y = row.get(y_column)
            if raw_y is None or raw_y.strip() == "":
                continue

            try:
                y = float(raw_y)
            except ValueError:
                continue

            if x_column is None:
                x = float(row_index)
            else:
                raw_x = row.get(x_column)
                if raw_x is None or raw_x.strip() == "":
                    continue

                try:
                    x = float(raw_x)
                except ValueError:
                    continue

            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError(f"No numerical data found in {path}")

    return xs, ys


def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "plot"


def plot_curves(
    curves: list[tuple[str, list[float], list[float]]],
    metric_name: str,
    output_dir: Path,
    x_label: str,
    y_label: str | None,
    title: str | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = sanitize_filename(metric_name)
    png_path = output_dir / f"{filename}.png"
    svg_path = output_dir / f"{filename}.svg"

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, xs, ys in curves:
        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            label=label,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(
        y_label if y_label is not None else metric_name
    )
    ax.set_title(
        title if title is not None else metric_name
    )
    ax.grid(True, alpha=0.25)

    if len(curves) > 1:
        ax.legend()

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create clean Matplotlib plots from "
            "TensorBoard-exported CSV files."
        )
    )

    parser.add_argument(
        "--metric",
        required=True,
        help="Metric name, e.g. 'Validation AP' or 'Training Loss'.",
    )

    parser.add_argument(
        "--curve",
        required=True,
        action="append",
        type=parse_curve_argument,
        help=(
            "Curve in the form 'MODEL NAME:CSV_PATH'. "
            "Repeat for multiple models."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory. Default: plots/",
    )

    parser.add_argument(
        "--xlabel",
        default="Epoch",
        help="X-axis label. Default: Epoch.",
    )

    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label. Default: metric name.",
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Plot title. Default: metric name.",
    )

    args = parser.parse_args()

    curves = []

    print()
    print("=" * 70)
    print("TENSORBOARD CSV PLOT")
    print("=" * 70)
    print(f"[PLOT] Metric: {args.metric}")
    print(f"[PLOT] Curves: {len(args.curve)}")

    for label, path in args.curve:
        print(f"[PLOT] Reading: {label}")
        print(f"       {path}")

        xs, ys = load_curve(path)

        print(f"       points={len(xs)}")

        curves.append(
            (label, xs, ys)
        )

    png_path, svg_path = plot_curves(
        curves=curves,
        metric_name=args.metric,
        output_dir=args.output_dir,
        x_label=args.xlabel,
        y_label=args.ylabel,
        title=args.title,
    )

    print()
    print("=" * 70)
    print("[PLOT] Saved")
    print("=" * 70)
    print(f"[PLOT] PNG: {png_path}")
    print(f"[PLOT] SVG: {svg_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
