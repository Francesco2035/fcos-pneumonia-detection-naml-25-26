from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

from src.config import (
    IMAGE_SIZE,
    NMS_THRESHOLD,
    SCORE_THRESHOLD,
)
from src.datasets.DICOMDataset import DICOMDataset
from src.datasets.transforms import get_test_transforms
from src.models.detector import DetectionFramework
from src.inference import DetectionPostProcessor


# ============================================================
# Defaults
# ============================================================

DEFAULT_TEST_DIR = (
    "/home/legion/shared/Projects/NAML_25-26/"
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_test_images"
)

DEFAULT_OUTPUT = (
    "results/submission.csv"
)


# ============================================================
# Test dataset
# ============================================================

class RSNATestDataset(DICOMDataset):
    """
    DICOM-only RSNA test dataset.

    The original DICOM dimensions are preserved so that the
    predictions can later be mapped from the 512x512 model
    canvas back to the original image coordinates.

    Preprocessing is exactly the deterministic validation/test
    transform used by the project.
    """

    def __init__(
        self,
        dcm_path: str | Path,
        transform=None,
    ):
        super().__init__(
            dcm_path=dcm_path,
            transform=transform,
        )

    def __getitem__(
        self,
        index: int,
    ):
        path = self.image_paths[index]

        patient_id = path.stem

        # -----------------------------------------------------
        # Load original DICOM
        # -----------------------------------------------------

        image = self._load_dicom(path)

        if image.ndim != 2:
            raise RuntimeError(
                f"[{patient_id}] Expected 2D DICOM image, "
                f"got shape={image.shape}"
            )

        original_height = int(
            image.shape[0]
        )

        original_width = int(
            image.shape[1]
        )

        # -----------------------------------------------------
        # NumPy -> torchvision Image
        # -----------------------------------------------------

        image = v2.ToImage()(image)

        # -----------------------------------------------------
        # Same deterministic transform used in validation
        # -----------------------------------------------------

        if self.transform is not None:
            image = self.transform(
                image
            )

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                f"[{patient_id}] Transform returned "
                f"{type(image)}, expected torch.Tensor."
            )

        return (
            image,
            patient_id,
            original_height,
            original_width,
        )


def test_collate_fn(batch):
    images, patient_ids, heights, widths = zip(
        *batch
    )

    images = torch.stack(
        list(images),
        dim=0,
    )

    return (
        images,
        list(patient_ids),
        list(heights),
        list(widths),
    )


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path: str | Path,
    backbone: str,
    device: torch.device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    # ---------------------------------------------------------
    # Backbone
    # ---------------------------------------------------------

    if backbone == "imagenet":

        path_model = None

        print(
            "[LOG] Backbone: "
            "ImageNet pretrained ResNet-50"
        )

    else:

        raise ValueError(
            "This submission script currently supports "
            "'imagenet' detector checkpoints only."
        )

    # ---------------------------------------------------------
    # Create detector
    # ---------------------------------------------------------

    print(
        "[LOG] Creating model..."
    )

    model = DetectionFramework(
        path_model=path_model,
    ).to(device)

    # ---------------------------------------------------------
    # Load detector checkpoint
    # ---------------------------------------------------------

    print(
        "[LOG] Loading detector checkpoint:"
    )

    print(
        f"      {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise RuntimeError(
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
        "[LOG] Checkpoint loaded."
    )

    return model


# ============================================================
# Box coordinate conversion
# ============================================================

def resize_boxes_to_original(
    boxes: torch.Tensor,
    original_width: int,
    original_height: int,
    model_size: int,
) -> torch.Tensor:
    """
    Convert XYXY boxes from the model canvas
    back to the original DICOM coordinates.

    The preprocessing resize is:

        original HxW -> model_size x model_size

    Therefore x and y need independent inverse scales.
    """

    if boxes.numel() == 0:
        return boxes.clone()

    scale_x = (
        float(original_width)
        / float(model_size)
    )

    scale_y = (
        float(original_height)
        / float(model_size)
    )

    boxes = boxes.clone()

    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x

    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    # Clamp to original image boundaries.

    boxes[:, 0].clamp_(
        0.0,
        float(original_width),
    )

    boxes[:, 2].clamp_(
        0.0,
        float(original_width),
    )

    boxes[:, 1].clamp_(
        0.0,
        float(original_height),
    )

    boxes[:, 3].clamp_(
        0.0,
        float(original_height),
    )

    return boxes


# ============================================================
# Kaggle PredictionString
# ============================================================

def build_prediction_string(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    original_width: int,
    original_height: int,
    model_size: int,
) -> str:
    """
    Kaggle RSNA format:

        confidence x y width height

    Multiple detections are separated by spaces.

    Empty detections -> empty string.
    """

    if boxes.numel() == 0:
        return ""

    boxes = resize_boxes_to_original(
        boxes=boxes,
        original_width=original_width,
        original_height=original_height,
        model_size=model_size,
    )

    pieces = []

    for box, score in zip(
        boxes,
        scores,
    ):
        x1 = float(
            box[0].item()
        )

        y1 = float(
            box[1].item()
        )

        x2 = float(
            box[2].item()
        )

        y2 = float(
            box[3].item()
        )

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        score_value = float(
            score.item()
        )

        pieces.extend(
            [
                f"{score_value:.6f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{width:.2f}",
                f"{height:.2f}",
            ]
        )

    return " ".join(
        pieces
    )


# ============================================================
# Submission generation
# ============================================================

@torch.no_grad()
def create_submission(
    checkpoint_path: str | Path,
    backbone: str,
    test_dir: str | Path,
    output_path: str | Path,
    score_threshold: float,
    nms_threshold: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_images: int | None = None,
):
    start_time = time.perf_counter()

    print()
    print("=" * 75)
    print(
        "RSNA TEST INFERENCE"
    )
    print("=" * 75)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print(
        "[LOG] Creating test dataset..."
    )

    print(
        f"[LOG] Test directory:"
    )

    print(
        f"      {test_dir}"
    )

    dataset = RSNATestDataset(
        dcm_path=test_dir,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    total_dataset_images = len(
        dataset
    )

    if total_dataset_images == 0:
        raise RuntimeError(
            "No DICOM files found in:\n"
            f"{test_dir}"
        )

    print(
        f"[LOG] Test images found: "
        f"{total_dataset_images}"
    )

    # ---------------------------------------------------------
    # Optional smoke-test subset
    # ---------------------------------------------------------

    if max_images is not None:

        if max_images < 1:
            raise ValueError(
                "--max-images must be >= 1."
            )

        effective_num_images = min(
            max_images,
            total_dataset_images,
        )

        print()
        print(
            "[LOG] Smoke-test mode enabled."
        )

        print(
            f"[LOG] Processing only "
            f"{effective_num_images} / "
            f"{total_dataset_images} images."
        )

        dataset_for_loader = Subset(
            dataset,
            range(
                effective_num_images
            ),
        )

    else:

        effective_num_images = (
            total_dataset_images
        )

        dataset_for_loader = (
            dataset
        )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        collate_fn=test_collate_fn,
    )

    total_batches = len(
        dataloader
    )

    print()
    print(
        "[LOG] DataLoader configuration:"
    )

    print(
        f"      Batch size : {batch_size}"
    )

    print(
        f"      Workers    : {num_workers}"
    )

    print(
        f"      Batches    : {total_batches}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = load_model(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device=device,
    )

    # ---------------------------------------------------------
    # Postprocessor
    # ---------------------------------------------------------

    print()
    print(
        "[LOG] Creating postprocessor..."
    )

    print(
        f"      Score threshold = "
        f"{score_threshold:.4f}"
    )

    print(
        f"      NMS threshold   = "
        f"{nms_threshold:.4f}"
    )

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[LOG] Output CSV:"
    )

    print(
        f"      {output_path}"
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_images = 0
    total_detections = 0
    images_with_detections = 0

    # ---------------------------------------------------------
    # CSV generation
    # ---------------------------------------------------------

    print()
    print(
        "[TEST] Starting inference..."
    )

    with output_path.open(
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "patientId",
                "PredictionString",
            ]
        )

        # -----------------------------------------------------
        # Inference loop
        # -----------------------------------------------------

        for batch_index, (
            images,
            patient_ids,
            heights,
            widths,
        ) in enumerate(
            dataloader,
            start=1,
        ):

            batch_start = (
                time.perf_counter()
            )

            # -------------------------------------------------
            # Move images to GPU
            # -------------------------------------------------

            images = images.to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            predictions = model(
                images
            )

            # -------------------------------------------------
            # Post-processing
            # -------------------------------------------------

            detections = (
                postprocessor(
                    predictions
                )
            )

            batch_detections = 0

            # -------------------------------------------------
            # Write submission rows
            # -------------------------------------------------

            for (
                patient_id,
                detection,
                original_height,
                original_width,
            ) in zip(
                patient_ids,
                detections,
                heights,
                widths,
            ):

                boxes = detection[
                    "boxes"
                ]

                scores = detection[
                    "scores"
                ]

                prediction_string = (
                    build_prediction_string(
                        boxes=boxes,
                        scores=scores,
                        original_width=(
                            original_width
                        ),
                        original_height=(
                            original_height
                        ),
                        model_size=IMAGE_SIZE,
                    )
                )

                writer.writerow(
                    [
                        patient_id,
                        prediction_string,
                    ]
                )

                num_detections = int(
                    boxes.shape[0]
                )

                batch_detections += (
                    num_detections
                )

                total_detections += (
                    num_detections
                )

                total_images += 1

                if num_detections > 0:
                    images_with_detections += 1

            # -------------------------------------------------
            # Progress log
            # -------------------------------------------------

            elapsed = (
                time.perf_counter()
                - start_time
            )

            batch_time = (
                time.perf_counter()
                - batch_start
            )

            progress = (
                100.0
                * batch_index
                / total_batches
            )

            mean_detections = (
                total_detections
                / total_images
                if total_images > 0
                else 0.0
            )

            images_per_second = (
                total_images / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"[TEST] "
                f"batch={batch_index}/"
                f"{total_batches} "
                f"progress={progress:.1f}% "
                f"images={total_images}/"
                f"{effective_num_images} "
                f"batch_det={batch_detections} "
                f"total_det={total_detections} "
                f"det/img={mean_detections:.4f} "
                f"speed={images_per_second:.2f} img/s "
                f"batch_time={batch_time:.3f}s "
                f"elapsed={elapsed / 60.0:.2f} min",
                flush=True,
            )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    mean_detections = (
        total_detections
        / total_images
        if total_images > 0
        else 0.0
    )

    detection_image_ratio = (
        images_with_detections
        / total_images
        if total_images > 0
        else 0.0
    )

    print()
    print("=" * 75)
    print(
        "[DONE] Submission created."
    )
    print("=" * 75)

    print(
        f"[DONE] Output file:"
    )

    print(
        f"       {output_path}"
    )

    print(
        f"[DONE] Images processed:"
        f"       {total_images}"
    )

    print(
        f"[DONE] Images with detections:"
        f" {images_with_detections} "
        f"({100.0 * detection_image_ratio:.2f}%)"
    )

    print(
        f"[DONE] Total detections:"
        f"        {total_detections}"
    )

    print(
        f"[DONE] Mean detections/image:"
        f" {mean_detections:.4f}"
    )

    print(
        f"[DONE] Total time:"
        f"             {elapsed / 60.0:.2f} min"
    )

    print("=" * 75)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create an RSNA Kaggle submission "
            "from a trained FCOS detector."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to detector checkpoint."
        ),
    )

    parser.add_argument(
        "--backbone",
        required=True,
        choices=[
            "imagenet",
            "chest_xray",
        ],
        help=(
            "Backbone used by detector."
        ),
    )

    parser.add_argument(
        "--test-dir",
        default=DEFAULT_TEST_DIR,
        help=(
            "Directory containing test DICOM files."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path."
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=(
            "Detection score threshold. "
            f"Default: {SCORE_THRESHOLD:.4f}"
        ),
    )

    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=NMS_THRESHOLD,
        help=(
            "NMS threshold. "
            f"Default: {NMS_THRESHOLD:.4f}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Inference batch size. "
            "Default: 8."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help=(
            "DataLoader workers. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Example: --max-images 50."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be >= 0."
        )

    if not (
        0.0
        <= args.score_threshold
        <= 1.0
    ):
        raise ValueError(
            "--score-threshold must be in [0, 1]."
        )

    if not (
        0.0
        <= args.nms_threshold
        <= 1.0
    ):
        raise ValueError(
            "--nms-threshold must be in [0, 1]."
        )

    if (
        args.max_images is not None
        and args.max_images < 1
    ):
        raise ValueError(
            "--max-images must be >= 1."
        )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[LOG] Device: {device}"
    )

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    create_submission(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        test_dir=args.test_dir,
        output_path=args.output,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()