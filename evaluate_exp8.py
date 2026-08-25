import argparse

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
    EVAL_IOU_THRESHOLD,
    AR_MAX_DETECTIONS,
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

from src.evaluate import (
    DetectionEvaluator,
)


CHECKPOINT = (
    "checkpoints/exp8/best.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exp8 checkpoint without training."
        )
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=(
            "Inference score threshold. "
            f"Default: {SCORE_THRESHOLD}"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help=(
            "Detector checkpoint."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=VAL_NUM_WORKERS,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 75)
    print("EXP8 EVALUATION")
    print("=" * 75)

    print(
        f"Checkpoint:          {args.checkpoint}"
    )

    print(
        "Backbone:            Chest-Xray pretrained ResNet-50"
    )

    print(
        f"Image size:          {IMAGE_SIZE}"
    )

    print(
        f"Batch size:          {BATCH_SIZE}"
    )

    print(
        f"Score threshold:     {args.score_threshold:.2f}"
    )

    print(
        f"NMS threshold:       {NMS_THRESHOLD:.2f}"
    )

    print(
        f"Evaluation IoU:      {EVAL_IOU_THRESHOLD:.2f}"
    )

    print(
        f"AR max detections:   {AR_MAX_DETECTIONS}"
    )

    print(
        f"Validation ratio:    {VAL_RATIO}"
    )

    print(
        f"Seed:                {SEED}"
    )

    print(
        f"Device:              {device}"
    )

    print("=" * 75)

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\n[LOG] Creating validation dataset..."
    )

    val_dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    # ========================================================
    # Exact same split
    # ========================================================

    print(
        "[LOG] Recreating validation split..."
    )

    _, val_indices = create_train_val_split(
        val_dataset,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    print(
        f"[LOG] Validation images: "
        f"{len(val_indices)}"
    )

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

    print(
        "\n[LOG] Creating detector..."
    )

    model = DetectionFramework(
        path_model=(
            RESNET50_CHEST_XRAY_CHECKPOINT
        ),
    ).to(device)

    # ========================================================
    # Load checkpoint
    # ========================================================

    print(
        "[LOG] Loading checkpoint..."
    )

    checkpoint = torch.load(
        args.checkpoint,
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
        "[LOG] Detector loaded successfully."
    )

    # ========================================================
    # Postprocessor
    # ========================================================

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=(
                args.score_threshold
            ),
            nms_threshold=NMS_THRESHOLD,
        )
    )

    print()
    print(
        "[LOG] Post-processing:"
    )

    print(
        f"      score threshold = "
        f"{args.score_threshold:.2f}"
    )

    print(
        f"      NMS threshold   = "
        f"{NMS_THRESHOLD:.2f}"
    )

    # ========================================================
    # Evaluator
    # ========================================================

    evaluator = DetectionEvaluator(
        model=model,
        postprocessor=postprocessor,
        device=device,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print(
        "[LOG] Starting validation..."
    )

    metrics = evaluator.evaluate(
        val_loader
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 75)
    print("EXP8 RESULTS")
    print("=" * 75)

    for key, value in metrics.items():

        if isinstance(
            value,
            (float, int),
        ):
            print(
                f"{key:25s}: "
                f"{float(value):.6f}"
            )

    print("=" * 75)


if __name__ == "__main__":
    main()