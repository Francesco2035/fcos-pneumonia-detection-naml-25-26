import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
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

from src.models.detector import DetectionFramework
from src.inference import DetectionPostProcessor
from src.evaluate import DetectionEvaluator


def test_validation_diagnostics():

    print()
    print("=" * 60)
    print("Validation diagnostics test")
    print("=" * 60)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] Device: {device}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    # ---------------------------------------------------------
    # Create a small validation subset
    #
    # We only want to inspect the behaviour of the detector.
    # 100 batches are enough for this diagnostic.
    # ---------------------------------------------------------

    from torch.utils.data import DataLoader

    dataloader = dataset.get_dataloader(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
    )

    # ---------------------------------------------------------
    # Model
    #
    # Load the BEST checkpoint from exp1.
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(device)

    checkpoint_path = (
        "checkpoints/exp1/best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # ---------------------------------------------------------
    # Support both possible checkpoint formats.
    # ---------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    else:

        raise KeyError(
            "Could not find model weights in checkpoint."
        )

    print(
        f"[TEST] Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    model.eval()

    # ---------------------------------------------------------
    # Postprocessor
    # ---------------------------------------------------------

    postprocessor = DetectionPostProcessor(
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )

    # ---------------------------------------------------------
    # Evaluator
    # ---------------------------------------------------------

    evaluator = DetectionEvaluator(
        model=model,
        postprocessor=postprocessor,
        device=device,
    )

    # ---------------------------------------------------------
    # Run only 100 validation batches
    # ---------------------------------------------------------

    metrics = evaluator.evaluate(
        dataloader,
        max_batches=100,
    )

    # ---------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------

    print()
    print("[TEST] Metrics")
    print("-" * 60)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    print("-" * 60)

    print(
        "[TEST] Validation diagnostics completed."
    )


if __name__ == "__main__":
    test_validation_diagnostics()