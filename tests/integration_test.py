import torch

from src.models.detector import DetectionFramework
from src.models.target_generator import TargetGenerator
from src.detection_loss import DetectionLoss
from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import get_train_transforms


LEVELS = ("P3", "P4", "P5", "P6", "P7")

STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}



CSV_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

TRAIN_DCM_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)


def build_targets(model_predictions, dataset_targets, target_generator, device):
    """
    Build TargetGenerator outputs using the spatial shapes produced
    by the real DetectionFramework.
    """

    batch_targets = []

    batch_size = model_predictions["P3"]["classification"].shape[0]

    for b in range(batch_size):

        image_target = {}

        boxes = dataset_targets[b]["boxes"]

        # Convert BoundingBoxes tensor to a regular tensor.
        boxes = boxes.to(device)

        for level in LEVELS:

            classification = model_predictions[level]["classification"]

            _, _, height, width = classification.shape

            stride = STRIDES[level]

            target = target_generator.generate_targets(
                label_boxes=boxes,
                feature_shape=(height, width),
                stride=stride,
                device=device,
            )

            image_target[level] = target

        batch_targets.append(image_target)

    return batch_targets





def test_real_training_step():
    """
    Full integration test:

        real image
        -> real model
        -> real targets
        -> real loss
        -> backward
        -> optimizer.step()

    This test uses one real dataset sample.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_train_transforms(224),
    )

    assert len(dataset) > 0

    # ---------------------------------------------------------
    # Find the first image containing at least one GT box
    # ---------------------------------------------------------

    sample_index = None

    for i in range(len(dataset)):

        _, target = dataset[i]

        if target["boxes"].shape[0] > 0:
            sample_index = i
            break

    assert sample_index is not None, "No image with GT boxes found."

    images, dataset_targets = dataset[sample_index]

    print(f"Using dataset sample: {sample_index}")
    print(f"GT boxes: {dataset_targets['boxes'].shape[0]}")
    print(f"GT boxes:\n{dataset_targets['boxes']}")

    # ---------------------------------------------------------
    # Prepare batch
    # ---------------------------------------------------------

    images = images.unsqueeze(0).to(device)

    dataset_targets = [
        {
            "boxes": dataset_targets["boxes"],
            "labels": dataset_targets["labels"],
        }
    ]

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=None
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    predictions = model(images)

    # ---------------------------------------------------------
    # Verify prediction structure
    # ---------------------------------------------------------

    assert set(predictions.keys()) == set(LEVELS)

    for level in LEVELS:

        assert "classification" in predictions[level]
        assert "regression" in predictions[level]
        assert "centerness" in predictions[level]

        classification = predictions[level]["classification"]
        regression = predictions[level]["regression"]
        centerness = predictions[level]["centerness"]

        assert classification.ndim == 4
        assert regression.ndim == 4
        assert centerness.ndim == 4

        assert classification.shape[0] == 1
        assert regression.shape[0] == 1
        assert centerness.shape[0] == 1

        assert classification.shape[1] == 1
        assert regression.shape[1] == 4
        assert centerness.shape[1] == 1

        assert classification.shape[2:] == regression.shape[2:]
        assert classification.shape[2:] == centerness.shape[2:]

    # ---------------------------------------------------------
    # Target generation
    # ---------------------------------------------------------

    target_generator = TargetGenerator()

    targets = build_targets(
        predictions,
        dataset_targets,
        target_generator,
        device,
    )

    print("\nPositive locations:")

    for level in LEVELS:

        positive_count = (
            targets[0][level]["positive"]
            .sum()
            .item()
        )

    print(f"{level}: {positive_count}")

    for level in LEVELS:

        target = targets[0][level]

        positive = target["positive"]
        ltrb = target["ltrb"]
        centerness = target["centerness"]

        _, _, height, width = predictions[level]["classification"].shape

        assert positive.shape == (height, width)
        assert ltrb.shape == (height, width, 4)
        assert centerness.shape == (height, width)

        assert positive.dtype == torch.bool

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = DetectionLoss()

    losses = criterion(
        predictions,
        targets,
    )

    # ---------------------------------------------------------
    # Print loss components
    # ---------------------------------------------------------

    print("\nLoss components:")

    print(f"Center:      {losses['center'].item():.6f}")
    print(f"Regression:  {losses['regression'].item():.6f}")
    print(f"Centerness:  {losses['centerness'].item():.6f}")
    print(f"Total:       {losses['total'].item():.6f}")

    print("\nLoss per level:")

    for level in ("P3", "P4", "P5", "P6", "P7"):
        result = losses["levels"][level][0]

        print(
            f"{level}: "
            f"positive={result['num_positive'].item()}, "
            f"center={result['center'].item():.6f}, "
            f"regression={result['regression'].item():.6f}, "
            f"centerness={result['centerness'].item():.6f}"
        )



    assert "total" in losses
    assert "center" in losses
    assert "regression" in losses
    assert "centerness" in losses

    total_loss = losses["total"]

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert total_loss.item() >= 0.0

    # ---------------------------------------------------------
    # Check individual loss components
    # ---------------------------------------------------------

    assert torch.isfinite(losses["center"])
    assert torch.isfinite(losses["regression"])
    assert torch.isfinite(losses["centerness"])

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    optimizer.zero_grad()

    total_loss.backward()

    # ---------------------------------------------------------
    # Verify gradients
    # ---------------------------------------------------------

    heads = {
        "P3": model.head3,
        "P4": model.head4,
        "P5": model.head5,
        "P6": model.head6,
        "P7": model.head7,
    }

    for level, head in heads.items():

        classification_grad = (
            head.classification_conv.weight.grad
        )

        regression_grad = (
            head.regression_conv.weight.grad
        )

        centerness_grad = (
            head.centerness_conv.weight.grad
        )

        assert classification_grad is not None
        assert regression_grad is not None
        assert centerness_grad is not None

    # ---------------------------------------------------------
    # Optimizer step
    # ---------------------------------------------------------

    optimizer.step()

    print("\nIntegration test passed.")
    print(f"Device: {device}")
    print(f"Total loss: {total_loss.item():.6f}")

    for level in LEVELS:

        positive_count = (
            targets[0][level]["positive"]
            .sum()
            .item()
        )

        print(
            f"{level}: "
            f"{positive_count} positive locations"
        )


if __name__ == "__main__":
    test_real_training_step()