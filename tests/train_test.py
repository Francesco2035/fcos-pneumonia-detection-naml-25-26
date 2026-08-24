import torch

from src.models.detector import DetectionFramework
from src.detection_loss import DetectionLoss
from src.models.target_generator import TargetGenerator
from src.inference import DetectionPostProcessor
from src.evaluate import DetectionEvaluator

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)
from src.datasets.split import create_train_val_split

from src.train import Trainer


CSV_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

TRAIN_DCM_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)


def test_trainer_split():

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    val_ratio = 0.2
    seed = 42

    # ---------------------------------------------------------
    # Create the two dataset objects
    #
    # They contain the same images, but use different transforms.
    # ---------------------------------------------------------

    train_dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_train_transforms(224),
    )

    val_dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(224),
    )

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------

    assert len(train_dataset) == len(val_dataset)

    # The datasets must contain the same images in the same order.
    train_ids = [
        path.stem
        for path in train_dataset.image_paths
    ]

    val_ids = [
        path.stem
        for path in val_dataset.image_paths
    ]

    assert train_ids == val_ids

    # ---------------------------------------------------------
    # Create split
    # ---------------------------------------------------------

    train_indices, val_indices = create_train_val_split(
        train_dataset,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)

    # ---------------------------------------------------------
    # Check that train and validation do not overlap
    # ---------------------------------------------------------

    overlap = train_indices_set.intersection(
        val_indices_set
    )

    assert len(overlap) == 0, (
        f"Train/validation overlap detected: {overlap}"
    )

    # ---------------------------------------------------------
    # Check that train + validation cover the whole dataset
    # ---------------------------------------------------------

    union = train_indices_set.union(
        val_indices_set
    )

    expected_indices = set(
        range(len(train_dataset))
    )

    assert union == expected_indices

    # ---------------------------------------------------------
    # Get positive / negative indices
    # ---------------------------------------------------------

    positive_indices, negative_indices = (
    _get_positive_negative_indices(
        train_dataset
    )
    )

    positive_set = set(positive_indices)
    negative_set = set(negative_indices)


    # ---------------------------------------------------------
    # Check sizes
    # ---------------------------------------------------------

    # Train + validation must cover the whole dataset.
    assert len(train_indices) + len(val_indices) == (
    len(train_dataset)
    )

    # The split is stratified, so validation size is computed
    # separately for positive and negative samples.
    expected_positive_val = round(
    len(positive_indices) * val_ratio
    )

    expected_negative_val = round(
    len(negative_indices) * val_ratio
    )

    expected_val_size = (
    expected_positive_val
    + expected_negative_val
    )

    assert len(val_indices) == expected_val_size

    assert len(train_indices) == (
    len(train_dataset) - expected_val_size
    )


    # ---------------------------------------------------------
    # Verify positive / negative balance
    # ---------------------------------------------------------

    train_positive = sum(
    index in positive_set
    for index in train_indices
    )

    train_negative = sum(
    index in negative_set
    for index in train_indices
    )

    val_positive = sum(
    index in positive_set
    for index in val_indices
    )

    val_negative = sum(
    index in negative_set
    for index in val_indices
    )

    # Every positive image must be in exactly one split.
    assert train_positive + val_positive == (
    len(positive_indices)
    )

    # Every negative image must be in exactly one split.
    assert train_negative + val_negative == (
    len(negative_indices)
    )
    # ---------------------------------------------------------
    # Create DataLoaders exactly as Trainer does
    # ---------------------------------------------------------

    train_loader = train_dataset.get_dataloader(
        batch_size=1,
        shuffle=True,
        indices=train_indices,
    )

    val_loader = val_dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
        indices=val_indices,
    )

    # ---------------------------------------------------------
    # Check DataLoader lengths
    #
    # With batch_size=1, number of batches == number of samples.
    # ---------------------------------------------------------

    assert len(train_loader) == len(train_indices)
    assert len(val_loader) == len(val_indices)

    # ---------------------------------------------------------
    # Check one train batch
    # ---------------------------------------------------------

    train_images, train_targets = next(
        iter(train_loader)
    )

    assert train_images.shape[0] == 1
    assert train_images.shape[1:] == (
        3,
        224,
        224,
    )

    assert len(train_targets) == 1

    # ---------------------------------------------------------
    # Check one validation batch
    # ---------------------------------------------------------

    val_images, val_targets = next(
        iter(val_loader)
    )

    assert val_images.shape[0] == 1
    assert val_images.shape[1:] == (
        3,
        224,
        224,
    )

    assert len(val_targets) == 1

    # ---------------------------------------------------------
    # Print diagnostics
    # ---------------------------------------------------------

    print("\nTrainer split test passed.")

    print(f"Dataset size:       {len(train_dataset)}")
    print(f"Train size:         {len(train_indices)}")
    print(f"Validation size:    {len(val_indices)}")

    print()
    print(f"Train positive:     {train_positive}")
    print(f"Train negative:     {train_negative}")
    print(f"Validation positive:{val_positive}")
    print(f"Validation negative:{val_negative}")

    print()
    print(f"Train batches:      {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")


def _get_positive_negative_indices(dataset):
    """
    Helper used only by this test.

    It follows the same definition used by split.py.
    """

    positive_indices = []
    negative_indices = []

    for index in range(len(dataset)):

        patient_id = dataset.image_paths[index].stem

        boxes = dataset.annotations[
            patient_id
        ]["boxes"]

        if len(boxes) > 0:
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    return positive_indices, negative_indices


if __name__ == "__main__":
    test_trainer_split()