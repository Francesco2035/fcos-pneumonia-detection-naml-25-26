import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_test_transforms,
)

from src.models.detector import (
    DetectionFramework,
)

from src.models.target_generator import (
    TargetGenerator,
)


CHECKPOINT_PATH = (
    "checkpoints/exp1/best.pt"
)

NUM_IMAGES = 100


LEVELS = (
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)


STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}


def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

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
            "Could not find model weights "
            "in checkpoint."
        )


def test_classification_branch():

    print()
    print("=" * 70)
    print("Classification branch diagnostics")
    print("=" * 70)

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
    # Target generator
    # ---------------------------------------------------------

    target_generator = TargetGenerator()

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(device)

    load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    model.eval()

    print(
        f"[TEST] Loaded checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # Positive images
    # ---------------------------------------------------------

    positive_indices = []

    for index in range(
        len(dataset)
    ):

        patient_id = (
            dataset.image_paths[index].stem
        )

        boxes = dataset.annotations[
            patient_id
        ]["boxes"]

        if len(boxes) > 0:

            positive_indices.append(
                index
            )

        if len(positive_indices) >= NUM_IMAGES:
            break

    print(
        f"[TEST] Positive images: "
        f"{len(positive_indices)}"
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    stats = {}

    for level in LEVELS:

        stats[level] = {
            "positive_sum": 0.0,
            "positive_count": 0,
            "negative_sum": 0.0,
            "negative_count": 0,
        }

    # =========================================================
    # Evaluate
    # =========================================================

    with torch.no_grad():

        for image_number, dataset_index in enumerate(
            positive_indices,
            start=1,
        ):

            image, target = dataset[
                dataset_index
            ]

            image = image.unsqueeze(
                0
            ).to(device)

            gt_boxes = (
                target["boxes"]
                .to(device)
                .float()
            )

            predictions = model(
                image
            )

            # -------------------------------------------------
            # Every FPN level
            # -------------------------------------------------

            for level in LEVELS:

                stride = STRIDES[level]

                classification = (
                    predictions[level]
                    ["classification"][0, 0]
                )

                height, width = (
                    classification.shape
                )

                targets = (
                    target_generator
                    .generate_targets(
                        label_boxes=gt_boxes,
                        feature_shape=(
                            height,
                            width,
                        ),
                        stride=stride,
                        device=device,
                    )
                )

                positive = (
                    targets["positive"]
                )

                probabilities = (
                    torch.sigmoid(
                        classification
                    )
                )

                positive_probs = (
                    probabilities[
                        positive
                    ]
                )

                negative_probs = (
                    probabilities[
                        ~positive
                    ]
                )

                if positive_probs.numel() > 0:

                    stats[level][
                        "positive_sum"
                    ] += (
                        positive_probs
                        .sum()
                        .item()
                    )

                    stats[level][
                        "positive_count"
                    ] += (
                        positive_probs
                        .numel()
                    )

                if negative_probs.numel() > 0:

                    stats[level][
                        "negative_sum"
                    ] += (
                        negative_probs
                        .sum()
                        .item()
                    )

                    stats[level][
                        "negative_count"
                    ] += (
                        negative_probs
                        .numel()
                    )

            if (
                image_number % 10 == 0
                or image_number == len(
                    positive_indices
                )
            ):

                print(
                    f"[TEST] "
                    f"Image {image_number}/"
                    f"{len(positive_indices)}"
                )

    # =========================================================
    # Results
    # =========================================================

    print()
    print("=" * 70)
    print("Classification probabilities")
    print("=" * 70)

    for level in LEVELS:

        positive_count = stats[level][
            "positive_count"
        ]

        negative_count = stats[level][
            "negative_count"
        ]

        if positive_count > 0:

            positive_mean = (
                stats[level][
                    "positive_sum"
                ]
                / positive_count
            )

        else:

            positive_mean = 0.0

        if negative_count > 0:

            negative_mean = (
                stats[level][
                    "negative_sum"
                ]
                / negative_count
            )

        else:

            negative_mean = 0.0

        print()
        print(
            f"{level}"
        )

        print(
            f"  Positive locations: "
            f"{positive_count}"
        )

        print(
            f"  Negative locations: "
            f"{negative_count}"
        )

        print(
            f"  Positive mean prob: "
            f"{positive_mean:.8f}"
        )

        print(
            f"  Negative mean prob: "
            f"{negative_mean:.8f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    test_classification_branch()