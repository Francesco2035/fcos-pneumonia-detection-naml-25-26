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


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = "checkpoints/exp2/best.pt"

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


# =========================================================
# CHECKPOINT
# =========================================================

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


# =========================================================
# TEST
# =========================================================

def test_classification_bias():

    print()
    print("=" * 70)
    print("Classification bias diagnostics")
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

    # =========================================================
    # HEAD BIAS SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("Classification head biases")
    print("=" * 70)

    for level_number, level in enumerate(
        LEVELS,
        start=3,
    ):

        head = getattr(
            model,
            f"head{level_number}",
        )

        bias = (
            head.classification_conv.bias
            .detach()
            .flatten()
        )

        if bias.numel() != 1:

            print(
                f"{level}: "
                f"WARNING - classification "
                f"bias has shape {tuple(bias.shape)}"
            )

        bias_value = bias.mean().item()

        probability = torch.sigmoid(
            bias
        ).mean().item()

        print()
        print(
            f"{level}"
        )

        print(
            f"  bias:               "
            f"{bias_value:.8f}"
        )

        print(
            f"  sigmoid(bias):      "
            f"{probability:.8f}"
        )

    # =========================================================
    # FIND POSITIVE IMAGES
    # =========================================================

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

    if len(positive_indices) == 0:

        raise RuntimeError(
            "No positive images found."
        )

    print()
    print(
        f"[TEST] Positive images: "
        f"{len(positive_indices)}"
    )

    # =========================================================
    # LOGIT STATISTICS ON POSITIVE LOCATIONS
    # =========================================================

    statistics = {}

    for level in LEVELS:

        statistics[level] = {
            "count": 0,
            "logit_sum": 0.0,
            "logit_min": float("inf"),
            "logit_max": float("-inf"),
            "probability_sum": 0.0,
            "probability_min": float("inf"),
            "probability_max": float("-inf"),
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

            image = (
                image
                .unsqueeze(0)
                .to(device)
            )

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

                # -------------------------------------------------
                # Generate positive mask
                # -------------------------------------------------

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

                positive_logits = (
                    classification[
                        positive
                    ]
                )

                if positive_logits.numel() == 0:
                    continue

                positive_probabilities = (
                    torch.sigmoid(
                        positive_logits
                    )
                )

                # -------------------------------------------------
                # Accumulate
                # -------------------------------------------------

                stats = statistics[level]

                stats["count"] += (
                    positive_logits.numel()
                )

                stats["logit_sum"] += (
                    positive_logits
                    .sum()
                    .item()
                )

                stats["logit_min"] = min(
                    stats["logit_min"],
                    positive_logits.min().item(),
                )

                stats["logit_max"] = max(
                    stats["logit_max"],
                    positive_logits.max().item(),
                )

                stats[
                    "probability_sum"
                ] += (
                    positive_probabilities
                    .sum()
                    .item()
                )

                stats[
                    "probability_min"
                ] = min(
                    stats[
                        "probability_min"
                    ],
                    positive_probabilities
                    .min()
                    .item(),
                )

                stats[
                    "probability_max"
                ] = max(
                    stats[
                        "probability_max"
                    ],
                    positive_probabilities
                    .max()
                    .item(),
                )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                image_number % 10 == 0
                or image_number
                == len(positive_indices)
            ):

                print(
                    f"[TEST] "
                    f"Image {image_number}/"
                    f"{len(positive_indices)}"
                )

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print(
        "Positive-location classification statistics"
    )
    print("=" * 70)

    for level in LEVELS:

        stats = statistics[level]

        count = stats["count"]

        print()
        print(
            f"{level}"
        )

        if count == 0:

            print(
                "  No positive locations."
            )

            continue

        mean_logit = (
            stats["logit_sum"]
            / count
        )

        mean_probability = (
            stats["probability_sum"]
            / count
        )

        print(
            f"  Count:               "
            f"{count}"
        )

        print(
            f"  Mean logit:          "
            f"{mean_logit:.8f}"
        )

        print(
            f"  Min logit:           "
            f"{stats['logit_min']:.8f}"
        )

        print(
            f"  Max logit:           "
            f"{stats['logit_max']:.8f}"
        )

        print(
            f"  Mean probability:    "
            f"{mean_probability:.8f}"
        )

        print(
            f"  Min probability:     "
            f"{stats['probability_min']:.8f}"
        )

        print(
            f"  Max probability:     "
            f"{stats['probability_max']:.8f}"
        )

    print()
    print("=" * 70)
    print(
        "Classification bias diagnostics completed."
    )
    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    test_classification_bias()