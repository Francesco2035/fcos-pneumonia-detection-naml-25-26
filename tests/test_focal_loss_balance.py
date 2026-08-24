import torch
import torch.nn.functional as F

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
# Configuration
# =========================================================

CHECKPOINT_PATH = (
    "checkpoints/exp1/best.pt"
)

NUM_IMAGES = 1000

ALPHA = 0.25
GAMMA = 2.0

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
# Checkpoint
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
# Focal loss without reduction
# =========================================================

def focal_loss_per_location(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    Reproduce DetectionLoss._focal_loss(),
    but keep one loss value per location.

    This is important because the normal implementation
    immediately calls .mean(), hiding the contribution
    of positive and negative locations separately.
    """

    targets = targets.to(
        dtype=logits.dtype
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    probabilities = torch.sigmoid(
        logits
    )

    p_t = (
        probabilities * targets
        +
        (1.0 - probabilities)
        * (1.0 - targets)
    )

    alpha_t = (
        alpha * targets
        +
        (1.0 - alpha)
        * (1.0 - targets)
    )

    loss = (
        alpha_t
        * (1.0 - p_t).pow(gamma)
        * bce
    )

    return loss


# =========================================================
# Test
# =========================================================

def test_focal_loss_balance():

    print()
    print("=" * 70)
    print("Focal loss balance diagnostics")
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
    # Select positive images
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

    if len(positive_indices) == 0:

        raise RuntimeError(
            "No positive images found."
        )

    print(
        f"[TEST] Positive images selected: "
        f"{len(positive_indices)}"
    )

    # ========================================================
    # Accumulators
    # ========================================================

    statistics = {}

    for level in LEVELS:

        statistics[level] = {

            # Number of locations
            "positive_count": 0,
            "negative_count": 0,

            # Sum of losses
            "positive_loss_sum": 0.0,
            "negative_loss_sum": 0.0,

            # Sum of probabilities
            "positive_probability_sum": 0.0,
            "negative_probability_sum": 0.0,

            # Correct positive/negative decisions
            "positive_correct": 0,
            "negative_correct": 0,
        }

    # ========================================================
    # Evaluate
    # ========================================================

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

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

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
                # Generate targets
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

                center_target = (
                    positive.to(
                        dtype=classification.dtype
                    )
                )

                # -------------------------------------------------
                # Focal loss per location
                # -------------------------------------------------

                losses = (
                    focal_loss_per_location(
                        logits=classification,
                        targets=center_target,
                        alpha=ALPHA,
                        gamma=GAMMA,
                    )
                )

                probabilities = (
                    torch.sigmoid(
                        classification
                    )
                )

                # -------------------------------------------------
                # Positive locations
                # -------------------------------------------------

                positive_losses = (
                    losses[
                        positive
                    ]
                )

                positive_probabilities = (
                    probabilities[
                        positive
                    ]
                )

                # -------------------------------------------------
                # Negative locations
                # -------------------------------------------------

                negative_mask = ~positive

                negative_losses = (
                    losses[
                        negative_mask
                    ]
                )

                negative_probabilities = (
                    probabilities[
                        negative_mask
                    ]
                )

                # -------------------------------------------------
                # Accumulate positive statistics
                # -------------------------------------------------

                if (
                    positive_losses.numel()
                    > 0
                ):

                    statistics[level][
                        "positive_count"
                    ] += (
                        positive_losses.numel()
                    )

                    statistics[level][
                        "positive_loss_sum"
                    ] += (
                        positive_losses
                        .sum()
                        .item()
                    )

                    statistics[level][
                        "positive_probability_sum"
                    ] += (
                        positive_probabilities
                        .sum()
                        .item()
                    )

                    statistics[level][
                        "positive_correct"
                    ] += (
                        (
                            positive_probabilities
                            >= 0.5
                        )
                        .sum()
                        .item()
                    )

                # -------------------------------------------------
                # Accumulate negative statistics
                # -------------------------------------------------

                if (
                    negative_losses.numel()
                    > 0
                ):

                    statistics[level][
                        "negative_count"
                    ] += (
                        negative_losses.numel()
                    )

                    statistics[level][
                        "negative_loss_sum"
                    ] += (
                        negative_losses
                        .sum()
                        .item()
                    )

                    statistics[level][
                        "negative_probability_sum"
                    ] += (
                        negative_probabilities
                        .sum()
                        .item()
                    )

                    statistics[level][
                        "negative_correct"
                    ] += (
                        (
                            negative_probabilities
                            < 0.5
                        )
                        .sum()
                        .item()
                    )

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                image_number % 100 == 0
                or image_number
                == len(positive_indices)
            ):

                print(
                    f"[TEST] "
                    f"Image {image_number}/"
                    f"{len(positive_indices)}"
                )

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 70)
    print("Focal loss balance summary")
    print("=" * 70)

    global_positive_loss = 0.0
    global_negative_loss = 0.0

    global_positive_count = 0
    global_negative_count = 0

    for level in LEVELS:

        positive_count = statistics[level][
            "positive_count"
        ]

        negative_count = statistics[level][
            "negative_count"
        ]

        positive_loss_sum = statistics[level][
            "positive_loss_sum"
        ]

        negative_loss_sum = statistics[level][
            "negative_loss_sum"
        ]

        positive_probability_sum = (
            statistics[level][
                "positive_probability_sum"
            ]
        )

        negative_probability_sum = (
            statistics[level][
                "negative_probability_sum"
            ]
        )

        positive_correct = statistics[level][
            "positive_correct"
        ]

        negative_correct = statistics[level][
            "negative_correct"
        ]

        if positive_count > 0:

            positive_loss_mean = (
                positive_loss_sum
                / positive_count
            )

            positive_probability_mean = (
                positive_probability_sum
                / positive_count
            )

            positive_accuracy = (
                positive_correct
                / positive_count
            )

        else:

            positive_loss_mean = 0.0
            positive_probability_mean = 0.0
            positive_accuracy = 0.0

        if negative_count > 0:

            negative_loss_mean = (
                negative_loss_sum
                / negative_count
            )

            negative_probability_mean = (
                negative_probability_sum
                / negative_count
            )

            negative_accuracy = (
                negative_correct
                / negative_count
            )

        else:

            negative_loss_mean = 0.0
            negative_probability_mean = 0.0
            negative_accuracy = 0.0

        total_loss_sum = (
            positive_loss_sum
            +
            negative_loss_sum
        )

        total_count = (
            positive_count
            +
            negative_count
        )

        total_loss_mean = (
            total_loss_sum
            / total_count
            if total_count > 0
            else 0.0
        )

        positive_loss_contribution = (
            positive_loss_sum
            / total_loss_sum
            if total_loss_sum > 0
            else 0.0
        )

        negative_loss_contribution = (
            negative_loss_sum
            / total_loss_sum
            if total_loss_sum > 0
            else 0.0
        )

        print()
        print(
            f"{level}"
        )

        print(
            f"  Positive count:          "
            f"{positive_count}"
        )

        print(
            f"  Negative count:          "
            f"{negative_count}"
        )

        print(
            f"  Positive mean loss:      "
            f"{positive_loss_mean:.8f}"
        )

        print(
            f"  Negative mean loss:      "
            f"{negative_loss_mean:.8f}"
        )

        print(
            f"  Total mean loss:         "
            f"{total_loss_mean:.8f}"
        )

        print(
            f"  Positive loss fraction:  "
            f"{100.0 * positive_loss_contribution:.2f}%"
        )

        print(
            f"  Negative loss fraction:  "
            f"{100.0 * negative_loss_contribution:.2f}%"
        )

        print(
            f"  Positive mean prob:       "
            f"{positive_probability_mean:.8f}"
        )

        print(
            f"  Negative mean prob:       "
            f"{negative_probability_mean:.8f}"
        )

        print(
            f"  Positive accuracy @0.5:  "
            f"{100.0 * positive_accuracy:.2f}%"
        )

        print(
            f"  Negative accuracy @0.5:  "
            f"{100.0 * negative_accuracy:.2f}%"
        )

        # -----------------------------------------------------
        # Global accumulation
        # -----------------------------------------------------

        global_positive_loss += (
            positive_loss_sum
        )

        global_negative_loss += (
            negative_loss_sum
        )

        global_positive_count += (
            positive_count
        )

        global_negative_count += (
            negative_count
        )

    # ========================================================
    # Global summary
    # ========================================================

    global_total_loss = (
        global_positive_loss
        +
        global_negative_loss
    )

    global_positive_fraction = (
        global_positive_loss
        / global_total_loss
        if global_total_loss > 0
        else 0.0
    )

    global_negative_fraction = (
        global_negative_loss
        / global_total_loss
        if global_total_loss > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("GLOBAL")
    print("=" * 70)

    print(
        f"Positive locations:       "
        f"{global_positive_count}"
    )

    print(
        f"Negative locations:       "
        f"{global_negative_count}"
    )

    print(
        f"Positive loss fraction:   "
        f"{100.0 * global_positive_fraction:.2f}%"
    )

    print(
        f"Negative loss fraction:   "
        f"{100.0 * global_negative_fraction:.2f}%"
    )

    if global_positive_count > 0:

        print(
            f"Positive mean loss:       "
            f"{global_positive_loss / global_positive_count:.8f}"
        )

    if global_negative_count > 0:

        print(
            f"Negative mean loss:       "
            f"{global_negative_loss / global_negative_count:.8f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    test_focal_loss_balance()