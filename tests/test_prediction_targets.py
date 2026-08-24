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

CHECKPOINT_PATH = (
    "checkpoints/exp1/best.pt"
)

NUM_IMAGES = 5

MAX_POSITIVE_LOCATIONS_PER_LEVEL = 5

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
# CHECKPOINT LOADING
# =========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load model weights from checkpoint.
    """

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
            "Could not find model weights in checkpoint."
        )


# =========================================================
# BOX DECODING
# =========================================================

def decode_ltrb(
    ltrb,
    x,
    y,
):
    """
    Decode [left, top, right, bottom]
    from a feature-map location into
    [x1, y1, x2, y2].
    """

    x1 = x - ltrb[0]
    y1 = y - ltrb[1]
    x2 = x + ltrb[2]
    y2 = y + ltrb[3]

    return torch.stack(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )


# =========================================================
# IOU
# =========================================================

def compute_iou(
    box_a,
    box_b,
):
    """
    Compute IoU between two boxes.
    """

    x1 = torch.maximum(
        box_a[0],
        box_b[0],
    )

    y1 = torch.maximum(
        box_a[1],
        box_b[1],
    )

    x2 = torch.minimum(
        box_a[2],
        box_b[2],
    )

    y2 = torch.minimum(
        box_a[3],
        box_b[3],
    )

    intersection = (
        (x2 - x1).clamp(min=0)
        *
        (y2 - y1).clamp(min=0)
    )

    area_a = (
        (box_a[2] - box_a[0]).clamp(min=0)
        *
        (box_a[3] - box_a[1]).clamp(min=0)
    )

    area_b = (
        (box_b[2] - box_b[0]).clamp(min=0)
        *
        (box_b[3] - box_b[1]).clamp(min=0)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return (
        intersection
        / union.clamp(min=1e-8)
    )


# =========================================================
# TEST
# =========================================================

def test_prediction_targets():

    print()
    print("=" * 70)
    print("Prediction vs target diagnostics")
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
    # Find positive images
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
    # Inspect images
    # ========================================================

    for image_number, dataset_index in enumerate(
        positive_indices,
        start=1,
    ):

        # -----------------------------------------------------
        # Load image and GT
        # -----------------------------------------------------

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

        print()
        print("=" * 70)

        print(
            f"[IMAGE {image_number}/"
            f"{len(positive_indices)}] "
            f"dataset_index={dataset_index}"
        )

        print(
            f"GT boxes:\n"
            f"{gt_boxes.cpu()}"
        )

        # -----------------------------------------------------
        # Forward pass
        # -----------------------------------------------------

        with torch.no_grad():

            predictions = model(
                image
            )

        # -----------------------------------------------------
        # Inspect FPN levels
        # -----------------------------------------------------

        for level in LEVELS:

            stride = STRIDES[level]

            pred_level = predictions[level]

            classification = (
                pred_level[
                    "classification"
                ][0, 0]
            )

            regression = (
                pred_level[
                    "regression"
                ][0]
            )

            centerness = (
                pred_level[
                    "centerness"
                ][0, 0]
            )

            # -------------------------------------------------
            # Feature-map dimensions
            # -------------------------------------------------

            height, width = (
                classification.shape
            )

            # -------------------------------------------------
            # Generate GT targets
            # -------------------------------------------------

            targets = (
                target_generator.generate_targets(
                    label_boxes=gt_boxes,
                    feature_shape=(
                        height,
                        width,
                    ),
                    stride=stride,
                    device=device,
                )
            )

            positive_mask = targets[
                "positive"
            ]

            positive_indices_level = (
                positive_mask.nonzero(
                    as_tuple=False
                )
            )

            print()
            print(
                f"--- {level} "
                f"(stride={stride}) ---"
            )

            print(
                f"Feature map: "
                f"{height}x{width}"
            )

            print(
                f"Positive locations: "
                f"{positive_indices_level.shape[0]}"
            )

            # -------------------------------------------------
            # No positive locations
            # -------------------------------------------------

            if (
                positive_indices_level.shape[0]
                == 0
            ):
                continue

            # -------------------------------------------------
            # Select a few positive locations
            # -------------------------------------------------

            num_locations = min(
                positive_indices_level.shape[0],
                MAX_POSITIVE_LOCATIONS_PER_LEVEL,
            )

            selected_locations = (
                positive_indices_level[
                    :num_locations
                ]
            )

            # -------------------------------------------------
            # Print positive locations
            # -------------------------------------------------

            for location in (
                selected_locations
            ):

                y = int(
                    location[0].item()
                )

                x = int(
                    location[1].item()
                )

                # ---------------------------------------------
                # Feature-map location -> image coordinates
                # ---------------------------------------------

                x_img = (
                    x + 0.5
                ) * stride

                y_img = (
                    y + 0.5
                ) * stride

                # ---------------------------------------------
                # GT targets
                # ---------------------------------------------

                gt_ltrb = (
                    targets["ltrb"][y, x]
                )

                gt_centerness = (
                    targets[
                        "centerness"
                    ][y, x]
                )

                # ---------------------------------------------
                # Predictions
                # ---------------------------------------------

                pred_ltrb = (
                    regression[:, y, x]
                )

                pred_centerness_logit = (
                    centerness[y, x]
                )

                pred_centerness_prob = (
                    torch.sigmoid(
                        pred_centerness_logit
                    )
                )

                classification_logit = (
                    classification[y, x]
                )

                classification_prob = (
                    torch.sigmoid(
                        classification_logit
                    )
                )

                # ---------------------------------------------
                # Decode boxes
                # ---------------------------------------------

                gt_box = decode_ltrb(
                    gt_ltrb,
                    torch.tensor(
                        x_img,
                        device=device,
                    ),
                    torch.tensor(
                        y_img,
                        device=device,
                    ),
                )

                pred_box = decode_ltrb(
                    pred_ltrb,
                    torch.tensor(
                        x_img,
                        device=device,
                    ),
                    torch.tensor(
                        y_img,
                        device=device,
                    ),
                )

                # ---------------------------------------------
                # IoU
                # ---------------------------------------------

                iou = compute_iou(
                    gt_box,
                    pred_box,
                )

                # ---------------------------------------------
                # Print diagnostics
                # ---------------------------------------------

                print()
                print(
                    f"Location: "
                    f"(x={x}, y={y})"
                )

                print(
                    f"Image coord: "
                    f"({x_img:.2f}, "
                    f"{y_img:.2f})"
                )

                print(
                    f"GT LTRB:     "
                    f"{gt_ltrb.detach().cpu()}"
                )

                print(
                    f"Pred LTRB:   "
                    f"{pred_ltrb.detach().cpu()}"
                )

                print(
                    f"GT center:   "
                    f"{gt_centerness.item():.6f}"
                )

                print(
                    f"Pred center: "
                    f"{pred_centerness_prob.item():.6f}"
                )

                print(
                    f"Cls prob:    "
                    f"{classification_prob.item():.6f}"
                )

                print(
                    f"GT box:      "
                    f"{gt_box.detach().cpu()}"
                )

                print(
                    f"Pred box:    "
                    f"{pred_box.detach().cpu()}"
                )

                print(
                    f"IoU:         "
                    f"{iou.item():.6f}"
                )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    test_prediction_targets()