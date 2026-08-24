import torch
import torch.nn.functional as F

from torchvision.ops import generalized_box_iou_loss
from torchvision.models.detection import _utils as det_utils

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
    "checkpoints/exp3/best.pt"
)

NUM_IMAGES = 10

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
# GRADIENT STATISTICS
# =========================================================

def gradient_norm(
    parameters,
):

    total_norm_squared = 0.0

    for parameter in parameters:

        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if grad.numel() == 0:
            continue

        total_norm_squared += (
            grad.norm(2).item() ** 2
        )

    return total_norm_squared ** 0.5


def collect_gradient_groups(model):

    groups = {
        "regression_tower": [],
        "regression_head": [],
        "fpn": [],
    }

    for level in LEVELS:

        head = getattr(
            model,
            f"head{level[-1]}",
        )

        groups[
            "regression_tower"
        ] += list(
            head.regression_feature_conv.parameters()
        )

        groups[
            "regression_tower"
        ] += list(
            head.regression_feature_norm.parameters()
        )

        groups[
            "regression_head"
        ] += list(
            head.regression_conv.parameters()
        )

    # ---------------------------------------------------------
    # Everything inside model.fpn is grouped as FPN.
    #
    # The previous diagnostic showed that this includes the
    # backbone components nested inside FPN, so we keep the same
    # grouping convention for direct A/B comparison.
    # ---------------------------------------------------------

    groups["fpn"] = list(
        model.fpn.parameters()
    )

    return groups


def group_gradient_norms(
    groups,
):

    return {
        name: gradient_norm(
            parameters
        )
        for name, parameters
        in groups.items()
    }


# =========================================================
# POSITIVE IMAGE SELECTION
# =========================================================

def find_positive_indices(
    dataset,
    num_images,
):

    indices = []

    for index in range(
        len(dataset)
    ):

        patient_id = (
            dataset.image_paths[
                index
            ].stem
        )

        boxes = (
            dataset.annotations[
                patient_id
            ]["boxes"]
        )

        if len(boxes) > 0:

            indices.append(
                index
            )

        if len(indices) >= num_images:
            break

    if len(indices) == 0:

        raise RuntimeError(
            "No positive images found."
        )

    return indices


# =========================================================
# FEATURE LOCATIONS
# =========================================================

def build_locations(
    height,
    width,
    stride,
    device,
):
    """
    Return FCOS point centers in image coordinates.

    Shape:
        [H, W, 2]

    Coordinates:
        x = (column + 0.5) * stride
        y = (row    + 0.5) * stride
    """

    ys = torch.arange(
        height,
        device=device,
        dtype=torch.float32,
    )

    xs = torch.arange(
        width,
        device=device,
        dtype=torch.float32,
    )

    grid_y, grid_x = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    x = (
        grid_x + 0.5
    ) * stride

    y = (
        grid_y + 0.5
    ) * stride

    return torch.stack(
        [
            x,
            y,
        ],
        dim=-1,
    )


# =========================================================
# TARGET LTRB -> XYXY
# =========================================================

def decode_target_ltrb(
    ltrb,
    positive,
    stride,
):
    """
    Decode the GT LTRB produced by our TargetGenerator into
    the actual matched GT boxes.

    This does NOT use prediction.

    For every positive location:

        x1 = x - left
        y1 = y - top
        x2 = x + right
        y2 = y + bottom
    """

    height, width = (
        positive.shape
    )

    locations = build_locations(
        height,
        width,
        stride,
        ltrb.device,
    )

    positive_locations = (
        locations[positive]
    )

    positive_ltrb = (
        ltrb[positive]
    )

    x = positive_locations[
        :, 0
    ]

    y = positive_locations[
        :, 1
    ]

    boxes = torch.stack(
        [
            x - positive_ltrb[:, 0],
            y - positive_ltrb[:, 1],
            x + positive_ltrb[:, 2],
            y + positive_ltrb[:, 3],
        ],
        dim=1,
    )

    return boxes


# =========================================================
# ANCHORS FOR TORCHVISION BOX LINEAR CODER
# =========================================================

def build_fcos_anchors(
    height,
    width,
    stride,
    device,
):
    """
    Reconstruct the single FCOS anchor/location used by the
    torchvision implementation.

    Anchor width = anchor height = stride.

    Anchor center:
        ((x + 0.5) * stride,
         (y + 0.5) * stride)
    """

    locations = build_locations(
        height,
        width,
        stride,
        device,
    )

    centers_x = (
        locations[..., 0]
    )

    centers_y = (
        locations[..., 1]
    )

    half_size = (
        float(stride) / 2.0
    )

    anchors = torch.stack(
        [
            centers_x - half_size,
            centers_y - half_size,
            centers_x + half_size,
            centers_y + half_size,
        ],
        dim=-1,
    )

    return anchors


# =========================================================
# REGRESSION OUTPUT
# =========================================================

def flatten_regression(
    regression,
):
    """
    [1, 4, H, W] -> [H, W, 4]
    """

    return (
        regression[0]
        .permute(1, 2, 0)
    )


# =========================================================
# A: CURRENT SMOOTH L1
# =========================================================

def compute_current_smooth_l1(
    regression,
    target_ltrb,
    positive,
    beta=1.0,
):
    """
    EXACT current regression formulation.

    No ReLU.
    No BoxLinearCoder.
    No decode.

        SmoothL1(
            predicted LTRB,
            target LTRB
        )
    """

    regression_pred = (
        flatten_regression(
            regression
        )
    )

    pred_pos = (
        regression_pred[
            positive
        ]
    )

    target_pos = (
        target_ltrb[
            positive
        ]
    )

    if not positive.any():

        return (
            regression.sum() * 0.0
        )

    return F.smooth_l1_loss(
        pred_pos,
        target_pos,
        beta=beta,
        reduction="sum",
    )


# =========================================================
# B: TORCHVISION FCOS + GIOU
# =========================================================

def compute_torchvision_giou(
    regression,
    target_ltrb,
    positive,
    stride,
):
    """
    Reproduce the regression objective used by the students'
    torchvision FCOS implementation:

        1. Apply ReLU to regression output.
        2. Treat output as normalized LTRB.
        3. Decode with BoxLinearCoder(normalize_by_size=True).
        4. Compare predicted XYXY boxes against GT XYXY boxes
           with generalized_box_iou_loss().
    """

    if not positive.any():

        return (
            regression.sum() * 0.0
        )

    regression_pred = (
        flatten_regression(
            regression
        )
    )

    H, W = positive.shape

    # ---------------------------------------------------------
    # Torchvision BoxLinearCoder
    # ---------------------------------------------------------

    box_coder = (
        det_utils.BoxLinearCoder(
            normalize_by_size=True
        )
    )

    # ---------------------------------------------------------
    # Build FCOS anchors
    # ---------------------------------------------------------

    anchors = build_fcos_anchors(
        H,
        W,
        stride,
        regression.device,
    )

    anchors_pos = (
        anchors[
            positive
        ]
    )

    # ---------------------------------------------------------
    # Torchvision regression head uses ReLU
    # ---------------------------------------------------------

    regression_normalized = F.relu(
        regression_pred
    )

    regression_normalized_pos = (
        regression_normalized[
            positive
        ]
    )

    # ---------------------------------------------------------
    # Decode predictions
    # ---------------------------------------------------------

    predicted_boxes = (
        box_coder.decode(
            regression_normalized_pos,
            anchors_pos,
        )
    )

    # ---------------------------------------------------------
    # Decode GT target into XYXY
    #
    # The TargetGenerator has already matched each positive
    # point to a specific GT and gives us its LTRB distances.
    # ---------------------------------------------------------

    target_boxes = (
        decode_target_ltrb(
            target_ltrb,
            positive,
            stride,
        )
    )

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    if predicted_boxes.shape != target_boxes.shape:

        raise RuntimeError(
            "Predicted and target box shapes do not match: "
            f"{predicted_boxes.shape} vs "
            f"{target_boxes.shape}"
        )

    # ---------------------------------------------------------
    # GIoU loss
    # ---------------------------------------------------------

    return generalized_box_iou_loss(
        predicted_boxes,
        target_boxes,
        reduction="sum",
    )


# =========================================================
# TEST
# =========================================================

def test_regression_loss_ab():

    print()
    print("=" * 90)
    print(
        "REGRESSION LOSS A/B TEST"
    )
    print("=" * 90)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] Device: {device}"
    )

    print(
        f"[TEST] Image size: {IMAGE_SIZE}"
    )

    print(
        f"[TEST] Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    positive_indices = (
        find_positive_indices(
            dataset,
            NUM_IMAGES,
        )
    )

    print(
        f"[TEST] Positive images: "
        f"{len(positive_indices)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(device)

    load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    model.train()

    groups = (
        collect_gradient_groups(
            model
        )
    )

    # -----------------------------------------------------
    # Target generator
    # -----------------------------------------------------

    target_generator = (
        TargetGenerator()
    )

    # =====================================================
    # ACCUMULATORS
    # =====================================================

    losses = {
        "smooth_l1": 0.0,
        "torchvision_giou": 0.0,
    }

    gradients = {
        "smooth_l1": 0.0,
        "torchvision_giou": 0.0,
    }

    group_gradients = {
        "smooth_l1": {
            group: 0.0
            for group in groups
        },
        "torchvision_giou": {
            group: 0.0
            for group in groups
        },
    }

    positive_counts = {
        level: 0.0
        for level in LEVELS
    }

    level_losses = {
        "smooth_l1": {
            level: 0.0
            for level in LEVELS
        },
        "torchvision_giou": {
            level: 0.0
            for level in LEVELS
        },
    }

    # =====================================================
    # IMAGE LOOP
    # =====================================================

    for image_number, dataset_index in enumerate(
        positive_indices,
        start=1,
    ):

        print()
        print("=" * 90)
        print(
            f"IMAGE {image_number}/"
            f"{len(positive_indices)}"
        )
        print("=" * 90)

        # -------------------------------------------------
        # Load image
        # -------------------------------------------------

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

        # =================================================
        # A — CURRENT SMOOTH L1
        # =================================================

        model.zero_grad(
            set_to_none=True
        )

        predictions = model(
            image
        )

        smooth_l1_total = (
            predictions[
                "P3"
            ][
                "classification"
            ].sum()
            * 0.0
        )

        image_level_smooth = {}

        for level in LEVELS:

            predictions_level = (
                predictions[level]
            )

            _, _, H, W = (
                predictions_level[
                    "regression"
                ].shape
            )

            target_level = (
                target_generator.generate_targets(
                    label_boxes=gt_boxes,
                    feature_shape=(
                        H,
                        W,
                    ),
                    stride=STRIDES[level],
                    device=device,
                )
            )

            positive = (
                target_level[
                    "positive"
                ]
            )

            ltrb = (
                target_level[
                    "ltrb"
                ]
                .to(
                    device=device,
                    dtype=predictions_level[
                        "regression"
                    ].dtype,
                )
            )

            positive_counts[
                level
            ] += (
                positive.sum()
                .item()
            )

            level_loss = (
                compute_current_smooth_l1(
                    predictions_level[
                        "regression"
                    ],
                    ltrb,
                    positive,
                )
            )

            image_level_smooth[
                level
            ] = level_loss

            smooth_l1_total = (
                smooth_l1_total
                + level_loss
            )

        # -------------------------------------------------
        # Global foreground normalization
        # -------------------------------------------------

        total_positive = sum(
            positive_counts[level]
            for level in LEVELS
        )

        # IMPORTANT:
        #
        # For this image we use its own foreground count.
        # This is the same global-over-FPN-level normalization
        # used by DetectionLoss.
        #

        image_positive_count = sum(
            int(
                (
                    target_generator
                    .generate_targets(
                        label_boxes=gt_boxes,
                        feature_shape=(
                            predictions[level][
                                "regression"
                            ].shape[-2],
                            predictions[level][
                                "regression"
                            ].shape[-1],
                        ),
                        stride=STRIDES[level],
                        device=device,
                    )[
                        "positive"
                    ]
                    .sum()
                    .item()
                )
            )
            for level in LEVELS
        )

        image_normalizer = max(
            1,
            image_positive_count,
        )

        smooth_l1_total = (
            smooth_l1_total
            / float(
                image_normalizer
            )
        )

        # -------------------------------------------------
        # Backward A
        # -------------------------------------------------

        smooth_l1_total.backward()

        smooth_global_grad = (
            gradient_norm(
                model.parameters()
            )
        )

        smooth_group_grads = (
            group_gradient_norms(
                groups
            )
        )

        losses[
            "smooth_l1"
        ] += smooth_l1_total.item()

        gradients[
            "smooth_l1"
        ] += smooth_global_grad

        for group in groups:

            group_gradients[
                "smooth_l1"
            ][group] += (
                smooth_group_grads[
                    group
                ]
            )

        for level in LEVELS:

            level_losses[
                "smooth_l1"
            ][level] += (
                image_level_smooth[
                    level
                ].item()
                / float(
                    image_normalizer
                )
            )

        print()
        print(
            "[A] CURRENT SMOOTH L1"
        )

        print(
            f"  loss:          "
            f"{smooth_l1_total.item():.8e}"
        )

        print(
            f"  global grad:   "
            f"{smooth_global_grad:.8e}"
        )

        # =================================================
        # B — TORCHVISION FCOS GIoU
        # =================================================

        model.zero_grad(
            set_to_none=True
        )

        predictions = model(
            image
        )

        giou_total = (
            predictions[
                "P3"
            ][
                "classification"
            ].sum()
            * 0.0
        )

        image_level_giou = {}

        # -------------------------------------------------
        # Recompute targets for the same image
        # -------------------------------------------------

        total_positive_this_image = 0

        targets_by_level = {}

        for level in LEVELS:

            predictions_level = (
                predictions[level]
            )

            _, _, H, W = (
                predictions_level[
                    "regression"
                ].shape
            )

            target_level = (
                target_generator.generate_targets(
                    label_boxes=gt_boxes,
                    feature_shape=(
                        H,
                        W,
                    ),
                    stride=STRIDES[level],
                    device=device,
                )
            )

            targets_by_level[
                level
            ] = target_level

            total_positive_this_image += int(
                target_level[
                    "positive"
                ]
                .sum()
                .item()
            )

        giou_normalizer = max(
            1,
            total_positive_this_image,
        )

        # -------------------------------------------------
        # Per-level GIoU
        # -------------------------------------------------

        for level in LEVELS:

            predictions_level = (
                predictions[level]
            )

            positive = (
                targets_by_level[
                    level
                ][
                    "positive"
                ]
            )

            ltrb = (
                targets_by_level[
                    level
                ][
                    "ltrb"
                ]
                .to(
                    device=device,
                    dtype=predictions_level[
                        "regression"
                    ].dtype,
                )
            )

            level_loss = (
                compute_torchvision_giou(
                    predictions_level[
                        "regression"
                    ],
                    ltrb,
                    positive,
                    STRIDES[level],
                )
            )

            image_level_giou[
                level
            ] = level_loss

            giou_total = (
                giou_total
                + level_loss
            )

        giou_total = (
            giou_total
            / float(
                giou_normalizer
            )
        )

        # -------------------------------------------------
        # Backward B
        # -------------------------------------------------

        giou_total.backward()

        giou_global_grad = (
            gradient_norm(
                model.parameters()
            )
        )

        giou_group_grads = (
            group_gradient_norms(
                groups
            )
        )

        losses[
            "torchvision_giou"
        ] += giou_total.item()

        gradients[
            "torchvision_giou"
        ] += giou_global_grad

        for group in groups:

            group_gradients[
                "torchvision_giou"
            ][group] += (
                giou_group_grads[
                    group
                ]
            )

        for level in LEVELS:

            level_losses[
                "torchvision_giou"
            ][level] += (
                image_level_giou[
                    level
                ].item()
                / float(
                    giou_normalizer
                )
            )

        print()
        print(
            "[B] TORCHVISION FCOS GIoU"
        )

        print(
            f"  loss:          "
            f"{giou_total.item():.8e}"
        )

        print(
            f"  global grad:   "
            f"{giou_global_grad:.8e}"
        )

        # -------------------------------------------------
        # Clear gradients
        # -------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    n = len(
        positive_indices
    )

    print()
    print("=" * 100)
    print(
        "FINAL A/B REGRESSION SUMMARY"
    )
    print("=" * 100)

    print()
    print(
        f"{'metric':40s}"
        f"{'Smooth L1':>22s}"
        f"{'Torchvision GIoU':>22s}"
        f"{'GIoU / SmoothL1':>22s}"
    )

    print("-" * 110)

    mean_smooth_loss = (
        losses["smooth_l1"]
        / n
    )

    mean_giou_loss = (
        losses["torchvision_giou"]
        / n
    )

    mean_smooth_grad = (
        gradients["smooth_l1"]
        / n
    )

    mean_giou_grad = (
        gradients["torchvision_giou"]
        / n
    )

    print(
        f"{'global regression loss':40s}"
        f"{mean_smooth_loss:22.8e}"
        f"{mean_giou_loss:22.8e}"
        f"{mean_giou_loss / max(mean_smooth_loss, 1e-12):22.8e}"
    )

    print(
        f"{'global regression gradient':40s}"
        f"{mean_smooth_grad:22.8e}"
        f"{mean_giou_grad:22.8e}"
        f"{mean_giou_grad / max(mean_smooth_grad, 1e-12):22.8e}"
    )

    # =====================================================
    # PER-LEVEL LOSSES
    # =====================================================

    print()
    print(
        "PER-LEVEL REGRESSION LOSSES"
    )

    print("-" * 100)

    print(
        f"{'level':10s}"
        f"{'Smooth L1':>22s}"
        f"{'GIoU':>22s}"
        f"{'GIoU / SmoothL1':>22s}"
    )

    print("-" * 100)

    for level in LEVELS:

        smooth = (
            level_losses[
                "smooth_l1"
            ][level]
            / n
        )

        giou = (
            level_losses[
                "torchvision_giou"
            ][level]
            / n
        )

        ratio = (
            giou / smooth
            if abs(smooth) > 1e-12
            else 0.0
        )

        print(
            f"{level:10s}"
            f"{smooth:22.8e}"
            f"{giou:22.8e}"
            f"{ratio:22.8e}"
        )

    # =====================================================
    # GRADIENT GROUPS
    # =====================================================

    print()
    print(
        "GRADIENTS BY MODEL COMPONENT"
    )

    print("-" * 100)

    print(
        f"{'group':28s}"
        f"{'Smooth L1':>22s}"
        f"{'GIoU':>22s}"
        f"{'GIoU / SmoothL1':>22s}"
    )

    print("-" * 100)

    for group in groups:

        smooth = (
            group_gradients[
                "smooth_l1"
            ][group]
            / n
        )

        giou = (
            group_gradients[
                "torchvision_giou"
            ][group]
            / n
        )

        ratio = (
            giou / smooth
            if smooth > 1e-12
            else 0.0
        )

        print(
            f"{group:28s}"
            f"{smooth:22.8e}"
            f"{giou:22.8e}"
            f"{ratio:22.8e}"
        )

    print()
    print("=" * 100)
    print(
        "NO optimizer.step() was performed."
    )
    print(
        "The checkpoint was not modified."
    )
    print("=" * 100)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    test_regression_loss_ab()