import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_train_transforms,
)

from src.models.detector import (
    DetectionFramework,
)

from src.models.target_generator import (
    TargetGenerator,
)

from src.detection_loss import (
    DetectionLoss,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_INDEX = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Same clipping used in the training loop.
GRADIENT_CLIP_MAX_NORM = 1.0


# =========================================================
# TENSOR STATISTICS
# =========================================================

def tensor_stats(tensor):

    tensor = tensor.detach()

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "abs_max": tensor.abs().max().item(),
    }


def print_tensor_stats(
    name,
    tensor,
):

    stats = tensor_stats(
        tensor
    )

    print(
        f"{name}"
    )

    print(
        f"  shape:   {tuple(tensor.shape)}"
    )

    print(
        f"  mean:    {stats['mean']:.8e}"
    )

    print(
        f"  std:     {stats['std']:.8e}"
    )

    print(
        f"  min:     {stats['min']:.8e}"
    )

    print(
        f"  max:     {stats['max']:.8e}"
    )

    print(
        f"  abs max: {stats['abs_max']:.8e}"
    )


# =========================================================
# PARAMETER STATISTICS
# =========================================================

def parameter_stats(
    parameter,
):

    tensor = parameter.detach()

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "abs_max": tensor.abs().max().item(),
        "norm": tensor.norm().item(),
    }


def print_parameter_stats(
    name,
    parameter,
):

    stats = parameter_stats(
        parameter
    )

    print(
        f"{name}"
    )

    print(
        f"  mean:    {stats['mean']:.8e}"
    )

    print(
        f"  std:     {stats['std']:.8e}"
    )

    print(
        f"  abs max: {stats['abs_max']:.8e}"
    )

    print(
        f"  norm:    {stats['norm']:.8e}"
    )


# =========================================================
# CHECKPOINT LOADING
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
# BUILD TARGETS
# =========================================================

def build_targets(
    target_generator,
    predictions,
    dataset_targets,
    device,
):

    batch_targets = []

    strides = {
        "P3": 8,
        "P4": 16,
        "P5": 32,
        "P6": 64,
        "P7": 128,
    }

    for target in dataset_targets:

        image_targets = {}

        boxes = (
            target["boxes"]
            .to(device)
            .float()
        )

        for level, stride in strides.items():

            _, _, height, width = (
                predictions[level]
                ["classification"]
                .shape
            )

            image_targets[level] = (
                target_generator.generate_targets(
                    label_boxes=boxes,
                    feature_shape=(
                        height,
                        width,
                    ),
                    stride=stride,
                    device=device,
                )
            )

        batch_targets.append(
            image_targets
        )

    return batch_targets


# =========================================================
# EXTRACT FPN FEATURES
# =========================================================

@torch.no_grad()
def get_fpn_features(
    model,
    images,
):

    C2, C3, C4, C5 = (
        model.fpn.backbone(images)
    )

    P5 = model.fpn.lat_c5(C5)
    P5 = model.fpn.conv_p5(P5)

    P4 = model.fpn.lat_c4(C4)

    P4 = (
        P4
        + model.fpn.upsampling(P5)
    )

    P4 = model.fpn.conv_p4(P4)

    P3 = model.fpn.lat_c3(C3)

    P3 = (
        P3
        + model.fpn.upsampling(P4)
    )

    P3 = model.fpn.conv_p3(P3)

    P6 = model.fpn.conv_p6(
        P5
    )

    P7 = model.fpn.conv_p7(
        torch.relu(P6)
    )

    return {
        "C2": C2,
        "C3": C3,
        "C4": C4,
        "C5": C5,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "P6": P6,
        "P7": P7,
    }


# =========================================================
# PRINT FPN SUMMARY
# =========================================================

def print_feature_summary(
    title,
    features,
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for name in (
        "C2",
        "C3",
        "C4",
        "C5",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
    ):

        print_tensor_stats(
            name,
            features[name],
        )


# =========================================================
# DETECTION HEAD PARAMETERS
# =========================================================

def get_head_parameters(
    head,
):

    return {
        # Classification tower
        "classification_feature_conv.weight":
            head.classification_feature_conv.weight,

        "classification_feature_conv.bias":
            head.classification_feature_conv.bias,

        "classification_feature_norm.weight":
            head.classification_feature_norm.weight,

        "classification_feature_norm.bias":
            head.classification_feature_norm.bias,

        "classification_conv.weight":
            head.classification_conv.weight,

        "classification_conv.bias":
            head.classification_conv.bias,

        # Regression / centerness tower
        "regression_feature_conv.weight":
            head.regression_feature_conv.weight,

        "regression_feature_conv.bias":
            head.regression_feature_conv.bias,

        "regression_feature_norm.weight":
            head.regression_feature_norm.weight,

        "regression_feature_norm.bias":
            head.regression_feature_norm.bias,

        "regression_conv.weight":
            head.regression_conv.weight,

        "regression_conv.bias":
            head.regression_conv.bias,

        "centerness_conv.weight":
            head.centerness_conv.weight,

        "centerness_conv.bias":
            head.centerness_conv.bias,
    }


# =========================================================
# MAIN TEST
# =========================================================

def test_single_training_step():

    print()
    print("=" * 70)
    print("Single training-step diagnostics")
    print("=" * 70)

    print(
        f"[TEST] Device: {DEVICE}"
    )

    print(
        f"[TEST] Gradient clipping max norm: "
        f"{GRADIENT_CLIP_MAX_NORM}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_train_transforms(
            IMAGE_SIZE
        ),
    )

    print(
        f"[TEST] Dataset size: "
        f"{len(dataset)}"
    )

    # ---------------------------------------------------------
    # Select one image
    # ---------------------------------------------------------

    image, target = dataset[
        DATASET_INDEX
    ]

    images = (
        image
        .unsqueeze(0)
        .to(DEVICE)
    )

    dataset_targets = [
        {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }
    ]

    print(
        f"[TEST] Dataset index: "
        f"{DATASET_INDEX}"
    )

    print(
        f"[TEST] GT boxes: "
        f"{target['boxes']}"
    )

    # =========================================================
    # MODEL
    # =========================================================

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(DEVICE)

    print(
        f"[TEST] Backbone checkpoint: "
        f"{RESNET50_CHEST_XRAY_CHECKPOINT}"
    )

    model.train()

    # =========================================================
    # OPTIMIZER
    # =========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = DetectionLoss()

    target_generator = TargetGenerator()

    # =========================================================
    # STATE 0
    # =========================================================

    print()
    print("=" * 70)
    print("STATE 0: BEFORE TRAINING STEP")
    print("=" * 70)

    # ---------------------------------------------------------
    # FPN features before update
    # ---------------------------------------------------------

    initial_features = (
        get_fpn_features(
            model,
            images,
        )
    )

    print_feature_summary(
        "Features BEFORE update",
        initial_features,
    )

    # =========================================================
    # SAVE INITIAL PARAMETERS
    # =========================================================

    initial_parameters = {
        "fpn.conv_p3.weight":
            model.fpn.conv_p3.weight.detach().clone(),

        "fpn.conv_p4.weight":
            model.fpn.conv_p4.weight.detach().clone(),

        "fpn.conv_p5.weight":
            model.fpn.conv_p5.weight.detach().clone(),

        "fpn.conv_p6.weight":
            model.fpn.conv_p6.weight.detach().clone(),

        "fpn.conv_p7.weight":
            model.fpn.conv_p7.weight.detach().clone(),
    }

    classification_head_parameters = (
        get_head_parameters(
            model.head3
        )
    )

    for name, parameter in (
        classification_head_parameters.items()
    ):

        initial_parameters[
            f"head3.{name}"
        ] = (
            parameter.detach().clone()
        )

    # =========================================================
    # TRAINING STEP
    # =========================================================

    print()
    print("=" * 70)
    print("ONE TRAINING STEP")
    print("=" * 70)

    optimizer.zero_grad(
        set_to_none=True
    )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    predictions = model(
        images
    )

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------

    batch_targets = build_targets(
        target_generator=target_generator,
        predictions=predictions,
        dataset_targets=dataset_targets,
        device=DEVICE,
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    losses = criterion(
        predictions,
        batch_targets,
    )

    loss = losses["total"]

    print(
        f"[STEP] Total loss: "
        f"{loss.item():.8f}"
    )

    print(
        f"[STEP] Center loss: "
        f"{losses['center'].item():.8f}"
    )

    print(
        f"[STEP] Regression loss: "
        f"{losses['regression'].item():.8f}"
    )

    print(
        f"[STEP] Centerness loss: "
        f"{losses['centerness'].item():.8f}"
    )

    # =========================================================
    # BACKWARD
    # =========================================================

    loss.backward()

    # =========================================================
    # GRADIENTS BEFORE CLIPPING
    # =========================================================

    print()
    print(
        "[STEP] Selected gradient norms "
        "BEFORE clipping"
    )

    gradient_parameters = {

        # FPN
        "fpn.conv_p3.weight":
            model.fpn.conv_p3.weight,

        "fpn.conv_p4.weight":
            model.fpn.conv_p4.weight,

        "fpn.conv_p5.weight":
            model.fpn.conv_p5.weight,

        "fpn.conv_p6.weight":
            model.fpn.conv_p6.weight,

        "fpn.conv_p7.weight":
            model.fpn.conv_p7.weight,

        # Classification tower
        "head3.classification_feature_conv.weight":
            model.head3
            .classification_feature_conv
            .weight,

        "head3.classification_conv.weight":
            model.head3
            .classification_conv
            .weight,

        # Regression tower
        "head3.regression_feature_conv.weight":
            model.head3
            .regression_feature_conv
            .weight,

        "head3.regression_conv.weight":
            model.head3
            .regression_conv
            .weight,

        # Centerness
        "head3.centerness_conv.weight":
            model.head3
            .centerness_conv
            .weight,
    }

    for name, parameter in (
        gradient_parameters.items()
    ):

        if parameter.grad is None:

            print(
                f"{name}: gradient=None"
            )

        else:

            gradient = (
                parameter.grad.detach()
            )

            print(
                f"{name}: "
                f"norm="
                f"{gradient.norm().item():.8e} "
                f"abs_max="
                f"{gradient.abs().max().item():.8e}"
            )

    # =========================================================
    # GRADIENT CLIPPING
    # =========================================================

    total_gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_MAX_NORM,
        )
    )

    print()
    print(
        "[STEP] Global gradient norm "
        "before clipping: "
        f"{total_gradient_norm.item():.8e}"
    )

    # =========================================================
    # GRADIENTS AFTER CLIPPING
    # =========================================================

    print()
    print(
        "[STEP] Selected gradient norms "
        "AFTER clipping"
    )

    for name, parameter in (
        gradient_parameters.items()
    ):

        if parameter.grad is None:

            print(
                f"{name}: gradient=None"
            )

        else:

            gradient = (
                parameter.grad.detach()
            )

            print(
                f"{name}: "
                f"norm="
                f"{gradient.norm().item():.8e} "
                f"abs_max="
                f"{gradient.abs().max().item():.8e}"
            )

    # =========================================================
    # OPTIMIZER STEP
    # =========================================================

    optimizer.step()

    print(
        "[STEP] optimizer.step() completed."
    )

    # =========================================================
    # STATE 1
    # =========================================================

    print()
    print("=" * 70)
    print("STATE 1: AFTER ONE TRAINING STEP")
    print("=" * 70)

    updated_features = (
        get_fpn_features(
            model,
            images,
        )
    )

    print_feature_summary(
        "Features AFTER one update",
        updated_features,
    )

    # =========================================================
    # FEATURE CHANGE
    # =========================================================

    print()
    print("=" * 70)
    print("FEATURE CHANGE")
    print("=" * 70)

    for level in (
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
    ):

        before = (
            initial_features[level]
        )

        after = (
            updated_features[level]
        )

        difference = (
            after - before
        )

        relative_change = (
            difference.norm()
            /
            before.norm().clamp_min(
                1e-12
            )
        )

        print()
        print(
            f"{level}"
        )

        print(
            f"  before abs max: "
            f"{before.abs().max().item():.8e}"
        )

        print(
            f"  after abs max:  "
            f"{after.abs().max().item():.8e}"
        )

        print(
            f"  difference norm: "
            f"{difference.norm().item():.8e}"
        )

        print(
            f"  relative change:"
            f" {relative_change.item():.8e}"
        )

    # =========================================================
    # PARAMETER CHANGE
    # =========================================================

    print()
    print("=" * 70)
    print("PARAMETER CHANGE")
    print("=" * 70)

    current_parameters = {
        "fpn.conv_p3.weight":
            model.fpn.conv_p3.weight,

        "fpn.conv_p4.weight":
            model.fpn.conv_p4.weight,

        "fpn.conv_p5.weight":
            model.fpn.conv_p5.weight,

        "fpn.conv_p6.weight":
            model.fpn.conv_p6.weight,

        "fpn.conv_p7.weight":
            model.fpn.conv_p7.weight,
    }

    for name, parameter in (
        get_head_parameters(
            model.head3
        ).items()
    ):

        current_parameters[
            f"head3.{name}"
        ] = parameter

    for name, current in (
        current_parameters.items()
    ):

        before = (
            initial_parameters[name]
        )

        current_value = (
            current.detach()
        )

        delta = (
            current_value - before
        )

        print()
        print(
            name
        )

        print(
            f"  before norm: "
            f"{before.norm().item():.8e}"
        )

        print(
            f"  after norm:  "
            f"{current_value.norm().item():.8e}"
        )

        print(
            f"  delta norm:  "
            f"{delta.norm().item():.8e}"
        )

        print(
            f"  delta abs max:"
            f" {delta.abs().max().item():.8e}"
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(
        "Architecture:"
    )

    print(
        "  Classification tower: "
        "Conv3x3 -> GroupNorm -> ReLU -> Conv1x1"
    )

    print(
        "  Regression tower: "
        "Conv3x3 -> GroupNorm -> ReLU -> Conv1x1"
    )

    print(
        "  Centerness: "
        "same regression tower -> Conv1x1"
    )

    print(
        f"  Gradient clipping: "
        f"max_norm={GRADIENT_CLIP_MAX_NORM}"
    )

    print()
    print(
        "No checkpoint was saved."
    )

    print(
        "The test model was updated only in memory."
    )

    print("=" * 70)


if __name__ == "__main__":
    test_single_training_step()