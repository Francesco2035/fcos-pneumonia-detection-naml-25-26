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

from src.detection_loss import (
    DetectionLoss,
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
            "Could not find model weights in checkpoint."
        )


# =========================================================
# GRADIENT STATISTICS
# =========================================================

def gradient_norm(
    parameters,
):
    total_norm_squared = 0.0
    max_abs = 0.0
    nonzero_elements = 0
    total_elements = 0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        if grad.numel() == 0:
            continue

        total_norm_squared += (
            grad.norm(2).item() ** 2
        )

        max_abs = max(
            max_abs,
            grad.abs().max().item(),
        )

        nonzero_elements += (
            grad != 0
        ).sum().item()

        total_elements += grad.numel()

    norm = total_norm_squared ** 0.5

    nonzero_ratio = (
        nonzero_elements / total_elements
        if total_elements > 0
        else 0.0
    )

    return {
        "norm": norm,
        "max_abs": max_abs,
        "nonzero_ratio": nonzero_ratio,
    }


def named_gradient_norm(
    model,
    keyword=None,
):
    """
    Compute gradient statistics for model parameters.

    If keyword is provided, only parameter names containing
    that keyword are included.
    """

    parameters = []

    for name, parameter in model.named_parameters():

        if parameter.grad is None:
            continue

        if keyword is None:
            parameters.append(parameter)

        elif keyword.lower() in name.lower():
            parameters.append(parameter)

    return gradient_norm(parameters)


# =========================================================
# PARAMETER GROUPS
# =========================================================

def collect_parameter_groups(model):
    """
    Collect parameter groups according to the actual FCOS head
    structure.

    Classification:
        classification feature tower + classification output

    Shared regression tower:
        regression feature conv + norm + activation parameters

    Regression:
        regression output conv

    Centerness:
        centerness output conv

    FPN:
        FPN parameters

    Backbone:
        backbone parameters
    """

    groups = {
        "classification_head": [],
        "shared_regression_tower": [],
        "regression_head": [],
        "centerness_head": [],
        "fpn": [],
        "backbone": [],
    }

    for level in LEVELS:

        head = getattr(model, f"head{level[-1]}")

        # -----------------------------------------------------
        # Classification branch
        # -----------------------------------------------------

        groups["classification_head"] += list(
            head.classification_feature_conv.parameters()
        )

        groups["classification_head"] += list(
            head.classification_feature_norm.parameters()
        )

        groups["classification_head"] += list(
            head.classification_conv.parameters()
        )

        # -----------------------------------------------------
        # Shared regression / centerness tower
        # -----------------------------------------------------

        groups["shared_regression_tower"] += list(
            head.regression_feature_conv.parameters()
        )

        groups["shared_regression_tower"] += list(
            head.regression_feature_norm.parameters()
        )

        # -----------------------------------------------------
        # Regression output
        # -----------------------------------------------------

        groups["regression_head"] += list(
            head.regression_conv.parameters()
        )

        # -----------------------------------------------------
        # Centerness output
        # -----------------------------------------------------

        groups["centerness_head"] += list(
            head.centerness_conv.parameters()
        )

    # ---------------------------------------------------------
    # FPN / backbone
    # ---------------------------------------------------------

    for name, parameter in model.named_parameters():

        name_lower = name.lower()

        if "head" in name_lower:
            continue

        if "fpn" in name_lower:
            groups["fpn"].append(parameter)

        elif "backbone" in name_lower:
            groups["backbone"].append(parameter)

    return groups


# =========================================================
# PRINT GRADIENT REPORT
# =========================================================

def print_gradient_report(
    title,
    model,
    groups,
):
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)

    print(
        f"{'group':28s}"
        f"{'norm':>18s}"
        f"{'max_abs':>18s}"
        f"{'nonzero %':>15s}"
    )

    print("-" * 80)

    for group_name, parameters in groups.items():

        stats = gradient_norm(
            parameters
        )

        print(
            f"{group_name:28s}"
            f"{stats['norm']:18.8e}"
            f"{stats['max_abs']:18.8e}"
            f"{100.0 * stats['nonzero_ratio']:14.2f}%"
        )


# =========================================================
# MAIN TEST
# =========================================================

def test_gradient_decomposition():

    print()
    print("=" * 80)
    print("FCOS LOSS / GRADIENT DECOMPOSITION TEST")
    print("=" * 80)

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
        f"[TEST] Positive images: "
        f"{len(positive_indices)}"
    )

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

    model.train()

    print(
        f"[TEST] Loaded checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # ---------------------------------------------------------
    # Loss / target generator
    # ---------------------------------------------------------

    criterion = DetectionLoss()

    target_generator = TargetGenerator()

    # ---------------------------------------------------------
    # Parameter groups
    # ---------------------------------------------------------

    groups = collect_parameter_groups(
        model
    )

    print()
    print("=" * 80)
    print("PARAMETER GROUPS")
    print("=" * 80)

    for group_name, parameters in groups.items():

        num_parameters = sum(
            p.numel()
            for p in parameters
        )

        print(
            f"{group_name:28s}: "
            f"{len(parameters):4d} tensors, "
            f"{num_parameters:,} parameters"
        )

    # ---------------------------------------------------------
    # Accumulators
    # ---------------------------------------------------------

    loss_names = (
        "classification",
        "regression",
        "centerness",
        "total",
    )

    accumulated = {
        loss_name: {
            group_name: 0.0
            for group_name in groups
        }
        for loss_name in loss_names
    }

    accumulated_global = {
        loss_name: 0.0
        for loss_name in loss_names
    }

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    for image_number, dataset_index in enumerate(
        positive_indices,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"IMAGE {image_number}/"
            f"{len(positive_indices)}"
        )
        print("=" * 80)

        # -----------------------------------------------------
        # Load image
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Forward
        # -----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

        predictions = model(
            image
        )

        # -----------------------------------------------------
        # Build targets
        # -----------------------------------------------------

        batch_targets = []

        image_targets = {}

        for level in LEVELS:

            _, _, height, width = (
                predictions[level][
                    "classification"
                ].shape
            )

            target_level = (
                target_generator.generate_targets(
                    label_boxes=gt_boxes,
                    feature_shape=(
                        height,
                        width,
                    ),
                    stride=STRIDES[level],
                    device=device,
                )
            )

            image_targets[level] = (
                target_level
            )

        batch_targets.append(
            image_targets
        )

        # -----------------------------------------------------
        # Compute all losses
        # -----------------------------------------------------

        losses = criterion(
            predictions,
            batch_targets,
        )

        print(
            f"classification loss = "
            f"{losses['center'].item():.8e}"
        )

        print(
            f"regression loss     = "
            f"{losses['regression'].item():.8e}"
        )

        print(
            f"centerness loss     = "
            f"{losses['centerness'].item():.8e}"
        )

        print(
            f"total loss          = "
            f"{losses['total'].item():.8e}"
        )

        # =====================================================
        # BACKWARD FOR EACH LOSS SEPARATELY
        # =====================================================

        objectives = {
            "classification":
                losses["center"],

            "regression":
                losses["regression"],

            "centerness":
                losses["centerness"],

            "total":
                losses["total"],
        }

        for objective_name, objective_loss in objectives.items():

            # -------------------------------------------------
            # Clear gradients
            # -------------------------------------------------

            model.zero_grad(
                set_to_none=True
            )

            # -------------------------------------------------
            # Backward
            #
            # retain_graph=True because we reuse the same
            # forward graph for the other objectives.
            # -------------------------------------------------

            objective_loss.backward(
                retain_graph=True
            )

            # -------------------------------------------------
            # Global gradient norm
            # -------------------------------------------------

            global_stats = gradient_norm(
                [
                    p
                    for p in model.parameters()
                ]
            )

            accumulated_global[
                objective_name
            ] += global_stats["norm"]

            print()
            print(
                f"[{objective_name.upper()}]"
            )

            print(
                f"  GLOBAL gradient norm = "
                f"{global_stats['norm']:.8e}"
            )

            # -------------------------------------------------
            # Group statistics
            # -------------------------------------------------

            for group_name, parameters in groups.items():

                stats = gradient_norm(
                    parameters
                )

                accumulated[
                    objective_name
                ][group_name] += (
                    stats["norm"]
                )

                print(
                    f"  {group_name:28s} "
                    f"norm={stats['norm']:.8e}"
                )

        # -----------------------------------------------------
        # Explicitly clear graph references
        # -----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    num_samples = len(
        positive_indices
    )

    print()
    print("=" * 80)
    print("FINAL GRADIENT SUMMARY")
    print("=" * 80)

    # ---------------------------------------------------------
    # Global objectives
    # ---------------------------------------------------------

    print()
    print("GLOBAL GRADIENT NORMS")
    print("-" * 80)

    for objective_name in loss_names:

        mean_norm = (
            accumulated_global[
                objective_name
            ]
            / num_samples
        )

        print(
            f"{objective_name:20s}: "
            f"{mean_norm:.8e}"
        )

    # ---------------------------------------------------------
    # Per-group objectives
    # ---------------------------------------------------------

    for objective_name in loss_names:

        print()
        print(
            f"{objective_name.upper()} "
            f"GRADIENTS"
        )

        print("-" * 80)

        for group_name in groups:

            mean_norm = (
                accumulated[
                    objective_name
                ][group_name]
                / num_samples
            )

            print(
                f"{group_name:28s}: "
                f"{mean_norm:.8e}"
            )

    # =========================================================
    # IMPORTANT RATIOS
    # =========================================================

    print()
    print("=" * 80)
    print("IMPORTANT RATIOS")
    print("=" * 80)

    cls_reg = (
        accumulated_global[
            "regression"
        ]
        /
        max(
            accumulated_global[
                "classification"
            ],
            1e-12,
        )
    )

    ctr_reg = (
        accumulated_global[
            "regression"
        ]
        /
        max(
            accumulated_global[
                "centerness"
            ],
            1e-12,
        )
    )

    print(
        f"global regression / classification "
        f"gradient ratio = {cls_reg:.6f}"
    )

    print(
        f"global regression / centerness "
        f"gradient ratio = {ctr_reg:.6f}"
    )

    # ---------------------------------------------------------
    # Shared regression tower ratios
    # ---------------------------------------------------------

    shared_cls = (
        accumulated["classification"][
            "shared_regression_tower"
        ]
        / num_samples
    )

    shared_reg = (
        accumulated["regression"][
            "shared_regression_tower"
        ]
        / num_samples
    )

    shared_ctr = (
        accumulated["centerness"][
            "shared_regression_tower"
        ]
        / num_samples
    )

    print()
    print(
        "Shared regression tower:"
    )

    print(
        f"  classification = "
        f"{shared_cls:.8e}"
    )

    print(
        f"  regression     = "
        f"{shared_reg:.8e}"
    )

    print(
        f"  centerness     = "
        f"{shared_ctr:.8e}"
    )

    print()
    print("=" * 80)
    print(
        "NO optimizer.step() was performed."
    )
    print(
        "The checkpoint was not modified."
    )
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    test_gradient_decomposition()