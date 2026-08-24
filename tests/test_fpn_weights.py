import torch


from src.config import (
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.models.detector import (
    DetectionFramework,
)


CHECKPOINT_PATH = (
    "checkpoints/exp1/best.pt"
)


def tensor_stats(tensor):

    tensor = tensor.detach()

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "abs_max": tensor.abs().max().item(),
        "norm": tensor.norm().item(),
    }


def print_stats(
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

    print(
        f"  norm:    {stats['norm']:.8e}"
    )


def test_fpn_weights():

    print()
    print("=" * 70)
    print("FPN weight diagnostics")
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
    # Model
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(device)

    # ---------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    model.eval()

    print(
        f"[TEST] Loaded checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # =========================================================
    # FPN weights
    # =========================================================

    fpn = model.fpn

    print()
    print("=" * 70)
    print("FPN convolution weights")
    print("=" * 70)

    layers = {
        "lat_c3": fpn.lat_c3,
        "lat_c4": fpn.lat_c4,
        "lat_c5": fpn.lat_c5,
        "conv_p3": fpn.conv_p3,
        "conv_p4": fpn.conv_p4,
        "conv_p5": fpn.conv_p5,
        "conv_p6": fpn.conv_p6,
        "conv_p7": fpn.conv_p7,
    }

    for name, layer in layers.items():

        print()
        print(
            f"--- {name} ---"
        )

        print_stats(
            "weight",
            layer.weight,
        )

        print_stats(
            "bias",
            layer.bias,
        )

    # =========================================================
    # Detection head weights
    # =========================================================

    print()
    print("=" * 70)
    print("Detection head feature convolution weights")
    print("=" * 70)

    for level, head in (
        [
            ("P3", model.head3),
            ("P4", model.head4),
            ("P5", model.head5),
            ("P6", model.head6),
            ("P7", model.head7),
        ]
    ):

        print()
        print(
            f"--- {level} ---"
        )

        print_stats(
            "feature_conv.weight",
            head.feature_conv.weight,
        )

        print_stats(
            "feature_conv.bias",
            head.feature_conv.bias,
        )

    print()
    print("=" * 70)
    print(
        "FPN weight diagnostics completed."
    )
    print("=" * 70)


if __name__ == "__main__":
    test_fpn_weights()