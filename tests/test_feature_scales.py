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


CHECKPOINT_PATH = (
    "checkpoints/exp1/best.pt"
)

NUM_IMAGES = 10

LEVELS = (
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)


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


def tensor_stats(tensor):
    """
    Return basic statistics for a tensor.
    """

    tensor = tensor.detach()

    return {
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "abs_max": tensor.abs().max().item(),
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
        f"  mean:    {stats['mean']:.6f}"
    )

    print(
        f"  std:     {stats['std']:.6f}"
    )

    print(
        f"  min:     {stats['min']:.6f}"
    )

    print(
        f"  max:     {stats['max']:.6f}"
    )

    print(
        f"  abs max: {stats['abs_max']:.6f}"
    )


def test_feature_scales():

    print()
    print("=" * 70)
    print("Feature scale diagnostics")
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
            positive_indices.append(index)

        if len(positive_indices) >= NUM_IMAGES:
            break

    if len(positive_indices) == 0:
        raise RuntimeError(
            "No positive images found."
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

    model.eval()

    print(
        f"[TEST] Loaded checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # =========================================================
    # Run test
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

            print()
            print("=" * 70)

            print(
                f"IMAGE {image_number}/"
                f"{len(positive_indices)}"
            )

            print(
                f"dataset_index="
                f"{dataset_index}"
            )

            print("=" * 70)

            # -------------------------------------------------
            # Backbone
            # -------------------------------------------------

            C2, C3, C4, C5 = (
                model.fpn.backbone(image)
            )

            print()
            print(
                "--- Backbone ---"
            )

            print_stats(
                "C2",
                C2,
            )

            print_stats(
                "C3",
                C3,
            )

            print_stats(
                "C4",
                C4,
            )

            print_stats(
                "C5",
                C5,
            )

            # -------------------------------------------------
            # FPN
            # -------------------------------------------------

            P3 = model.fpn.lat_c3(C3)

            P5 = model.fpn.lat_c5(C5)
            P5 = model.fpn.conv_p5(P5)

            P4 = model.fpn.lat_c4(C4)
            P4 = (
                P4
                + model.fpn.upsampling(P5)
            )
            P4 = model.fpn.conv_p4(P4)

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

            print()
            print(
                "--- FPN ---"
            )

            print_stats(
                "P3",
                P3,
            )

            print_stats(
                "P4",
                P4,
            )

            print_stats(
                "P5",
                P5,
            )

            print_stats(
                "P6",
                P6,
            )

            print_stats(
                "P7",
                P7,
            )

            # -------------------------------------------------
            # Detection heads
            # -------------------------------------------------

            fpn_features = {
                "P3": P3,
                "P4": P4,
                "P5": P5,
                "P6": P6,
                "P7": P7,
            }

            heads = {
                "P3": model.head3,
                "P4": model.head4,
                "P5": model.head5,
                "P6": model.head6,
                "P7": model.head7,
            }

            print()
            print(
                "--- Detection Heads ---"
            )

            for level in LEVELS:

                features = fpn_features[
                    level
                ]

                head = heads[
                    level
                ]

                # Shared feature convolution
                shared = (
                    head.feature_conv(
                        features
                    )
                )

                classification = (
                    head.classification_conv(
                        shared
                    )
                )

                regression = (
                    head.regression_conv(
                        shared
                    )
                )

                centerness = (
                    head.centerness_conv(
                        shared
                    )
                )

                print()
                print(
                    f"{level}"
                )

                print_stats(
                    "  FPN feature",
                    features,
                )

                print_stats(
                    "  Shared head feature",
                    shared,
                )

                print_stats(
                    "  Classification logits",
                    classification,
                )

                print_stats(
                    "  Regression output",
                    regression,
                )

                print_stats(
                    "  Centerness logits",
                    centerness,
                )


if __name__ == "__main__":
    test_feature_scales()