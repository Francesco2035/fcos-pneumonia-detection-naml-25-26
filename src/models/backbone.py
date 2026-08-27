import torch
import torch.nn as nn
import torchvision

from src.models.resnet import (
    ResNet50,
    ResNet101,
)


class Backbone(nn.Module):
    """
    ResNet-50 / ResNet-101 backbone used as feature extractor
    for the FPN.

    The backbone exposes:

        C2
        C3
        C4
        C5

    with channel dimensions:

        C2 = 256
        C3 = 512
        C4 = 1024
        C5 = 2048

    This keeps the interface compatible with the existing FPN.
    """

    def __init__(
        self,
        path_model=None,
        resnet_depth=50,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Validate depth
        # ---------------------------------------------------------

        if resnet_depth not in (50, 101):
            raise ValueError(
                "Unsupported ResNet depth: "
                f"{resnet_depth}. "
                "Supported values: 50, 101."
            )

        self.resnet_depth = resnet_depth

        # ---------------------------------------------------------
        # ImageNet pretrained backbone
        # ---------------------------------------------------------

        if path_model is None:

            print(
                f"[BACKBONE] Initializing "
                f"ImageNet ResNet-{resnet_depth}"
            )

            if resnet_depth == 50:

                self.model = (
                    torchvision.models.resnet50(
                        weights=(
                            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                        )
                    )
                )

            else:

                self.model = (
                    torchvision.models.resnet101(
                        weights=(
                            torchvision.models.ResNet101_Weights.IMAGENET1K_V2
                        )
                    )
                )

        # ---------------------------------------------------------
        # Custom Chest-Xray pretrained backbone
        # ---------------------------------------------------------

        else:

            print(
                f"[BACKBONE] Initializing "
                f"custom Chest-Xray ResNet-{resnet_depth}"
            )

            print(
                "[BACKBONE] Checkpoint:"
            )

            print(
                f"            {path_model}"
            )

            if resnet_depth == 50:

                self.model = ResNet50(
                    3,
                    2,
                )

            else:

                self.model = ResNet101(
                    3,
                    2,
                )

            # -----------------------------------------------------
            # Load checkpoint
            # -----------------------------------------------------

            checkpoint = torch.load(
                path_model,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(
                checkpoint,
                dict,
            ):
                raise TypeError(
                    "Backbone checkpoint must be a dictionary."
                )

            if (
                "model_state_dict"
                not in checkpoint
            ):
                raise KeyError(
                    "Backbone checkpoint does not contain "
                    "'model_state_dict'."
                )

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

            # -----------------------------------------------------
            # Strict loading
            # -----------------------------------------------------

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

            del checkpoint

            print(
                "[BACKBONE] Checkpoint loaded successfully."
            )

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, x):

        # ---------------------------------------------------------
        # Stem
        # ---------------------------------------------------------

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # ---------------------------------------------------------
        # ResNet stages
        # ---------------------------------------------------------

        C2 = self.model.layer1(x)

        C3 = self.model.layer2(C2)

        C4 = self.model.layer3(C3)

        C5 = self.model.layer4(C4)

        # ---------------------------------------------------------
        # Return FPN features
        # ---------------------------------------------------------

        return (
            C2,
            C3,
            C4,
            C5,
        )