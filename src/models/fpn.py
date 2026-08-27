import torch
import torch.nn as nn

from src.models.backbone import Backbone


class FPN(nn.Module):
    """
    Feature Pyramid Network used to build the five feature levels
    required by the FCOS detector.

    The backbone can be either ResNet-50 or ResNet-101.

    Both architectures expose the same feature dimensions:

        C3 -> 512 channels
        C4 -> 1024 channels
        C5 -> 2048 channels

    Therefore the FPN itself is independent of the ResNet depth.

    P6 and P7 are added here using stride-2 convolutions.
    """

    def __init__(
        self,
        path_model=None,
        resnet_depth=50,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Backbone
        # ---------------------------------------------------------

        self.backbone = Backbone(
            path_model=path_model,
            resnet_depth=resnet_depth,
        )

        self.resnet_depth = resnet_depth

        # ---------------------------------------------------------
        # Top-down pathway
        # ---------------------------------------------------------

        self.upsampling = nn.UpsamplingNearest2d(
            scale_factor=2
        )

        # ---------------------------------------------------------
        # Lateral connections
        #
        # These channel dimensions are identical for ResNet-50
        # and ResNet-101.
        # ---------------------------------------------------------

        self.lat_c3 = nn.Conv2d(
            512,
            256,
            kernel_size=1,
        )

        self.lat_c4 = nn.Conv2d(
            1024,
            256,
            kernel_size=1,
        )

        self.lat_c5 = nn.Conv2d(
            2048,
            256,
            kernel_size=1,
        )

        # ---------------------------------------------------------
        # Feature refinement
        # ---------------------------------------------------------

        self.conv_p3 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        self.conv_p4 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        self.conv_p5 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # Additional pyramid levels
        # ---------------------------------------------------------

        self.conv_p6 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv_p7 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, x):

        # ---------------------------------------------------------
        # Backbone features
        # ---------------------------------------------------------

        C2, C3, C4, C5 = self.backbone(x)

        # ---------------------------------------------------------
        # P5
        # ---------------------------------------------------------

        P5 = self.lat_c5(C5)
        P5 = self.conv_p5(P5)

        # ---------------------------------------------------------
        # P4
        # ---------------------------------------------------------

        P4 = self.lat_c4(C4)

        P4 = (
            P4
            + self.upsampling(P5)
        )

        P4 = self.conv_p4(P4)

        # ---------------------------------------------------------
        # P3
        # ---------------------------------------------------------

        P3 = self.lat_c3(C3)

        P3 = (
            P3
            + self.upsampling(P4)
        )

        P3 = self.conv_p3(P3)

        # ---------------------------------------------------------
        # P6
        # ---------------------------------------------------------

        P6 = self.conv_p6(P5)

        # ---------------------------------------------------------
        # P7
        # ---------------------------------------------------------

        P7 = self.conv_p7(
            torch.relu(P6)
        )

        return (
            P3,
            P4,
            P5,
            P6,
            P7,
        )