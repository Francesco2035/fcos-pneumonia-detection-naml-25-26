
import torch
import torch.nn as nn

from src.models.backbone import Backbone


class FPN(nn.Module):
        
    """
        Feature Pyramid Network used to build the five feature levels required
        by the FCOS detector.

        The ResNet-50 backbone provides features up to stride 32 (C5).
        P6 and P7 are therefore added here, in the FPN, using stride-2
        convolutions to extend the pyramid to strides 64 and 128 without
        modifying the ResNet backbone itself.
    """

    def __init__(self, path_model):
        super().__init__()

        self.backbone = Backbone(
            path_model
        )

        self.upsampling = nn.UpsamplingNearest2d(
            scale_factor=2
        )

        self.lat_c3 = nn.Conv2d(512, 256, 1)
        self.lat_c4 = nn.Conv2d(1024, 256, 1)
        self.lat_c5 = nn.Conv2d(2048, 256, 1)

        self.conv_p3 = nn.Conv2d(
            256, 256, 3, padding=1
        )

        self.conv_p4 = nn.Conv2d(
            256, 256, 3, padding=1
        )

        self.conv_p5 = nn.Conv2d(
            256, 256, 3, padding=1
        )

        self.conv_p6 = nn.Conv2d(
            256, 256,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.conv_p7 = nn.Conv2d(
            256, 256,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):

        C2, C3, C4, C5 = self.backbone(x)

        P5 = self.lat_c5(C5)
        P5 = self.conv_p5(P5)

        P4 = self.lat_c4(C4)
        P4 = P4 + self.upsampling(P5)
        P4 = self.conv_p4(P4)

        P3 = self.lat_c3(C3)
        P3 = P3 + self.upsampling(P4)
        P3 = self.conv_p3(P3)

        P6 = self.conv_p6(P5)

        P7 = self.conv_p7(
            torch.relu(P6)
        )

        return P3, P4, P5, P6, P7