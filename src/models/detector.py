import torch.nn as nn

from src.models.fpn import FPN
from src.models.detection_head import DetectionHead


class DetectionFramework(nn.Module):

    def __init__(self, path_model=None):
        super().__init__()

        self.fpn = FPN(
            path_model=path_model
        )

        self.head3 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            center_channels=1,
            scale_channels=4,
        )

        self.head4 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            center_channels=1,
            scale_channels=4,
        )

        self.head5 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            center_channels=1,
            scale_channels=4,
        )

        self.head6 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            center_channels=1,
            scale_channels=4,
        )

        self.head7 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            center_channels=1,
            scale_channels=4,
        )

    def forward(self, x):

        P3, P4, P5, P6, P7 = self.fpn(x)

        center3, scale3 = self.head3(P3)
        center4, scale4 = self.head4(P4)
        center5, scale5 = self.head5(P5)
        center6, scale6 = self.head6(P6)
        center7, scale7 = self.head7(P7)

        return {
            "P3": {
                "center": center3,
                "scale": scale3,
            },
            "P4": {
                "center": center4,
                "scale": scale4,
            },
            "P5": {
                "center": center5,
                "scale": scale5,
            },
            "P6": {
                "center": center6,
                "scale": scale6,
            },
            "P7": {
                "center": center7,
                "scale": scale7,
            },
        }