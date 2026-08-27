import torch.nn as nn

from src.models.fpn import FPN
from src.models.detection_head import DetectionHead


class DetectionFramework(nn.Module):
    """
    Full anchor-free detection model built on top of the FPN.

    A separate DetectionHead is applied to each FPN level (P3-P7),
    producing classification, regression, and centerness predictions.

    The backbone can be ResNet-50 or ResNet-101.
    """

    def __init__(
        self,
        path_model=None,
        resnet_depth=50,
    ):
        super().__init__()

        # -----------------------------------------------------
        # Feature Pyramid Network
        # -----------------------------------------------------

        self.fpn = FPN(
            path_model=path_model,
            resnet_depth=resnet_depth,
        )

        self.resnet_depth = resnet_depth

        # -----------------------------------------------------
        # Detection heads
        #
        # Each head receives 256-channel FPN features and
        # produces:
        #
        #   classification -> 1 value per location
        #   regression     -> 4 LTRB values per location
        #   centerness     -> 1 value per location
        # -----------------------------------------------------

        self.head3 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            classification_channels=1,
            regression_channels=4,
            centerness_channels=1,
        )

        self.head4 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            classification_channels=1,
            regression_channels=4,
            centerness_channels=1,
        )

        self.head5 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            classification_channels=1,
            regression_channels=4,
            centerness_channels=1,
        )

        self.head6 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            classification_channels=1,
            regression_channels=4,
            centerness_channels=1,
        )

        self.head7 = DetectionHead(
            in_channels=256,
            hidden_channels=256,
            classification_channels=1,
            regression_channels=4,
            centerness_channels=1,
        )

    def forward(self, x):

        # -----------------------------------------------------
        # Feature pyramid
        # -----------------------------------------------------

        P3, P4, P5, P6, P7 = self.fpn(x)

        # -----------------------------------------------------
        # Detection heads
        # -----------------------------------------------------

        classification3, regression3, centerness3 = (
            self.head3(P3)
        )

        classification4, regression4, centerness4 = (
            self.head4(P4)
        )

        classification5, regression5, centerness5 = (
            self.head5(P5)
        )

        classification6, regression6, centerness6 = (
            self.head6(P6)
        )

        classification7, regression7, centerness7 = (
            self.head7(P7)
        )

        # -----------------------------------------------------
        # Return predictions for every FPN level
        # -----------------------------------------------------

        return {
            "P3": {
                "classification": classification3,
                "regression": regression3,
                "centerness": centerness3,
            },

            "P4": {
                "classification": classification4,
                "regression": regression4,
                "centerness": centerness4,
            },

            "P5": {
                "classification": classification5,
                "regression": regression5,
                "centerness": centerness5,
            },

            "P6": {
                "classification": classification6,
                "regression": regression6,
                "centerness": centerness6,
            },

            "P7": {
                "classification": classification7,
                "regression": regression7,
                "centerness": centerness7,
            },
        }