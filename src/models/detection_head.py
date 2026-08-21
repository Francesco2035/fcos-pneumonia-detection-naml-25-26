import torch.nn as nn


class DetectionHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        classification_channels: int,
        regression_channels: int,
        centerness_channels: int,
    ):
        super().__init__()

        # -----------------------------------------------------
        # Shared feature convolution
        # -----------------------------------------------------

        self.feature_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # -----------------------------------------------------
        # Classification branch
        #
        # Binary classification:
        # 1 logit per location
        # -----------------------------------------------------

        self.classification_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=classification_channels,
            kernel_size=1,
        )

        # -----------------------------------------------------
        # Regression branch
        #
        # 4 values per location:
        # left, top, right, bottom
        # -----------------------------------------------------

        self.regression_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=regression_channels,
            kernel_size=1,
        )

        # -----------------------------------------------------
        # Center-ness branch
        #
        # 1 value per location
        # -----------------------------------------------------

        self.centerness_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=centerness_channels,
            kernel_size=1,
        )

    def forward(self, x):

        # Shared features
        features = self.feature_conv(x)

        # Three prediction branches
        classification = self.classification_conv(features)

        regression = self.regression_conv(features)

        centerness = self.centerness_conv(features)

        return classification, regression, centerness