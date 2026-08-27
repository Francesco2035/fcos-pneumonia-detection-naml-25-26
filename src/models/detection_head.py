import torch.nn as nn

class DetectionHead(nn.Module):
    """
    Detection head applied to a single FPN level.

    It transforms the input feature map into three predictions:
    object classification, bounding-box regression, and centerness.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        classification_channels: int,
        regression_channels: int,
        centerness_channels: int,
    ):
        super().__init__()

        # =====================================================
        # Classification tower
        # =====================================================

        self.classification_feature_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.classification_feature_norm = nn.GroupNorm(
            num_groups=32,
            num_channels=hidden_channels,
        )

        self.classification_feature_activation = nn.ReLU(
            inplace=True
        )

        # =====================================================
        # Regression / centerness tower
        # =====================================================

        self.regression_feature_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.regression_feature_norm = nn.GroupNorm(
            num_groups=32,
            num_channels=hidden_channels,
        )

        self.regression_feature_activation = nn.ReLU(
            inplace=True
        )

        # =====================================================
        # Final prediction layers
        # =====================================================

        self.classification_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=classification_channels,
            kernel_size=1,
        )

        self.regression_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=regression_channels,
            kernel_size=1,
        )

        self.centerness_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=centerness_channels,
            kernel_size=1,
        )

    def forward(self, x):

        # =====================================================
        # Classification branch
        # =====================================================

        classification_features = (
            self.classification_feature_conv(x)
        )

        classification_features = (
            self.classification_feature_norm(
                classification_features
            )
        )

        classification_features = (
            self.classification_feature_activation(
                classification_features
            )
        )

        classification = (
            self.classification_conv(
                classification_features
            )
        )

        # =====================================================
        # Regression / centerness branch
        # =====================================================

        regression_features = (
            self.regression_feature_conv(x)
        )

        regression_features = (
            self.regression_feature_norm(
                regression_features
            )
        )

        regression_features = (
            self.regression_feature_activation(
                regression_features
            )
        )

        regression = (
            self.regression_conv(
                regression_features
            )
        )

        centerness = (
            self.centerness_conv(
                regression_features
            )
        )

        # =====================================================
        # Return predictions
        # =====================================================

        return (
            classification,
            regression,
            centerness,
        )