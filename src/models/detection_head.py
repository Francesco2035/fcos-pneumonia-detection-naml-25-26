import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        center_channels: int,
        scale_channels: int,
    ):
        super().__init__()

        # Shared 3x3 convolution
        self.feature_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Center branch: one prediction per location
        self.center_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=center_channels,
            kernel_size=1,
        )

        # Scale branch: four regression values per location
        self.scale_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=scale_channels,
            kernel_size=1,
        )


    def forward(self, x):
        features = self.feature_conv(x)

        center = self.center_conv(features)
        scale = self.scale_conv(features)

        return center, scale