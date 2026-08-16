import torch
import matplotlib.pyplot as plt

from src.models.backbone import Backbone
from src.models.fpn import FPN


class FeatureVisualizer:

    def __init__(
        self,
        backbone,
        fpn,
        max_channels=10,
    ):
        self.backbone = backbone
        self.fpn = fpn
        self.max_channels = max_channels

    # =====================================================
    # Feature extraction
    # =====================================================

    @torch.no_grad()
    def extract_features(self, x):

        self.backbone.eval()
        self.fpn.eval()

        # Backbone features
        C2, C3, C4, C5 = self.backbone(x)

        # FPN features
        P2, P3, P4, P5 = self.fpn(x)

        backbone_features = {
            "C2": C2,
            "C3": C3,
            "C4": C4,
            "C5": C5,
        }

        fpn_features = {
            "P2": P2,
            "P3": P3,
            "P4": P4,
            "P5": P5,
        }

        return backbone_features, fpn_features

    # =====================================================
    # Single feature map
    # =====================================================

    def plot_feature_map(
        self,
        feature,
        level_name,
        sample_index=0,
        channel_index=0,
    ):

        feature_map = feature[
            sample_index,
            channel_index,
        ]

        feature_map = feature_map.detach().cpu()

        plt.figure(figsize=(6, 6))

        plt.imshow(
            feature_map,
            cmap="gray",
        )

        plt.colorbar()

        plt.title(
            f"{level_name} - Channel {channel_index}"
        )

        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # =====================================================
    # Multiple channels from one level
    # =====================================================

    def plot_channels(
        self,
        feature,
        level_name, 
        sample_index=0,
        num_channels=None,
    ):

        if num_channels is None:
            num_channels = self.max_channels

        num_channels = min(
            num_channels,
            feature.shape[1],
        )

        columns = 5
        rows = (
            num_channels + columns - 1
        ) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(15, 3 * rows),
        )

        # Quando c'è una sola riga
        axes = axes.flatten()

        for channel_index in range(
            num_channels
        ):

            feature_map = feature[
                sample_index,
                channel_index,
            ]

            feature_map = (
                feature_map
                .detach()
                .cpu()
            )

            axes[channel_index].imshow(
                feature_map,
                cmap="gray",
            )

            axes[channel_index].set_title(
                f"{level_name} - "
                f"Channel {channel_index}"
            )

            axes[channel_index].axis("off")

        # Nasconde gli assi inutilizzati
        for index in range(
            num_channels,
            len(axes),
        ):
            axes[index].axis("off")

        plt.tight_layout()
        plt.show()

    # =====================================================
    # Backbone pyramid
    # =====================================================

    def plot_backbone_pyramid(
        self,
        features,
        sample_index=0,
        channel_index=0,
    ):

        self._plot_levels(
            features=features,
            sample_index=sample_index,
            channel_index=channel_index,
            title="Backbone Feature Pyramid",
        )

    # =====================================================
    # FPN pyramid
    # =====================================================

    def plot_fpn_pyramid(
        self,
        features,
        sample_index=0,
        channel_index=0,
    ):

        self._plot_levels(
            features=features,
            sample_index=sample_index,
            channel_index=channel_index,
            title="FPN Feature Pyramid",
        )

    # =====================================================
    # Generic pyramid plotting
    # =====================================================

    def _plot_levels(
        self,
        features,
        sample_index,
        channel_index,
        title,
    ):

        fig, axes = plt.subplots(
            1,
            len(features),
            figsize=(16, 4),
        )

        for ax, (level_name, feature) in zip(
            axes,
            features.items(),
        ):

            feature_map = feature[
                sample_index,
                channel_index,
            ]

            feature_map = (
                feature_map
                .detach()
                .cpu()
            )

            ax.imshow(
                feature_map,
                cmap="gray",
            )

            ax.set_title(
                f"{level_name}\n"
                f"Channel {channel_index}"
            )

            ax.axis("off")

        fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    # =====================================================
    # Complete visualization
    # =====================================================

    def visualize(
        self,
        x,
        sample_index=0,
        channel_index=0,
    ):

        backbone_features, fpn_features = (
            self.extract_features(x)
        )

        self.plot_backbone_pyramid(
            backbone_features,
            sample_index=sample_index,
            channel_index=channel_index,
        )

        self.plot_fpn_pyramid(
            fpn_features,
            sample_index=sample_index,
            channel_index=channel_index,
        )

        return (
            backbone_features,
            fpn_features,
        )
    



    def plot_input_image(
        self,
        image,
        sample_index=0,
    ):
        image = image[  
            sample_index
        ].detach().cpu()

        image = image.permute(
            1, 2, 0
        )

        plt.figure(figsize=(6, 6))

        plt.imshow(image)

        plt.title("Input Image")

        plt.axis("off")

        plt.tight_layout()
        plt.show()