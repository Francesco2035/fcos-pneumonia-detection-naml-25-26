import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.backbone import Backbone
from src.models.fpn import FPN
from src.visualization.visualizer import FeatureVisualizer


# =========================================================
# Configuration
# =========================================================

IMAGE_PATH = (
    "data/chest_xray/test/PNEUMONIA/person12_bacteria_47.jpeg"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BACKBONE_TYPE = None

NUM_CHANNEL = 0


# =========================================================
# Image loading
# =========================================================

def load_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    # [C, H, W] -> [1, C, H, W]
    image = image.unsqueeze(0)

    return image


# =========================================================
# Main
# =========================================================

def main():

    print(f"Device: {DEVICE}")

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = load_image(IMAGE_PATH)

    image = image.to(DEVICE)

    print(
        f"Input shape: {image.shape}"
    )

    # -----------------------------------------------------
    # Backbone
    # -----------------------------------------------------

    backbone = Backbone(
        path_model=BACKBONE_TYPE,
        device=DEVICE,
    ).to(DEVICE)

    # -----------------------------------------------------
    # FPN
    # -----------------------------------------------------

    fpn = FPN(
        path_model=BACKBONE_TYPE,
        device=DEVICE,
    ).to(DEVICE)

    # -----------------------------------------------------
    # Visualizer
    # -----------------------------------------------------

    visualizer = FeatureVisualizer(
        backbone=backbone,
        fpn=fpn,
        max_channels=10,
    )
    
    # -----------------------------------------------------
    # Original image
    # -----------------------------------------------------
    visualizer.plot_input_image(
        image,
    )


    # -----------------------------------------------------
    # Extract and visualize
    # -----------------------------------------------------

    backbone_features, fpn_features = (
        visualizer.extract_features(image)
    )

    # -----------------------------------------------------
    # Print shapes
    # -----------------------------------------------------

    print("\nBackbone features:")

    for name, feature in backbone_features.items():

        print(
            f"{name}: {feature.shape}"
        )

    print("\nFPN features:")

    for name, feature in fpn_features.items():

        print(
            f"{name}: {feature.shape}"
        )

    # -----------------------------------------------------
    # Visualize Backbone
    # -----------------------------------------------------

    visualizer.plot_backbone_pyramid(
        backbone_features,
        sample_index=0,
        channel_index=NUM_CHANNEL,
    )

    # -----------------------------------------------------
    # Visualize FPN
    # -----------------------------------------------------

    visualizer.plot_fpn_pyramid(
        fpn_features,
        sample_index=0,
        channel_index=NUM_CHANNEL,
    )


if __name__ == "__main__":
    main()