import torch

from src.models.fpn import FPN


def main():

    path  = '/home/legion/shared/Projects/NAML_25-26/checkpoints/resnet50_scratch_chest_xray_best.pth'
    rel_path  ='checkpoints/resnet50_scratch_chest_xray_best.pth'

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------
    # Dummy input
    # ---------------------------------

    x = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    # ---------------------------------
    # FPN con backbone ImageNet
    # ---------------------------------

    model = FPN(
        path_model= rel_path,
        device=device
    ).to(device)

    model.eval()

    # ---------------------------------
    # Forward
    # ---------------------------------

    with torch.no_grad():

        P2, P3, P4, P5 = model(x)

    # ---------------------------------
    # Check shapes
    # ---------------------------------

    print("\nFPN output shapes:")

    print("P2:", P2.shape)
    print("P3:", P3.shape)
    print("P4:", P4.shape)
    print("P5:", P5.shape)


if __name__ == "__main__":
    main()