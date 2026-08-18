import torch

from src.models.fpn import FPN


def main():

    checkpoint_path = (
        "checkpoints/resnet50_scratch_chest_xray_best.pth"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
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
    # FPN con backbone pre-addestrato
    # ---------------------------------

    model = FPN(
        path_model=checkpoint_path,
        device=device,
    ).to(device)

    model.eval()

    # ---------------------------------
    # Forward
    # ---------------------------------

    with torch.no_grad():

        P3, P4, P5, P6, P7 = model(x)

    # ---------------------------------
    # Check shapes
    # ---------------------------------

    print("\nFPN output shapes:")

    print("P3:", P3.shape)
    print("P4:", P4.shape)
    print("P5:", P5.shape)
    print("P6:", P6.shape)
    print("P7:", P7.shape)

    # ---------------------------------
    # Assertions
    # ---------------------------------

    assert P3.shape == (2, 256, 28, 28)
    assert P4.shape == (2, 256, 14, 14)
    assert P5.shape == (2, 256, 7, 7)
    assert P6.shape == (2, 256, 4, 4)
    assert P7.shape == (2, 256, 2, 2)

    print("\n✓ FPN test passed")


if __name__ == "__main__":
    main()