import torch

from src.models.fpn import FPN

from src.models.detection_head import DetectionHead



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

    P3_head = DetectionHead(
        in_channels=256,    
        hidden_channels=256,
        center_channels=1,
        scale_channels=4,
    ).to(device)

    center, scale = P3_head(P3)

    print("P3:", P3.shape)
    print("Center:", center.shape)
    print("Scale:", scale.shape)

    print(center)
    print(scale)







if __name__ == "__main__":
    main()