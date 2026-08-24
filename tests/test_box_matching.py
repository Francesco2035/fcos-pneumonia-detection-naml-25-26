import torch
import torch.nn.functional as F

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    RESNET50_CHEST_XRAY_CHECKPOINT,
    VAL_RATIO,
    SEED,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_test_transforms,
)

from src.datasets.split import (
    create_train_val_split,
)

from src.models.detector import (
    DetectionFramework,
)

from src.inference import (
    DetectionPostProcessor,
)


def compute_iou(box, boxes):
    """
    Compute IoU between one box [4] and a set of boxes [N, 4].
    """

    if boxes.shape[0] == 0:
        return torch.empty(
            0,
            device=box.device,
        )

    x1 = torch.maximum(
        box[0],
        boxes[:, 0],
    )

    y1 = torch.maximum(
        box[1],
        boxes[:, 1],
    )

    x2 = torch.minimum(
        box[2],
        boxes[:, 2],
    )

    y2 = torch.minimum(
        box[3],
        boxes[:, 3],
    )

    intersection = (
        (x2 - x1).clamp(min=0)
        *
        (y2 - y1).clamp(min=0)
    )

    box_area = (
        (box[2] - box[0]).clamp(min=0)
        *
        (box[3] - box[1]).clamp(min=0)
    )

    boxes_area = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        *
        (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    )

    union = (
        box_area
        + boxes_area
        - intersection
    )

    return intersection / union.clamp(min=1e-8)


def test_box_matching():

    print()
    print("=" * 60)
    print("Detection box diagnostics")
    print("=" * 60)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] Device: {device}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use the same deterministic train/validation split
    # as the Trainer.
    # ---------------------------------------------------------

    _, val_indices = create_train_val_split(
        dataset,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(device)

    checkpoint_path = (
        "checkpoints/exp2/best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint["model"]
        )

    else:

        raise KeyError(
            "Model weights not found in checkpoint."
        )

    model.eval()

    print(
        f"[TEST] Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    # ---------------------------------------------------------
    # Postprocessor
    # ---------------------------------------------------------

    postprocessor = DetectionPostProcessor(
        score_threshold=0.0,
        nms_threshold=0.5,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_images = 0
    positive_images = 0

    total_predictions = 0
    valid_predictions = 0

    best_ious = []

    printed_examples = 0

    # ---------------------------------------------------------
    # Inspect validation images
    # ---------------------------------------------------------

    with torch.no_grad():

        for index in val_indices[:100]:

            image, target = dataset[index]

            image = image.unsqueeze(0).to(
                device
            )

            predictions = model(
                image
            )

            detections = postprocessor(
                predictions
            )[0]

            pred_boxes = (
                detections["boxes"]
            )

            pred_scores = (
                detections["scores"]
            )

            gt_boxes = (
                target["boxes"]
                .to(device)
                .float()
            )

            total_images += 1

            if gt_boxes.shape[0] > 0:
                positive_images += 1

            total_predictions += (
                pred_boxes.shape[0]
            )

            # -------------------------------------------------
            # Check box geometry
            # -------------------------------------------------

            if pred_boxes.shape[0] > 0:

                valid_mask = (
                    (pred_boxes[:, 2] > pred_boxes[:, 0])
                    &
                    (pred_boxes[:, 3] > pred_boxes[:, 1])
                )

                valid_predictions += (
                    valid_mask.sum().item()
                )

            # -------------------------------------------------
            # Match predictions to GT
            # -------------------------------------------------

            if (
                gt_boxes.shape[0] > 0
                and pred_boxes.shape[0] > 0
            ):

                image_best_iou = 0.0

                for gt_box in gt_boxes:

                    ious = compute_iou(
                        gt_box,
                        pred_boxes,
                    )

                    if ious.numel() > 0:

                        image_best_iou = max(
                            image_best_iou,
                            ious.max().item(),
                        )

                best_ious.append(
                    image_best_iou
                )

                # -------------------------------------------------
                # Print a few examples
                # -------------------------------------------------

                if printed_examples < 10:

                    print()
                    print(
                        f"[IMAGE] dataset_index={index}"
                    )

                    print(
                        f"GT boxes:\n"
                        f"{gt_boxes.cpu()}"
                    )

                    print(
                        f"Pred boxes:\n"
                        f"{pred_boxes.cpu()}"
                    )

                    print(
                        f"Pred scores:\n"
                        f"{pred_scores.cpu()}"
                    )

                    print(
                        f"Best IoU: "
                        f"{image_best_iou:.4f}"
                    )

                    printed_examples += 1

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("Box diagnostics summary")
    print("=" * 60)

    print(
        f"Images inspected:       "
        f"{total_images}"
    )

    print(
        f"Positive images:        "
        f"{positive_images}"
    )

    print(
        f"Total predictions:      "
        f"{total_predictions}"
    )

    if total_predictions > 0:

        valid_ratio = (
            valid_predictions
            / total_predictions
        )

    else:

        valid_ratio = 0.0

    print(
        f"Valid prediction boxes: "
        f"{valid_predictions} "
        f"({100.0 * valid_ratio:.2f}%)"
    )

    if best_ious:

        mean_best_iou = (
            sum(best_ious)
            / len(best_ious)
        )

        max_best_iou = max(
            best_ious
        )

        print(
            f"Mean best IoU:          "
            f"{mean_best_iou:.4f}"
        )

        print(
            f"Maximum best IoU:       "
            f"{max_best_iou:.4f}"
        )

    else:

        print(
            "No positive images with predictions."
        )

    print("=" * 60)


if __name__ == "__main__":
    test_box_matching()