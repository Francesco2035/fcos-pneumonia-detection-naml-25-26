import os

import torch
from torch.utils.data import DataLoader, Subset

from src.models.detector import DetectionFramework
from src.inference import DetectionPostProcessor
from src.evaluate import DetectionEvaluator
from src.metrics import _compute_iou_matrix

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import get_test_transforms


CSV_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

TRAIN_DCM_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)

# -------------------------------------------------------------
# Evaluation size
#
# FULL_EVAL=1 -> evaluate the whole dataset
# otherwise -> evaluate only the first N positive samples
# -------------------------------------------------------------

FULL_EVAL = os.getenv("FULL_EVAL", "0") == "1"

DEFAULT_NUM_SAMPLES = 4


def build_eval_indices(dataset):
    """
    Select samples for evaluation.

    In the quick mode we select a few images containing GT boxes.
    In full mode we evaluate the whole dataset.
    """

    if FULL_EVAL:
        return list(range(len(dataset)))

    selected_indices = []

    for i in range(len(dataset)):

        _, target = dataset[i]

        if len(target["boxes"]) > 0:
            selected_indices.append(i)

        if len(selected_indices) == DEFAULT_NUM_SAMPLES:
            break

    assert len(selected_indices) > 0, (
        "Could not find images containing ground-truth boxes."
    )

    return selected_indices


def test_evaluator_real_data():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(224),
    )

    assert len(dataset) > 0

    # ---------------------------------------------------------
    # Select evaluation samples
    # ---------------------------------------------------------

    selected_indices = build_eval_indices(dataset)

    print("\nEvaluation setup:")
    print(f"Dataset size: {len(dataset)}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Full evaluation: {FULL_EVAL}")

    if not FULL_EVAL:
        print("Selected positive samples:")

        for index in selected_indices:

            _, target = dataset[index]

            print(
                f"  index={index}, "
                f"GT boxes={len(target['boxes'])}"
            )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    subset = Subset(
        dataset,
        indices=selected_indices,
    )

    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            [item[1] for item in batch],
        ),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = DetectionFramework(
        path_model=None,
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Postprocessor
    # ---------------------------------------------------------

    postprocessor = DetectionPostProcessor(
        score_threshold=0.1,
        nms_threshold=0.5,
    )

    # ---------------------------------------------------------
    # Evaluator
    # ---------------------------------------------------------

    evaluator = DetectionEvaluator(
        model=model,
        postprocessor=postprocessor,
        device=device,
    )

    # ---------------------------------------------------------
    # Run evaluator
    # ---------------------------------------------------------

    metrics = evaluator.evaluate(dataloader)

    # ---------------------------------------------------------
    # Independent diagnostics
    #
    # We run inference once more here only to inspect:
    #   - number of GT boxes
    #   - number of final predictions
    #   - maximum IoU per image
    #
    # This is diagnostic code, not part of the Evaluator.
    # ---------------------------------------------------------

    total_gt = 0
    total_predictions = 0

    max_ious = []

    with torch.no_grad():

        for images, targets in dataloader:

            images = images.to(device)

            predictions = model(images)

            detections = postprocessor(predictions)

            for detection, target in zip(
                detections,
                targets,
            ):

                pred_boxes = detection["boxes"].cpu()
                gt_boxes = target["boxes"].cpu()

                total_predictions += len(pred_boxes)
                total_gt += len(gt_boxes)

                # No GT or no predictions -> no IoU.
                if (
                    len(pred_boxes) == 0
                    or len(gt_boxes) == 0
                ):
                    max_ious.append(0.0)
                    continue

                iou_matrix = _compute_iou_matrix(
                    pred_boxes,
                    gt_boxes,
                )

                max_iou = iou_matrix.max().item()

                max_ious.append(max_iou)

    # ---------------------------------------------------------
    # Metric checks
    # ---------------------------------------------------------

    assert isinstance(metrics, dict)

    expected_metrics = {
        "AP",
        "AP_M",
        "AP_L",
        "AR@10",
        "AR_M",
        "AR_L",
    }

    assert expected_metrics.issubset(
        metrics.keys()
    )

    for name in expected_metrics:

        value = metrics[name]

        assert isinstance(
            value,
            (float, int),
        )

        assert torch.isfinite(
            torch.tensor(value)
        )

        assert value >= 0.0

    # ---------------------------------------------------------
    # Final diagnostics
    # ---------------------------------------------------------

    print("\nEvaluator test passed.")
    print(f"Device: {device}")

    print("\nDataset diagnostics:")
    print(f"Ground-truth boxes: {total_gt}")
    print(f"Final predictions:  {total_predictions}")

    if len(max_ious) > 0:

        max_iou_overall = max(max_ious)

        print(
            f"Maximum IoU observed: "
            f"{max_iou_overall:.6f}"
        )

    print("\nMetrics:")

    for name in sorted(expected_metrics):

        print(
            f"{name}: "
            f"{metrics[name]:.6f}"
        )


if __name__ == "__main__":
    test_evaluator_real_data()