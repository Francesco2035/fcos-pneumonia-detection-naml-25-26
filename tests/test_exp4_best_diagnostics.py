import torch
from pathlib import Path

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_test_transforms,
)

from src.models.detector import (
    DetectionFramework,
)

from src.models.target_generator import (
    TargetGenerator,
)

from src.inference import (
    DetectionPostProcessor,
)


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = (
    "checkpoints/exp4/best.pt"
)

NUM_IMAGES = 20

LEVELS = (
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
)

STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}


# =========================================================
# CHECKPOINT
# =========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    print()
    print("=" * 80)
    print("CHECKPOINT")
    print("=" * 80)

    print(
        f"path: {checkpoint_path}"
    )

    if isinstance(
        checkpoint,
        dict,
    ):
        print(
            f"keys: {list(checkpoint.keys())}"
        )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "model" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model"
            ]
        )

    else:

        raise KeyError(
            "Could not find model weights "
            "in checkpoint."
        )


# =========================================================
# IOU
# =========================================================

def box_iou(
    boxes1,
    boxes2,
):
    """
    Compute IoU between boxes1 [N,4] and boxes2 [M,4].
    """

    if boxes1.numel() == 0:
        return torch.empty(
            (
                0,
                boxes2.shape[0],
            ),
            device=boxes1.device,
        )

    if boxes2.numel() == 0:
        return torch.empty(
            (
                boxes1.shape[0],
                0,
            ),
            device=boxes1.device,
        )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0])
        .clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1])
        .clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0])
        .clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1])
        .clamp(min=0)
    )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = (
        rb - lt
    ).clamp(
        min=0
    )

    intersection = (
        wh[..., 0]
        * wh[..., 1]
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    ).clamp(
        min=1e-12
    )

    return (
        intersection
        / union
    )


# =========================================================
# SAFE MEAN
# =========================================================

def safe_mean(
    tensor,
):
    if tensor.numel() == 0:
        return 0.0

    return tensor.mean().item()


def safe_max(
    tensor,
):
    if tensor.numel() == 0:
        return 0.0

    return tensor.max().item()


def safe_min(
    tensor,
):
    if tensor.numel() == 0:
        return 0.0

    return tensor.min().item()


# =========================================================
# MAIN TEST
# =========================================================

def test_exp4_best_diagnostics():

    print()
    print("=" * 80)
    print(
        "EXP4 BEST CHECKPOINT DIAGNOSTICS"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] device = {device}"
    )

    print(
        f"[TEST] image size = {IMAGE_SIZE}"
    )

    # -----------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------

    if not Path(
        CHECKPOINT_PATH
    ).exists():

        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    # -----------------------------------------------------
    # Find positive images
    # -----------------------------------------------------

    positive_indices = []

    for index in range(
        len(dataset)
    ):

        patient_id = (
            dataset.image_paths[
                index
            ].stem
        )

        boxes = (
            dataset.annotations[
                patient_id
            ]["boxes"]
        )

        if len(boxes) > 0:

            positive_indices.append(
                index
            )

        if len(
            positive_indices
        ) >= NUM_IMAGES:

            break

    if len(
        positive_indices
    ) == 0:

        raise RuntimeError(
            "No positive images found."
        )

    print(
        f"[TEST] positive images = "
        f"{len(positive_indices)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = DetectionFramework(
        path_model=
            RESNET50_CHEST_XRAY_CHECKPOINT,
    ).to(
        device
    )

    load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    model.eval()

    # -----------------------------------------------------
    # Target generator
    # -----------------------------------------------------

    target_generator = (
        TargetGenerator()
    )

    # -----------------------------------------------------
    # Postprocessor
    # -----------------------------------------------------

    postprocessor = (
        DetectionPostProcessor(
            strides=(
                8,
                16,
                32,
                64,
                128,
            ),
            score_threshold=0.0,
            nms_threshold=0.5,
        )
    ).to(
        device
    )

    # =====================================================
    # ACCUMULATORS
    # =====================================================

    cls_positive = []
    cls_negative = []

    ctr_positive = []
    ctr_negative = []

    score_positive = []
    score_negative = []

    best_ious = []

    best_scores = []

    num_detections = []

    positive_counts = {
        level: []
        for level in LEVELS
    }

    # Per-level classification scores
    level_cls_positive = {
        level: []
        for level in LEVELS
    }

    level_cls_negative = {
        level: []
        for level in LEVELS
    }

    level_ctr_positive = {
        level: []
        for level in LEVELS
    }

    level_ctr_negative = {
        level: []
        for level in LEVELS
    }

    # =====================================================
    # IMAGE LOOP
    # =====================================================

    with torch.no_grad():

        for image_number, dataset_index in enumerate(
            positive_indices,
            start=1,
        ):

            print()
            print("-" * 80)
            print(
                f"IMAGE {image_number}/"
                f"{len(positive_indices)}"
            )
            print("-" * 80)

            # -------------------------------------------------
            # Load image
            # -------------------------------------------------

            image, target = dataset[
                dataset_index
            ]

            image = (
                image
                .unsqueeze(0)
                .to(device)
            )

            gt_boxes = (
                target["boxes"]
                .to(device)
                .float()
            )

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            predictions = model(
                image
            )

            # =================================================
            # Per-level score diagnostics
            # =================================================

            image_positive_cls = []
            image_negative_cls = []

            image_positive_ctr = []
            image_negative_ctr = []

            image_positive_score = []
            image_negative_score = []

            for level in LEVELS:

                stride = STRIDES[level]

                classification = (
                    predictions[level][
                        "classification"
                    ][0, 0]
                )

                centerness = (
                    predictions[level][
                        "centerness"
                    ][0, 0]
                )

                # ---------------------------------------------
                # Generate target
                # ---------------------------------------------

                _, _, H, W = (
                    predictions[level][
                        "classification"
                    ].shape
                )

                target_level = (
                    target_generator.generate_targets(
                        label_boxes=gt_boxes,
                        feature_shape=(
                            H,
                            W,
                        ),
                        stride=stride,
                        device=device,
                    )
                )

                positive = (
                    target_level[
                        "positive"
                    ]
                    .bool()
                )

                positive_counts[
                    level
                ].append(
                    positive.sum()
                    .item()
                )

                # ---------------------------------------------
                # Convert logits to probabilities
                # ---------------------------------------------

                cls_prob = (
                    torch.sigmoid(
                        classification
                    )
                )

                ctr_prob = (
                    torch.sigmoid(
                        centerness
                    )
                )

                score = torch.sqrt(
                    cls_prob
                    * ctr_prob
                )

                # ---------------------------------------------
                # Positive / negative
                # ---------------------------------------------

                pos_cls = (
                    cls_prob[
                        positive
                    ]
                )

                neg_cls = (
                    cls_prob[
                        ~positive
                    ]
                )

                pos_ctr = (
                    ctr_prob[
                        positive
                    ]
                )

                neg_ctr = (
                    ctr_prob[
                        ~positive
                    ]
                )

                pos_score = (
                    score[
                        positive
                    ]
                )

                neg_score = (
                    score[
                        ~positive
                    ]
                )

                image_positive_cls.append(
                    pos_cls
                )

                image_negative_cls.append(
                    neg_cls
                )

                image_positive_ctr.append(
                    pos_ctr
                )

                image_negative_ctr.append(
                    neg_ctr
                )

                image_positive_score.append(
                    pos_score
                )

                image_negative_score.append(
                    neg_score
                )

                # ---------------------------------------------
                # Accumulate per-level
                # ---------------------------------------------

                if pos_cls.numel() > 0:

                    level_cls_positive[
                        level
                    ].append(
                        pos_cls.mean().item()
                    )

                    level_ctr_positive[
                        level
                    ].append(
                        pos_ctr.mean().item()
                    )

                if neg_cls.numel() > 0:

                    level_cls_negative[
                        level
                    ].append(
                        neg_cls.mean().item()
                    )

                    level_ctr_negative[
                        level
                    ].append(
                        neg_ctr.mean().item()
                    )

            # =================================================
            # Aggregate image score statistics
            # =================================================

            image_pos_cls = torch.cat(
                [
                    x
                    for x in image_positive_cls
                    if x.numel() > 0
                ],
                dim=0,
            )

            image_neg_cls = torch.cat(
                [
                    x
                    for x in image_negative_cls
                    if x.numel() > 0
                ],
                dim=0,
            )

            image_pos_ctr = torch.cat(
                [
                    x
                    for x in image_positive_ctr
                    if x.numel() > 0
                ],
                dim=0,
            )

            image_neg_ctr = torch.cat(
                [
                    x
                    for x in image_negative_ctr
                    if x.numel() > 0
                ],
                dim=0,
            )

            image_pos_score = torch.cat(
                [
                    x
                    for x in image_positive_score
                    if x.numel() > 0
                ],
                dim=0,
            )

            image_neg_score = torch.cat(
                [
                    x
                    for x in image_negative_score
                    if x.numel() > 0
                ],
                dim=0,
            )

            # -------------------------------------------------
            # Store
            # -------------------------------------------------

            cls_positive.append(
                image_pos_cls.mean().item()
            )

            cls_negative.append(
                image_neg_cls.mean().item()
            )

            ctr_positive.append(
                image_pos_ctr.mean().item()
            )

            ctr_negative.append(
                image_neg_ctr.mean().item()
            )

            score_positive.append(
                image_pos_score.mean().item()
            )

            score_negative.append(
                image_neg_score.mean().item()
            )

            # =================================================
            # Actual postprocessed predictions
            # =================================================

            detections = (
                postprocessor(
                    predictions
                )[0]
            )

            boxes = (
                detections[
                    "boxes"
                ]
            )

            scores = (
                detections[
                    "scores"
                ]
            )

            num_detections.append(
                boxes.shape[0]
            )

            # -------------------------------------------------
            # Best prediction IoU
            # -------------------------------------------------

            if (
                boxes.shape[0] > 0
                and gt_boxes.shape[0] > 0
            ):

                ious = box_iou(
                    boxes,
                    gt_boxes,
                )

                best_iou = (
                    ious.max()
                    .item()
                )

                best_index = (
                    ious.max(
                        dim=0
                    ).values.argmax()
                    if ious.numel() > 0
                    else None
                )

                best_score = (
                    scores.max().item()
                )

            else:

                best_iou = 0.0
                best_score = 0.0

            best_ious.append(
                best_iou
            )

            best_scores.append(
                best_score
            )

            print(
                f"positive cls mean = "
                f"{image_pos_cls.mean().item():.6f}"
            )

            print(
                f"negative cls mean = "
                f"{image_neg_cls.mean().item():.6f}"
            )

            print(
                f"positive ctr mean = "
                f"{image_pos_ctr.mean().item():.6f}"
            )

            print(
                f"negative ctr mean = "
                f"{image_neg_ctr.mean().item():.6f}"
            )

            print(
                f"positive score mean = "
                f"{image_pos_score.mean().item():.6f}"
            )

            print(
                f"negative score mean = "
                f"{image_neg_score.mean().item():.6f}"
            )

            print(
                f"num detections = "
                f"{boxes.shape[0]}"
            )

            print(
                f"best IoU = "
                f"{best_iou:.6f}"
            )

            print(
                f"best score = "
                f"{best_score:.6f}"
            )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print()
    print("=" * 80)
    print(
        "EXP4 BEST CHECKPOINT SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        "GLOBAL SCORE DIAGNOSTICS"
    )

    print("-" * 80)

    print(
        f"positive classification mean: "
        f"{sum(cls_positive) / len(cls_positive):.6f}"
    )

    print(
        f"negative classification mean: "
        f"{sum(cls_negative) / len(cls_negative):.6f}"
    )

    print(
        f"positive centerness mean:     "
        f"{sum(ctr_positive) / len(ctr_positive):.6f}"
    )

    print(
        f"negative centerness mean:     "
        f"{sum(ctr_negative) / len(ctr_negative):.6f}"
    )

    print(
        f"positive score mean:           "
        f"{sum(score_positive) / len(score_positive):.6f}"
    )

    print(
        f"negative score mean:           "
        f"{sum(score_negative) / len(score_negative):.6f}"
    )

    print()
    print(
        "DETECTION QUALITY"
    )

    print("-" * 80)

    print(
        f"mean detections/image: "
        f"{sum(num_detections) / len(num_detections):.4f}"
    )

    print(
        f"mean best IoU:          "
        f"{sum(best_ious) / len(best_ious):.6f}"
    )

    print(
        f"max best IoU:           "
        f"{max(best_ious):.6f}"
    )

    print(
        f"mean best score:        "
        f"{sum(best_scores) / len(best_scores):.6f}"
    )

    # =====================================================
    # IMPORTANT RANKING CHECK
    # =====================================================

    positive_score_mean = (
        sum(score_positive)
        / len(score_positive)
    )

    negative_score_mean = (
        sum(score_negative)
        / len(score_negative)
    )

    positive_ctr_mean = (
        sum(ctr_positive)
        / len(ctr_positive)
    )

    negative_ctr_mean = (
        sum(ctr_negative)
        / len(ctr_negative)
    )

    positive_cls_mean = (
        sum(cls_positive)
        / len(cls_positive)
    )

    negative_cls_mean = (
        sum(cls_negative)
        / len(cls_negative)
    )

    print()
    print(
        "=" * 80
    )

    print(
        "RANKING CHECK"
    )

    print(
        f"CLS positive > negative: "
        f"{positive_cls_mean > negative_cls_mean}"
    )

    print(
        f"CTR positive > negative: "
        f"{positive_ctr_mean > negative_ctr_mean}"
    )

    print(
        f"SCORE positive > negative: "
        f"{positive_score_mean > negative_score_mean}"
    )

    print(
        "=" * 80
    )

    # =====================================================
    # PER LEVEL
    # =====================================================

    print()
    print(
        "PER-LEVEL POSITIVE/NEGATIVE MEANS"
    )

    print("-" * 100)

    print(
        f"{'level':8s}"
        f"{'pos':>10s}"
        f"{'cls+':>14s}"
        f"{'cls-':>14s}"
        f"{'ctr+':>14s}"
        f"{'ctr-':>14s}"
    )

    print("-" * 100)

    for level in LEVELS:

        pos_mean = (
            sum(
                positive_counts[level]
            )
            /
            max(
                1,
                len(
                    positive_counts[level]
                ),
            )
        )

        cls_pos_mean = (
            sum(
                level_cls_positive[level]
            )
            /
            max(
                1,
                len(
                    level_cls_positive[level]
                ),
            )
        )

        cls_neg_mean = (
            sum(
                level_cls_negative[level]
            )
            /
            max(
                1,
                len(
                    level_cls_negative[level]
                ),
            )
        )

        ctr_pos_mean = (
            sum(
                level_ctr_positive[level]
            )
            /
            max(
                1,
                len(
                    level_ctr_positive[level]
                ),
            )
        )

        ctr_neg_mean = (
            sum(
                level_ctr_negative[level]
            )
            /
            max(
                1,
                len(
                    level_ctr_negative[level]
                ),
            )
        )

        print(
            f"{level:8s}"
            f"{pos_mean:10.2f}"
            f"{cls_pos_mean:14.6f}"
            f"{cls_neg_mean:14.6f}"
            f"{ctr_pos_mean:14.6f}"
            f"{ctr_neg_mean:14.6f}"
        )

    print()
    print("=" * 80)
    print(
        "No optimizer.step() was performed."
    )
    print(
        "The checkpoint was not modified."
    )
    print("=" * 80)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    test_exp4_best_diagnostics()