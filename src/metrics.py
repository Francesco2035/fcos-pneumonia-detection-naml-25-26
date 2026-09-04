import torch


# ============================================================
# COCO-style area thresholds used by the project
# ============================================================

MEDIUM_AREA_MIN = 32.0 ** 2
MEDIUM_AREA_MAX = 96.0 ** 2
LARGE_AREA_MIN = 96.0 ** 2


# ============================================================
# IoU
# ============================================================

def _compute_iou_matrix(
    boxes1,
    boxes2,
):
    """
    Compute pairwise IoU between two sets of XYXY boxes.

    This implementation is fully vectorized.

    Args:
        boxes1: Tensor[N, 4]
        boxes2: Tensor[M, 4]

    Returns:
        Tensor[N, M]
    """

    # --------------------------------------------------------
    # Keep both tensors on the same device.
    # The first tensor determines the computation device.
    # --------------------------------------------------------

    device = boxes1.device

    boxes2 = boxes2.to(device)

    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    n = boxes1.shape[0]
    m = boxes2.shape[0]

    if n == 0 or m == 0:

        return torch.zeros(
            (n, m),
            dtype=torch.float32,
            device=device,
        )

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    intersection_wh = torch.clamp(
        bottom_right - top_left,
        min=0.0,
    )

    intersection_area = (
        intersection_wh[..., 0]
        *
        intersection_wh[..., 1]
    )

    # --------------------------------------------------------
    # Area of boxes1
    # --------------------------------------------------------

    width1 = torch.clamp(
        boxes1[:, 2] - boxes1[:, 0],
        min=0.0,
    )

    height1 = torch.clamp(
        boxes1[:, 3] - boxes1[:, 1],
        min=0.0,
    )

    area1 = (
        width1 * height1
    )

    # --------------------------------------------------------
    # Area of boxes2
    # --------------------------------------------------------

    width2 = torch.clamp(
        boxes2[:, 2] - boxes2[:, 0],
        min=0.0,
    )

    height2 = torch.clamp(
        boxes2[:, 3] - boxes2[:, 1],
        min=0.0,
    )

    area2 = (
        width2 * height2
    )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    union_area = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection_area
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    return (
        intersection_area
        /
        (union_area + 1e-6)
    )


# ============================================================
# Greedy matching
# ============================================================

def _match_predictions(
    pred,
    gt,
    iou_threshold=0.5,
):
    """
    Match predictions to GT boxes for one image.

    Original semantics are preserved:

        1. predictions are sorted by descending score;
        2. each prediction selects its best-IoU GT;
        3. IoU < threshold -> FP;
        4. already matched GT -> FP;
        5. otherwise -> TP.

    GPU is used for:
        - IoU computation
        - best-GT computation
        - duplicate resolution

    Only the compact matching results are moved to CPU
    after processing the image.
    """

    # --------------------------------------------------------
    # Prediction tensors
    # --------------------------------------------------------

    boxes_pred = pred["boxes"]
    scores = pred["scores"]

    # Predictions should already live on the same device.
    # Use scores as the canonical device.
    device = scores.device

    boxes_pred = boxes_pred.to(
        device
    )

    # --------------------------------------------------------
    # Ground-truth boxes
    #
    # DetectionEvaluator stores GT on CPU, so move only the
    # GT boxes for this image to the prediction device.
    # --------------------------------------------------------

    boxes_gt = gt["boxes"].to(
        device
    )

    # --------------------------------------------------------
    # Sort predictions by confidence
    # --------------------------------------------------------

    order = scores.argsort(
        descending=True
    )

    sorted_boxes = boxes_pred[
        order
    ]

    sorted_scores = scores[
        order
    ]

    num_predictions = (
        sorted_boxes.shape[0]
    )

    num_gt = (
        boxes_gt.shape[0]
    )

    # --------------------------------------------------------
    # No predictions
    # --------------------------------------------------------

    if num_predictions == 0:

        return {
            "scores": sorted_scores.cpu(),
            "tp": torch.empty(
                (0,),
                dtype=torch.int64,
            ),
            "fp": torch.empty(
                (0,),
                dtype=torch.int64,
            ),
            "matched_gt": set(),
            "num_gt": num_gt,
        }

    # --------------------------------------------------------
    # No ground truth
    # --------------------------------------------------------

    if num_gt == 0:

        return {
            "scores": sorted_scores.cpu(),
            "tp": torch.zeros(
                num_predictions,
                dtype=torch.int64,
            ),
            "fp": torch.ones(
                num_predictions,
                dtype=torch.int64,
            ),
            "matched_gt": set(),
            "num_gt": 0,
        }

    # --------------------------------------------------------
    # Pairwise IoU
    # --------------------------------------------------------

    iou_matrix = _compute_iou_matrix(
        sorted_boxes,
        boxes_gt,
    )

    # For every prediction:
    #
    #     best_iou[p]
    #     best_gt[p]
    #
    # are exactly the values previously obtained by:
    #
    #     torch.max(iou_matrix[pred_idx], dim=0)
    #

    best_iou, best_gt = torch.max(
        iou_matrix,
        dim=1,
    )

    # --------------------------------------------------------
    # Predictions passing the IoU threshold
    # --------------------------------------------------------

    valid = (
        best_iou
        >= iou_threshold
    )

    # --------------------------------------------------------
    # Prediction positions in descending-score order
    # --------------------------------------------------------

    prediction_positions = torch.arange(
        num_predictions,
        device=device,
        dtype=torch.long,
    )

    valid_positions = (
        prediction_positions[
            valid
        ]
    )

    valid_gt = (
        best_gt[
            valid
        ]
    )

    # --------------------------------------------------------
    # First qualifying prediction for each GT
    #
    # Since predictions are already sorted by descending
    # confidence, the first valid prediction for a GT is
    # exactly the one selected by the original greedy matcher.
    # --------------------------------------------------------

    first_match_position = torch.full(
        (
            num_gt,
        ),
        num_predictions,
        dtype=torch.long,
        device=device,
    )

    if valid_gt.numel() > 0:

        first_match_position.scatter_reduce_(
            0,
            valid_gt,
            valid_positions,
            reduce="amin",
            include_self=True,
        )

    # --------------------------------------------------------
    # True positives
    # --------------------------------------------------------

    tp_mask = (
        valid
        &
        (
            prediction_positions
            ==
            first_match_position[
                best_gt
            ]
        )
    )

    tp = tp_mask.to(
        torch.int64
    )

    # Every non-TP prediction is FP.
    fp = 1 - tp

    # --------------------------------------------------------
    # Matched GT set
    # --------------------------------------------------------

    matched_gt = set(
        best_gt[
            tp_mask
        ]
        .detach()
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Move only compact matching results to CPU.
    #
    # We do NOT keep millions of detection-related tensors
    # on the GPU for global AP aggregation.
    # --------------------------------------------------------

    return {
        "scores": sorted_scores.detach().cpu(),
        "tp": tp.detach().cpu(),
        "fp": fp.detach().cpu(),
        "matched_gt": matched_gt,
        "num_gt": num_gt,
    }


# ============================================================
# Average Precision
# ============================================================

def _compute_ap(
    predictions,
    targets,
    iou_threshold=0.5,
):
    """
    Compute Average Precision over the full dataset.

    Matching is image-local.
    AP aggregation is dataset-global.

    All global aggregation is performed on CPU.
    """

    all_scores = []
    all_tp = []

    total_gt = 0

    # --------------------------------------------------------
    # Per-image matching
    # --------------------------------------------------------

    print(
        "[METRICS] AP: starting per-image matching..."
    )

    for image_idx, (
        pred,
        gt,
    ) in enumerate(
        zip(
            predictions,
            targets,
        ),
        start=1,
    ):

        result = _match_predictions(
            pred,
            gt,
            iou_threshold=iou_threshold,
        )

        all_scores.append(
            result["scores"]
        )

        all_tp.append(
            result["tp"]
        )

        total_gt += (
            result["num_gt"]
        )

        # Log occasionally because a dataset with millions
        # of detections can take some time.
        if (
            image_idx % 500 == 0
            or image_idx == len(predictions)
        ):

            print(
                f"[METRICS] AP matching: "
                f"{image_idx}/{len(predictions)}"
            )

    # --------------------------------------------------------
    # No ground truth
    # --------------------------------------------------------

    if total_gt == 0:

        num_predictions = sum(
            tensor.numel()
            for tensor in all_scores
        )

        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": 0,
            "num_pred": int(
                num_predictions
            ),
        }

    # --------------------------------------------------------
    # No predictions
    # --------------------------------------------------------

    if len(all_scores) == 0:

        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": total_gt,
            "num_pred": 0,
        }

    # --------------------------------------------------------
    # Global aggregation
    #
    # Every tensor returned by _match_predictions is already
    # on CPU.
    # --------------------------------------------------------

    scores = torch.cat(
        all_scores,
        dim=0,
    )

    tp = torch.cat(
        all_tp,
        dim=0,
    ).float()

    if scores.numel() == 0:

        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": total_gt,
            "num_pred": 0,
        }

    print(
        f"[METRICS] AP: "
        f"{scores.numel()} total predictions"
    )

    # --------------------------------------------------------
    # Global ranking
    # --------------------------------------------------------

    order = scores.argsort(
        descending=True
    )

    tp = tp[
        order
    ]

    fp = (
        1.0 - tp
    )

    # --------------------------------------------------------
    # Cumulative TP / FP
    # --------------------------------------------------------

    tp_cumsum = torch.cumsum(
        tp,
        dim=0,
    )

    fp_cumsum = torch.cumsum(
        fp,
        dim=0,
    )

    recalls = (
        tp_cumsum
        /
        float(total_gt)
    )

    precisions = (
        tp_cumsum
        /
        (
            tp_cumsum
            +
            fp_cumsum
            +
            1e-6
        )
    )

    # --------------------------------------------------------
    # 101-point interpolation
    #
    # Same calculation as the original implementation.
    # --------------------------------------------------------

    recall_grid = torch.linspace(
        0.0,
        1.0,
        steps=101,
    )

    ap = 0.0

    for recall_threshold in (
        recall_grid
    ):

        valid_precisions = precisions[
            recalls
            >= recall_threshold
        ]

        if (
            valid_precisions.numel()
            > 0
        ):

            ap += float(
                valid_precisions.max().item()
            )

    ap /= 101.0

    return {
        "AP": float(ap),
        "precisions": (
            precisions.tolist()
        ),
        "recalls": (
            recalls.tolist()
        ),
        "num_gt": total_gt,
        "num_pred": int(
            scores.numel()
        ),
    }


def compute_ap_curve(
    predictions,
    targets,
    iou_threshold=0.5,
):
    """
    Compute the full precision-recall data used to obtain AP.

    This is intentionally a thin public wrapper around the existing
    ``_compute_ap`` implementation.  The training-time ``compute_metrics``
    function is left unchanged, so this addition does not alter the
    training/validation metric pipeline.

    Returns:
        dict with:
            AP: scalar Average Precision.
            precisions: precision values of the ranked predictions.
            recalls: recall values of the ranked predictions.
            num_gt: total ground-truth boxes.
            num_pred: total predictions.
    """

    return _compute_ap(
        predictions=predictions,
        targets=targets,
        iou_threshold=iou_threshold,
    )


# ============================================================
# Box areas
# ============================================================

def _compute_box_areas(
    boxes,
):
    """
    Compute XYXY box areas.
    """

    if boxes.numel() == 0:

        return torch.empty(
            (0,),
            dtype=torch.float32,
            device=boxes.device,
        )

    boxes = boxes.float()

    widths = torch.clamp(
        boxes[:, 2]
        -
        boxes[:, 0],
        min=0.0,
    )

    heights = torch.clamp(
        boxes[:, 3]
        -
        boxes[:, 1],
        min=0.0,
    )

    return (
        widths
        *
        heights
    )


# ============================================================
# Area-filtered targets
# ============================================================

def _filter_targets_by_area(
    targets,
    min_area=None,
    max_area=None,
):
    """
    Keep only GT boxes whose area is inside the requested range.

    Conditions:
        area >= min_area
        area <  max_area
    """

    filtered = []

    for target in targets:

        boxes = target["boxes"]

        labels = target.get(
            "labels",
            None,
        )

        areas = _compute_box_areas(
            boxes
        )

        mask = torch.ones(
            len(boxes),
            dtype=torch.bool,
            device=boxes.device,
        )

        if min_area is not None:

            mask &= (
                areas
                >= min_area
            )

        if max_area is not None:

            mask &= (
                areas
                < max_area
            )

        new_target = {
            "boxes": boxes[mask],
        }

        if labels is not None:

            new_target[
                "labels"
            ] = labels[mask]

        filtered.append(
            new_target
        )

    return filtered


# ============================================================
# Average Recall
# ============================================================

def _compute_ar(
    predictions,
    targets,
    iou_threshold=0.5,
    max_dets=100,
):
    """
    Compute Average Recall over the dataset.

    For each image:

        1. Keep the top-K predictions by score.
        2. Compute IoU on the prediction device.
        3. Apply the same greedy matching rule as the AP code.
        4. Count recalled GT boxes.

    Only scalar counts return to CPU.
    """

    total_gt = 0
    total_recalled = 0

    print(
        f"[METRICS] AR@{max_dets}: "
        "starting per-image matching..."
    )

    for image_idx, (
        pred,
        gt,
    ) in enumerate(
        zip(
            predictions,
            targets,
        ),
        start=1,
    ):

        gt_boxes = gt["boxes"]

        pred_boxes = pred["boxes"]

        pred_scores = pred["scores"]

        # ----------------------------------------------------
        # Use prediction device.
        # ----------------------------------------------------

        device = pred_scores.device

        gt_boxes = gt_boxes.to(
            device
        )

        pred_boxes = pred_boxes.to(
            device
        )

        total_gt += len(
            gt_boxes
        )

        if (
            len(gt_boxes) == 0
            or len(pred_boxes) == 0
        ):
            continue

        # ----------------------------------------------------
        # Keep top-K predictions
        # ----------------------------------------------------

        order = (
            pred_scores
            .argsort(
                descending=True
            )
            [:max_dets]
        )

        pred_boxes = (
            pred_boxes[
                order
            ]
        )

        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        iou_matrix = (
            _compute_iou_matrix(
                pred_boxes,
                gt_boxes,
            )
        )

        best_iou, best_gt = (
            torch.max(
                iou_matrix,
                dim=1,
            )
        )

        valid = (
            best_iou
            >= iou_threshold
        )

        if not torch.any(
            valid
        ):

            continue

        # ----------------------------------------------------
        # Greedy duplicate resolution
        # ----------------------------------------------------

        prediction_positions = torch.arange(
            len(pred_boxes),
            device=device,
            dtype=torch.long,
        )

        valid_positions = (
            prediction_positions[
                valid
            ]
        )

        valid_gt = (
            best_gt[
                valid
            ]
        )

        first_match_position = torch.full(
            (
                len(gt_boxes),
            ),
            len(pred_boxes),
            dtype=torch.long,
            device=device,
        )

        first_match_position.scatter_reduce_(
            0,
            valid_gt,
            valid_positions,
            reduce="amin",
            include_self=True,
        )

        matched = (
            valid
            &
            (
                prediction_positions
                ==
                first_match_position[
                    best_gt
                ]
            )
        )

        total_recalled += int(
            matched.sum().item()
        )

        if (
            image_idx % 500 == 0
            or image_idx
            == len(predictions)
        ):

            print(
                f"[METRICS] AR@{max_dets}: "
                f"{image_idx}/{len(predictions)}"
            )

    return (
        total_recalled
        /
        max(total_gt, 1)
    )


# ============================================================
# Full metrics
# ============================================================

def compute_metrics(
    predictions,
    targets,
):
    """
    Compute the complete set of project metrics.

    Returns:
        AP
        AP_M
        AP_L
        AR@10
        AR_M
        AR_L
    """

    print()
    print(
        "=" * 70
    )

    print(
        "[METRICS] Starting metric computation"
    )

    print(
        f"[METRICS] Images: "
        f"{len(predictions)}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Main AP
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AP..."
    )

    ap = _compute_ap(
        predictions,
        targets,
        iou_threshold=0.5,
    )["AP"]

    print(
        f"[METRICS] AP = {ap:.6f}"
    )

    # --------------------------------------------------------
    # Medium and large objects
    # --------------------------------------------------------

    medium_targets = (
        _filter_targets_by_area(
            targets,
            min_area=MEDIUM_AREA_MIN,
            max_area=MEDIUM_AREA_MAX,
        )
    )

    large_targets = (
        _filter_targets_by_area(
            targets,
            min_area=LARGE_AREA_MIN,
            max_area=None,
        )
    )

    # --------------------------------------------------------
    # AP_M
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AP_M..."
    )

    ap_m = _compute_ap(
        predictions,
        medium_targets,
        iou_threshold=0.5,
    )["AP"]

    print(
        f"[METRICS] AP_M = {ap_m:.6f}"
    )

    # --------------------------------------------------------
    # AP_L
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AP_L..."
    )

    ap_l = _compute_ap(
        predictions,
        large_targets,
        iou_threshold=0.5,
    )["AP"]

    print(
        f"[METRICS] AP_L = {ap_l:.6f}"
    )

    # --------------------------------------------------------
    # AR@10
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AR@10..."
    )

    ar_10 = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    print(
        f"[METRICS] AR@10 = {ar_10:.6f}"
    )

    # --------------------------------------------------------
    # AR_M
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AR_M..."
    )

    ar_m = _compute_ar(
        predictions,
        medium_targets,
        iou_threshold=0.5,
        max_dets=100,
    )

    print(
        f"[METRICS] AR_M = {ar_m:.6f}"
    )

    # --------------------------------------------------------
    # AR_L
    # --------------------------------------------------------

    print(
        "[METRICS] Computing AR_L..."
    )

    ar_l = _compute_ar(
        predictions,
        large_targets,
        iou_threshold=0.5,
        max_dets=100,
    )

    print(
        f"[METRICS] AR_L = {ar_l:.6f}"
    )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = {
        "AP": ap,
        "AP_M": ap_m,
        "AP_L": ap_l,
        "AR@10": ar_10,
        "AR_M": ar_m,
        "AR_L": ar_l,
    }

    print()
    print(
        "-" * 70
    )

    print(
        "[METRICS] Final results"
    )

    for (
        name,
        value,
    ) in metrics.items():

        print(
            f"[METRICS] "
            f"{name:<8} = "
            f"{value:.6f}"
        )

    print(
        "-" * 70
    )

    return metrics