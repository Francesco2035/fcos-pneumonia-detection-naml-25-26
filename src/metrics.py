import torch


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

    This is mathematically equivalent to the previous
    implementation, but fully vectorized.

    Args:
        boxes1: Tensor[N, 4]
        boxes2: Tensor[M, 4]

    Returns:
        Tensor[N, M]
    """

    n = boxes1.shape[0]
    m = boxes2.shape[0]

    if n == 0 or m == 0:

        return torch.zeros(
            (n, m),
            dtype=torch.float32,
            device=boxes1.device,
        )

    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

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
    # Box areas
    # --------------------------------------------------------

    width1 = torch.clamp(
        boxes1[:, 2] - boxes1[:, 0],
        min=0.0,
    )

    height1 = torch.clamp(
        boxes1[:, 3] - boxes1[:, 1],
        min=0.0,
    )

    width2 = torch.clamp(
        boxes2[:, 2] - boxes2[:, 0],
        min=0.0,
    )

    height2 = torch.clamp(
        boxes2[:, 3] - boxes2[:, 1],
        min=0.0,
    )

    area1 = width1 * height1
    area2 = width2 * height2

    # --------------------------------------------------------
    # Union + IoU
    # --------------------------------------------------------

    union_area = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection_area
    )

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

    This preserves the exact semantics of the original
    implementation:

        1. predictions are sorted by descending score;
        2. every prediction chooses its best-IoU GT;
        3. a prediction is TP iff:
              best_iou >= threshold
              AND that GT has not already been matched;
        4. otherwise it is FP.

    The expensive pairwise IoU computation and duplicate
    resolution are vectorized.
    """

    boxes_pred = pred["boxes"]
    boxes_gt = gt["boxes"]
    scores = pred["scores"]

    # --------------------------------------------------------
    # Sort predictions by descending confidence.
    # --------------------------------------------------------

    order = scores.argsort(
        descending=True
    )

    sorted_boxes = boxes_pred[order]
    sorted_scores = scores[order]

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
            "scores": sorted_scores,
            "tp": torch.empty(
                (0,),
                dtype=torch.int64,
                device=scores.device,
            ),
            "fp": torch.empty(
                (0,),
                dtype=torch.int64,
                device=scores.device,
            ),
            "matched_gt": set(),
            "num_gt": num_gt,
        }

    # --------------------------------------------------------
    # No GT
    #
    # Every prediction is FP.
    # --------------------------------------------------------

    if num_gt == 0:

        return {
            "scores": sorted_scores,
            "tp": torch.zeros(
                num_predictions,
                dtype=torch.int64,
                device=scores.device,
            ),
            "fp": torch.ones(
                num_predictions,
                dtype=torch.int64,
                device=scores.device,
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
    #   best_iou[p]
    #   best_gt[p]
    #
    # exactly corresponds to:
    #
    # torch.max(iou_matrix[pred_idx])
    #

    best_iou, best_gt = (
        torch.max(
            iou_matrix,
            dim=1,
        )
    )

    # --------------------------------------------------------
    # Predictions satisfying IoU threshold.
    # --------------------------------------------------------

    valid = (
        best_iou
        >=
        iou_threshold
    )

    # --------------------------------------------------------
    # Resolve duplicates exactly like the original greedy
    # matching:
    #
    # because predictions are already ordered by descending
    # score, the FIRST qualifying prediction assigned to each
    # GT becomes TP.
    #
    # Every subsequent qualifying prediction for that GT
    # becomes FP.
    # --------------------------------------------------------

    prediction_positions = (
        torch.arange(
            num_predictions,
            device=scores.device,
            dtype=torch.long,
        )
    )

    # We only care about qualifying predictions.
    valid_positions = (
        prediction_positions[valid]
    )

    valid_gt = (
        best_gt[valid]
    )

    # One position per GT, initialized to "not found".
    first_match_position = torch.full(
        (
            num_gt,
        ),
        num_predictions,
        dtype=torch.long,
        device=scores.device,
    )

    # For each GT, get the FIRST qualifying prediction
    # position.
    #
    # This replaces the Python "matched_gt" set while
    # preserving the original score-ordered greedy semantics.
    first_match_position.scatter_reduce_(
        0,
        valid_gt,
        valid_positions,
        reduce="amin",
        include_self=True,
    )

    # --------------------------------------------------------
    # TP:
    #
    # a prediction is TP exactly when:
    #
    #   1. IoU >= threshold
    #   2. it is the first qualifying prediction for its GT
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

    fp = (
        1 - tp
    )

    # --------------------------------------------------------
    # Matched GT set
    #
    # Kept for API compatibility with the original code.
    # --------------------------------------------------------

    matched_gt_tensor = (
        best_gt[tp_mask]
    )

    matched_gt = set(
        matched_gt_tensor
        .detach()
        .cpu()
        .tolist()
    )

    return {
        "scores": sorted_scores,
        "tp": tp,
        "fp": fp,
        "matched_gt": matched_gt,
        "num_gt": num_gt,
    }


# ============================================================
# AP
# ============================================================

def _compute_ap(
    predictions,
    targets,
    iou_threshold=0.5,
):
    """
    Compute Average Precision over the whole dataset.

    Matching is image-local.
    AP aggregation is dataset-global.

    Uses the same 101-point interpolated precision/recall
    calculation as the original implementation.
    """

    all_scores = []
    all_tp = []

    total_gt = 0

    # --------------------------------------------------------
    # Per-image matching
    # --------------------------------------------------------

    for pred, gt in zip(
        predictions,
        targets,
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

    # --------------------------------------------------------
    # No GT
    # --------------------------------------------------------

    if total_gt == 0:

        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": 0,
            "num_pred": int(
                sum(
                    tensor.numel()
                    for tensor in all_scores
                )
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

    scores = torch.cat(
        all_scores,
        dim=0,
    )

    tp = torch.cat(
        all_tp,
        dim=0,
    ).to(
        torch.float32
    )

    if scores.numel() == 0:

        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": total_gt,
            "num_pred": 0,
        }

    # --------------------------------------------------------
    # Global ranking required for AP
    # --------------------------------------------------------

    order = scores.argsort(
        descending=True
    )

    tp = tp[order]

    fp = 1.0 - tp

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
    # Identical numerical procedure to the original.
    # --------------------------------------------------------

    ap = 0.0

    recall_grid = torch.linspace(
        0.0,
        1.0,
        steps=101,
        device=recalls.device,
    )

    for recall_threshold in (
        recall_grid
    ):

        valid_precisions = (
            precisions[
                recalls
                >= recall_threshold
            ]
        )

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


# ============================================================
# Box area helpers
# ============================================================

def _compute_box_areas(
    boxes,
):
    """
    Return XYXY box areas as a 1D tensor.
    """

    if len(boxes) == 0:

        return torch.empty(
            (0,),
            dtype=torch.float32,
            device=boxes.device,
        )

    widths = torch.clamp(
        boxes[:, 2] - boxes[:, 0],
        min=0.0,
    )

    heights = torch.clamp(
        boxes[:, 3] - boxes[:, 1],
        min=0.0,
    )

    return widths * heights


# ============================================================
# Area-filtered targets
# ============================================================

def _filter_targets_by_area(
    targets,
    min_area=None,
    max_area=None,
):
    """
    Keep only GT boxes whose area is within the requested
    interval.

    Bounds:
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
                >=
                min_area
            )

        if max_area is not None:

            mask &= (
                areas
                <
                max_area
            )

        new_target = {
            "boxes": boxes[mask],
        }

        if labels is not None:

            new_target["labels"] = (
                labels[mask]
            )

        filtered.append(
            new_target
        )

    return filtered


# ============================================================
# AR
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
        1. keep top-K predictions by score;
        2. greedily match predictions to GT;
        3. count unique recalled GT boxes.

    The greedy matching is vectorized while preserving the
    original score-ordered behavior.
    """

    total_gt = 0
    total_recalled = 0

    for pred, gt in zip(
        predictions,
        targets,
    ):

        gt_boxes = gt["boxes"]

        pred_boxes = pred["boxes"]

        pred_scores = pred["scores"]

        total_gt += len(
            gt_boxes
        )

        if (
            len(gt_boxes) == 0
            or
            len(pred_boxes) == 0
        ):
            continue

        # ----------------------------------------------------
        # Keep highest-scoring detections.
        # ----------------------------------------------------

        order = (
            pred_scores
            .argsort(
                descending=True
            )[:max_dets]
        )

        pred_boxes = (
            pred_boxes[order]
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
            >=
            iou_threshold
        )

        if not torch.any(valid):
            continue

        # ----------------------------------------------------
        # Same greedy rule as original:
        #
        # first qualifying prediction for every GT wins.
        # ----------------------------------------------------

        prediction_positions = (
            torch.arange(
                len(pred_boxes),
                device=pred_boxes.device,
                dtype=torch.long,
            )
        )

        valid_positions = (
            prediction_positions[
                valid
            ]
        )

        valid_gt = (
            best_gt[valid]
        )

        first_match_position = (
            torch.full(
                (
                    len(gt_boxes),
                ),
                len(pred_boxes),
                dtype=torch.long,
                device=pred_boxes.device,
            )
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
    Compute the metrics reported by the project.

    Returns:
        AP      : AP at IoU 0.5
        AP_M    : AP for medium GT boxes
        AP_L    : AP for large GT boxes
        AR@10   : recall using at most 10 detections/image
        AR_M    : recall for medium GT boxes
        AR_L    : recall for large GT boxes
    """

    # --------------------------------------------------------
    # Main AP
    # --------------------------------------------------------

    ap = _compute_ap(
        predictions,
        targets,
        iou_threshold=0.5,
    )["AP"]

    # --------------------------------------------------------
    # Medium / large GT boxes
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

    ap_m = _compute_ap(
        predictions,
        medium_targets,
        iou_threshold=0.5,
    )["AP"]

    # --------------------------------------------------------
    # AP_L
    # --------------------------------------------------------

    ap_l = _compute_ap(
        predictions,
        large_targets,
        iou_threshold=0.5,
    )["AP"]

    # --------------------------------------------------------
    # AR@10
    # --------------------------------------------------------

    ar_10 = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    # --------------------------------------------------------
    # AR_M
    # --------------------------------------------------------

    ar_m = _compute_ar(
        predictions,
        medium_targets,
        iou_threshold=0.5,
        max_dets=100,
    )

    # --------------------------------------------------------
    # AR_L
    # --------------------------------------------------------

    ar_l = _compute_ar(
        predictions,
        large_targets,
        iou_threshold=0.5,
        max_dets=100,
    )

    return {
        "AP": ap,
        "AP_M": ap_m,
        "AP_L": ap_l,
        "AR@10": ar_10,
        "AR_M": ar_m,
        "AR_L": ar_l,
    }