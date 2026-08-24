import torch


MEDIUM_AREA_MIN = 32.0 ** 2
MEDIUM_AREA_MAX = 96.0 ** 2
LARGE_AREA_MIN = 96.0 ** 2


# ============================================================
# IoU
# ============================================================


def _compute_iou_matrix(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of XYXY boxes.

    Args:
        boxes1: Tensor[N, 4]
        boxes2: Tensor[M, 4]

    Returns:
        Tensor[N, M], where result[i, j] = IoU(boxes1[i], boxes2[j])
    """

    N, _ = boxes1.shape
    M, _ = boxes2.shape

    iou_matrix = torch.zeros(
        (N, M),
        dtype=torch.float32,
        device=boxes1.device,
    )

    if N == 0 or M == 0:
        return iou_matrix

    for i in range(N):
        for j in range(M):
            box1 = boxes1[i]
            box2 = boxes2[j]

            x11, y11, x12, y12 = box1
            x21, y21, x22, y22 = box2

            # Intersection rectangle.
            x_left = torch.maximum(x11, x21)
            y_top = torch.maximum(y11, y21)
            x_right = torch.minimum(x12, x22)
            y_bottom = torch.minimum(y12, y22)

            # No overlap -> zero width/height.
            intersection_width = torch.clamp(
                x_right - x_left,
                min=0.0,
            )
            intersection_height = torch.clamp(
                y_bottom - y_top,
                min=0.0,
            )

            intersection_area = (
                intersection_width * intersection_height
            )

            area1 = torch.clamp(x12 - x11, min=0.0) * torch.clamp(
                y12 - y11, min=0.0
            )
            area2 = torch.clamp(x22 - x21, min=0.0) * torch.clamp(
                y22 - y21, min=0.0
            )

            union_area = area1 + area2 - intersection_area

            # Small epsilon avoids division by zero.
            iou = intersection_area / (union_area + 1e-6)
            iou_matrix[i, j] = iou

    return iou_matrix


# ============================================================
# Matching
# ============================================================


def _match_predictions(pred, gt, iou_threshold=0.5):
    """
    Match predictions to GT boxes for one image.

    Predictions are processed by descending confidence score.
    Each GT box can be matched at most once.

    Returns a dictionary with scores, TP/FP flags, matched GT indices,
    and the number of GT boxes in the image.
    """

    boxes_pred = pred["boxes"]
    boxes_gt = gt["boxes"]
    scores = pred["scores"]

    order = scores.argsort(descending=True)
    sorted_scores = scores[order]

    matched_gt = set()
    tp = []
    fp = []

    # No predictions.
    if len(boxes_pred) == 0:
        return {
            "scores": sorted_scores,
            "tp": torch.empty(
                (0,), dtype=torch.int64, device=scores.device
            ),
            "fp": torch.empty(
                (0,), dtype=torch.int64, device=scores.device
            ),
            "matched_gt": matched_gt,
            "num_gt": len(boxes_gt),
        }

    # No GT: every prediction is a false positive.
    if len(boxes_gt) == 0:
        num_predictions = len(boxes_pred)
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
            "matched_gt": matched_gt,
            "num_gt": 0,
        }

    iou_matrix = _compute_iou_matrix(boxes_pred, boxes_gt)

    for pred_idx in order:
        prediction_ious = iou_matrix[pred_idx]

        best_iou, best_gt = torch.max(prediction_ious, dim=0)
        best_iou = float(best_iou.item())
        best_gt = int(best_gt.item())

        # Follows the >= 0.5 convention used by the students' implementation.
        if best_iou < iou_threshold:
            tp.append(0)
            fp.append(1)
            continue

        # Duplicate prediction for an already matched GT.
        if best_gt in matched_gt:
            tp.append(0)
            fp.append(1)
            continue

        tp.append(1)
        fp.append(0)
        matched_gt.add(best_gt)

    return {
        "scores": sorted_scores,
        "tp": torch.tensor(
            tp,
            dtype=torch.int64,
            device=scores.device,
        ),
        "fp": torch.tensor(
            fp,
            dtype=torch.int64,
            device=scores.device,
        ),
        "matched_gt": matched_gt,
        "num_gt": len(boxes_gt),
    }


# ============================================================
# AP
# ============================================================


def _compute_ap(predictions, targets, iou_threshold=0.5):
    """
    Compute Average Precision over the whole dataset.

    Matching is image-local; AP aggregation is dataset-global.
    Uses 101-point interpolated precision/recall, following the reference
    implementation used by the other students when the paper leaves this
    numerical detail unspecified.
    """

    all_scores = []
    all_tp = []
    total_gt = 0

    for pred, gt in zip(predictions, targets):
        result = _match_predictions(
            pred,
            gt,
            iou_threshold=iou_threshold,
        )

        all_scores.append(result["scores"])
        all_tp.append(result["tp"])
        total_gt += result["num_gt"]

    if total_gt == 0:
        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": 0,
            "num_pred": int(sum(t.numel() for t in all_scores)),
        }

    if len(all_scores) == 0:
        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": total_gt,
            "num_pred": 0,
        }

    scores = torch.cat(all_scores, dim=0)
    tp = torch.cat(all_tp, dim=0).to(torch.float32)

    if scores.numel() == 0:
        return {
            "AP": 0.0,
            "precisions": [],
            "recalls": [],
            "num_gt": total_gt,
            "num_pred": 0,
        }

    # Global ranking is required for AP.
    order = scores.argsort(descending=True)
    tp = tp[order]
    fp = 1.0 - tp

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / float(total_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 101-point interpolation.
    ap = 0.0
    recall_grid = torch.linspace(
        0.0,
        1.0,
        steps=101,
        device=recalls.device,
    )

    for recall_threshold in recall_grid:
        valid_precisions = precisions[recalls >= recall_threshold]
        if valid_precisions.numel() > 0:
            ap += float(valid_precisions.max().item())

    ap /= 101.0

    return {
        "AP": float(ap),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "num_gt": total_gt,
        "num_pred": int(scores.numel()),
    }


# ============================================================
# Box area helpers
# ============================================================


def _compute_box_areas(boxes):
    """Return XYXY box areas as a 1D tensor."""

    if len(boxes) == 0:
        return torch.empty(
            (0,),
            dtype=torch.float32,
            device=boxes.device,
        )

    widths = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0)
    heights = torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)

    return widths * heights


# ============================================================
# Area-filtered targets
# ============================================================


def _filter_targets_by_area(targets, min_area=None, max_area=None):
    """
    Keep only GT boxes whose area is within the requested interval.

    Bounds are interpreted as:
        area >= min_area
        area <  max_area
    """

    filtered = []

    for target in targets:
        boxes = target["boxes"]
        labels = target.get("labels", None)

        areas = _compute_box_areas(boxes)
        mask = torch.ones(
            len(boxes),
            dtype=torch.bool,
            device=boxes.device,
        )

        if min_area is not None:
            mask &= areas >= min_area

        if max_area is not None:
            mask &= areas < max_area

        new_target = {
            "boxes": boxes[mask],
        }

        if labels is not None:
            new_target["labels"] = labels[mask]

        filtered.append(new_target)

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

    For each image, keep the top-K predictions by score, then greedily
    match them to GT boxes using the same IoU threshold as AP.
    """

    total_gt = 0
    total_recalled = 0

    for pred, gt in zip(predictions, targets):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        # Keep only the highest-scoring detections.
        order = pred_scores.argsort(descending=True)[:max_dets]
        pred_boxes = pred_boxes[order]

        iou_matrix = _compute_iou_matrix(
            pred_boxes,
            gt_boxes,
        )

        matched_gt = set()

        for pred_idx in range(len(pred_boxes)):
            best_iou, best_gt = torch.max(
                iou_matrix[pred_idx],
                dim=0,
            )

            best_iou = float(best_iou.item())
            best_gt = int(best_gt.item())

            if best_iou >= iou_threshold and best_gt not in matched_gt:
                matched_gt.add(best_gt)
                total_recalled += 1

    return total_recalled / max(total_gt, 1)


# ============================================================
# Full metrics
# ============================================================


def compute_metrics(predictions, targets):
    """
    Compute the metrics reported by the paper.

    Returns:
        AP      : AP at IoU 0.5
        AP_M    : AP for medium GT boxes at IoU 0.5
        AP_L    : AP for large GT boxes at IoU 0.5
        AR@10   : recall using at most 10 detections per image
        AR_M    : recall for medium GT boxes
        AR_L    : recall for large GT boxes
    """

    # Main AP uses the paper's IoU > 0.5 evaluation setting;
    # the reference student implementation uses >= 0.5 in code.
    ap = _compute_ap(
        predictions,
        targets,
        iou_threshold=0.5,
    )["AP"]

    # COCO area ranges are used by the paper's experimental setup.
    medium_targets = _filter_targets_by_area(
        targets,
        min_area=MEDIUM_AREA_MIN,
        max_area=MEDIUM_AREA_MAX,
    )

    large_targets = _filter_targets_by_area(
        targets,
        min_area=LARGE_AREA_MIN,
        max_area=None,
    )

    ap_m = _compute_ap(
        predictions,
        medium_targets,
        iou_threshold=0.5,
    )["AP"]

    ap_l = _compute_ap(
        predictions,
        large_targets,
        iou_threshold=0.5,
    )["AP"]

    ar_10 = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    ar_m = _compute_ar(
        predictions,
        medium_targets,
        iou_threshold=0.5,
        max_dets=100,
    )

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
