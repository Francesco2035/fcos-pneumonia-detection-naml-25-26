import torch

def compute_box_area(box):
    """
    Compute the area of a bounding box in XYXY format.
    """

    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection coordinates.
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1,
    )

    intersection_height = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    # Union = area A + area B - intersection.
    union = (
        compute_box_area(box_a)
        + compute_box_area(box_b)
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


def compute_overlap_over_smaller_area(
    box_a,
    box_b,
):
    """
    Compute intersection area relative to the smaller box area.

    This measure is used only for visualization filtering.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection coordinates.
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1,
    )

    intersection_height = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    smaller_area = min(
        compute_box_area(box_a),
        compute_box_area(box_b),
    )

    if smaller_area <= 0.0:
        return 0.0

    return intersection / smaller_area


def best_iou_against_gt(
    prediction_box,
    gt_boxes,
):
    """
    Return the highest IoU between one prediction and all GT boxes.
    """

    if not gt_boxes:
        return 0.0

    return max(
        compute_iou(
            prediction_box,
            gt_box,
        )
        for gt_box in gt_boxes
    )


def suppress_redundant_predictions(
    boxes,
    scores,
    overlap_threshold,
    max_detections,
):
    """
    Remove highly overlapping predictions for visualization only.

    The larger-area box is kept when two predictions overlap strongly.
    Score is used only as a tie-breaker.

    This function does not affect official AP/AR computation.
    """

    if boxes.numel() == 0:
        return boxes, scores

    boxes_list = (
        boxes
        .detach()
        .cpu()
        .tolist()
    )

    scores_list = (
        scores
        .detach()
        .cpu()
        .tolist()
    )

    candidates = []

    for box, score in zip(
        boxes_list,
        scores_list,
    ):
        candidates.append(
            {
                "box": box,
                "score": float(score),
                "area": compute_box_area(box),
            }
        )

    kept = []

    while candidates:

        # Prefer larger boxes.
        # Score is used only to break equal-area ties.
        candidates.sort(
            key=lambda item: (
                item["area"],
                item["score"],
            ),
            reverse=True,
        )

        current = candidates.pop(0)

        kept.append(current)

        if len(kept) >= max_detections:
            break

        remaining = []

        for candidate in candidates:

            overlap = compute_overlap_over_smaller_area(
                current["box"],
                candidate["box"],
            )

            if overlap < overlap_threshold:
                remaining.append(candidate)

        candidates = remaining

    kept_boxes = torch.tensor(
        [
            item["box"]
            for item in kept
        ],
        dtype=boxes.dtype,
        device=boxes.device,
    )

    kept_scores = torch.tensor(
        [
            item["score"]
            for item in kept
        ],
        dtype=scores.dtype,
        device=scores.device,
    )

    return kept_boxes, kept_scores


def match_predictions_to_ground_truth(
    gt_boxes,
    pred_boxes,
    iou_threshold=0.50,
):
    """
    Match predictions to ground-truth boxes using greedy IoU matching.

    Each prediction and each ground-truth box can be matched at most once.

    Returns:
        tp:
            Number of matched predictions.
        fp:
            Number of unmatched predictions.
        fn:
            Number of unmatched ground-truth boxes.
        matched_ious:
            IoU values of all successful matches.
    """

    gt_boxes = [
        list(box)
        for box in gt_boxes
    ]

    pred_boxes = [
        list(box)
        for box in pred_boxes
    ]

    if not gt_boxes:
        return (
            0,
            len(pred_boxes),
            0,
            [],
        )

    if not pred_boxes:
        return (
            0,
            0,
            len(gt_boxes),
            [],
        )

    matches = []

    # Compute every prediction/GT IoU pair.
    for pred_index, pred_box in enumerate(
        pred_boxes
    ):
        for gt_index, gt_box in enumerate(
            gt_boxes
        ):
            iou = compute_iou(
                pred_box,
                gt_box,
            )

            matches.append(
                (
                    iou,
                    pred_index,
                    gt_index,
                )
            )

    # Process the strongest matches first.
    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched_predictions = set()
    matched_gt = set()
    matched_ious = []

    for (
        iou,
        pred_index,
        gt_index,
    ) in matches:

        if iou < iou_threshold:
            break

        # Prediction already matched.
        if pred_index in matched_predictions:
            continue

        # Ground-truth already matched.
        if gt_index in matched_gt:
            continue

        matched_predictions.add(
            pred_index
        )

        matched_gt.add(
            gt_index
        )

        matched_ious.append(
            float(iou)
        )

    tp = len(
        matched_predictions
    )

    fp = (
        len(pred_boxes)
        - tp
    )

    fn = (
        len(gt_boxes)
        - tp
    )

    return (
        tp,
        fp,
        fn,
        matched_ious,
    )