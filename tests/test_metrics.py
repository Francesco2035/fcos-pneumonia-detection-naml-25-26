import torch
import random


# ============================================================
# VERSIONE ORIGINALE
# ============================================================

def old_iou_matrix(boxes1, boxes2):
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

            x_left = torch.maximum(x11, x21)
            y_top = torch.maximum(y11, y21)
            x_right = torch.minimum(x12, x22)
            y_bottom = torch.minimum(y12, y22)

            intersection_width = torch.clamp(
                x_right - x_left,
                min=0.0,
            )

            intersection_height = torch.clamp(
                y_bottom - y_top,
                min=0.0,
            )

            intersection_area = (
                intersection_width
                * intersection_height
            )

            area1 = (
                torch.clamp(
                    x12 - x11,
                    min=0.0,
                )
                *
                torch.clamp(
                    y12 - y11,
                    min=0.0,
                )
            )

            area2 = (
                torch.clamp(
                    x22 - x21,
                    min=0.0,
                )
                *
                torch.clamp(
                    y22 - y21,
                    min=0.0,
                )
            )

            union_area = (
                area1
                + area2
                - intersection_area
            )

            iou = (
                intersection_area
                / (union_area + 1e-6)
            )

            iou_matrix[i, j] = iou

    return iou_matrix


def old_match(pred, gt, threshold=0.5):

    boxes_pred = pred["boxes"]
    boxes_gt = gt["boxes"]
    scores = pred["scores"]

    order = scores.argsort(descending=True)

    sorted_scores = scores[order]

    matched_gt = set()

    tp = []
    fp = []

    if len(boxes_pred) == 0:
        return sorted_scores, tp, fp, matched_gt

    if len(boxes_gt) == 0:
        return (
            sorted_scores,
            [0] * len(boxes_pred),
            [1] * len(boxes_pred),
            matched_gt,
        )

    iou_matrix = old_iou_matrix(
        boxes_pred,
        boxes_gt,
    )

    for pred_idx in order:

        prediction_ious = iou_matrix[pred_idx]

        best_iou, best_gt = torch.max(
            prediction_ious,
            dim=0,
        )

        best_iou = float(best_iou)
        best_gt = int(best_gt)

        if best_iou < threshold:
            tp.append(0)
            fp.append(1)
            continue

        if best_gt in matched_gt:
            tp.append(0)
            fp.append(1)
            continue

        tp.append(1)
        fp.append(0)
        matched_gt.add(best_gt)

    return (
        sorted_scores,
        tp,
        fp,
        matched_gt,
    )


# ============================================================
# VERSIONE VETTORIZZATA
# ============================================================

def new_iou_matrix(boxes1, boxes2):

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros(
            (N, M),
            dtype=torch.float32,
            device=boxes1.device,
        )

    boxes1 = boxes1.float()
    boxes2 = boxes2.float()

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = torch.clamp(
        bottom_right - top_left,
        min=0.0,
    )

    intersection = (
        wh[..., 0]
        * wh[..., 1]
    )

    area1 = (
        torch.clamp(
            boxes1[:, 2] - boxes1[:, 0],
            min=0.0,
        )
        *
        torch.clamp(
            boxes1[:, 3] - boxes1[:, 1],
            min=0.0,
        )
    )

    area2 = (
        torch.clamp(
            boxes2[:, 2] - boxes2[:, 0],
            min=0.0,
        )
        *
        torch.clamp(
            boxes2[:, 3] - boxes2[:, 1],
            min=0.0,
        )
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / (
        union + 1e-6
    )


def new_match(pred, gt, threshold=0.5):

    boxes_pred = pred["boxes"]
    boxes_gt = gt["boxes"]
    scores = pred["scores"]

    order = scores.argsort(descending=True)

    sorted_boxes = boxes_pred[order]
    sorted_scores = scores[order]

    n_pred = len(sorted_boxes)
    n_gt = len(boxes_gt)

    if n_pred == 0:
        return (
            sorted_scores,
            [],
            [],
            set(),
        )

    if n_gt == 0:
        return (
            sorted_scores,
            [0] * n_pred,
            [1] * n_pred,
            set(),
        )

    iou_matrix = new_iou_matrix(
        sorted_boxes,
        boxes_gt,
    )

    best_iou, best_gt = torch.max(
        iou_matrix,
        dim=1,
    )

    valid = best_iou >= threshold

    positions = torch.arange(
        n_pred,
        device=scores.device,
    )

    valid_positions = positions[valid]
    valid_gt = best_gt[valid]

    # Prima prediction valida per ogni GT.
    first_match = torch.full(
        (n_gt,),
        n_pred,
        dtype=torch.long,
        device=scores.device,
    )

    first_match.scatter_reduce_(
        0,
        valid_gt,
        valid_positions,
        reduce="amin",
        include_self=True,
    )

    tp_mask = (
        valid
        &
        (
            positions
            == first_match[best_gt]
        )
    )

    tp = tp_mask.int().tolist()
    fp = (1 - tp_mask.int()).tolist()

    matched_gt = set(
        best_gt[tp_mask]
        .cpu()
        .tolist()
    )

    return (
        sorted_scores,
        tp,
        fp,
        matched_gt,
    )


# ============================================================
# TEST
# ============================================================

def generate_random_boxes(n):
    """
    Generate valid XYXY boxes in a 512x512 image.
    """

    x1 = torch.rand(n) * 400
    y1 = torch.rand(n) * 400

    x2 = x1 + torch.rand(n) * (512 - x1)
    y2 = y1 + torch.rand(n) * (512 - y1)

    return torch.stack(
        [x1, y1, x2, y2],
        dim=1,
    )


print("=" * 70)
print("TESTING OLD VS VECTORIZED IMPLEMENTATION")
print("=" * 70)

torch.manual_seed(42)
random.seed(42)


# ------------------------------------------------------------
# 1. IoU comparison
# ------------------------------------------------------------

print("\n[1] IoU MATRIX")

for case in range(100):

    n_pred = random.choice(
        [0, 1, 2, 5, 10, 20]
    )

    n_gt = random.choice(
        [0, 1, 2, 5, 10]
    )

    pred_boxes = generate_random_boxes(
        n_pred
    )

    gt_boxes = generate_random_boxes(
        n_gt
    )

    old_iou = old_iou_matrix(
        pred_boxes,
        gt_boxes,
    )

    new_iou = new_iou_matrix(
        pred_boxes,
        gt_boxes,
    )

    if not torch.allclose(
        old_iou,
        new_iou,
        atol=1e-6,
        rtol=1e-6,
    ):
        print(
            f"❌ IoU mismatch in case {case}"
        )

        print(
            "Max difference:",
            (
                old_iou - new_iou
            ).abs().max().item(),
        )

        raise RuntimeError(
            "IoU implementations differ!"
        )

print(
    "✅ IoU: 100/100 cases identical"
)


# ------------------------------------------------------------
# 2. Matching comparison
# ------------------------------------------------------------

print("\n[2] MATCHING")

for case in range(100):

    n_pred = random.choice(
        [0, 1, 2, 5, 10, 20]
    )

    n_gt = random.choice(
        [0, 1, 2, 5, 10]
    )

    pred = {
        "boxes": generate_random_boxes(
            n_pred
        ),
        "scores": torch.rand(
            n_pred
        ),
    }

    gt = {
        "boxes": generate_random_boxes(
            n_gt
        ),
    }

    for threshold in [
        0.0,
        0.3,
        0.5,
        0.7,
        1.0,
    ]:

        old_result = old_match(
            pred,
            gt,
            threshold,
        )

        new_result = new_match(
            pred,
            gt,
            threshold,
        )

        old_scores, old_tp, old_fp, old_gt = (
            old_result
        )

        new_scores, new_tp, new_fp, new_gt = (
            new_result
        )

        if not torch.allclose(
            old_scores,
            new_scores,
            atol=1e-7,
            rtol=1e-7,
        ):
            raise RuntimeError(
                f"Score mismatch in case {case}"
            )

        if old_tp != new_tp:
            raise RuntimeError(
                f"TP mismatch in case {case}, "
                f"threshold={threshold}\n"
                f"old={old_tp}\n"
                f"new={new_tp}"
            )

        if old_fp != new_fp:
            raise RuntimeError(
                f"FP mismatch in case {case}, "
                f"threshold={threshold}\n"
                f"old={old_fp}\n"
                f"new={new_fp}"
            )

        if old_gt != new_gt:
            raise RuntimeError(
                f"Matched GT mismatch in case {case}, "
                f"threshold={threshold}\n"
                f"old={old_gt}\n"
                f"new={new_gt}"
            )

print(
    "✅ Matching: 100/100 cases identical "
    "for all tested thresholds"
)


# ------------------------------------------------------------
# 3. Explicit duplicate-GT case
# ------------------------------------------------------------

print("\n[3] DUPLICATE MATCH CASE")

gt_box = torch.tensor(
    [[100.0, 100.0, 200.0, 200.0]]
)

pred = {
    "boxes": torch.tensor(
        [
            [100.0, 100.0, 200.0, 200.0],
            [100.0, 100.0, 200.0, 200.0],
            [100.0, 100.0, 200.0, 200.0],
        ]
    ),
    "scores": torch.tensor(
        [0.9, 0.8, 0.7]
    ),
}

gt = {
    "boxes": gt_box
}

old_result = old_match(
    pred,
    gt,
    0.5,
)

new_result = new_match(
    pred,
    gt,
    0.5,
)

print(
    "Old TP:",
    old_result[1]
)

print(
    "New TP:",
    new_result[1]
)

print(
    "Old FP:",
    old_result[2]
)

print(
    "New FP:",
    new_result[2]
)

assert old_result[1] == new_result[1]
assert old_result[2] == new_result[2]

print(
    "✅ Duplicate matching identical"
)


print("\n" + "=" * 70)
print("ALL TESTS PASSED ✅")
print("=" * 70)
print(
    "The vectorized IoU and matching produce the same "
    "outputs as the original implementation on all tests."
)