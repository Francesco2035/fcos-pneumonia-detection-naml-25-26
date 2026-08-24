import torch

from src.metrics import (
    MEDIUM_AREA_MIN,
    MEDIUM_AREA_MAX,
    LARGE_AREA_MIN,
    _compute_ap,
    _compute_ar,
    _compute_box_areas,
    _compute_iou_matrix,
    _filter_targets_by_area,
    _match_predictions,
    compute_metrics,
)


# ============================================================
# IoU tests
# ============================================================


def test_identical_boxes():
    boxes1 = torch.tensor([[0., 0., 10., 10.]])
    boxes2 = torch.tensor([[0., 0., 10., 10.]])

    iou = _compute_iou_matrix(boxes1, boxes2)

    assert iou.shape == (1, 1)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))


def test_no_overlap():
    boxes1 = torch.tensor([[0., 0., 10., 10.]])
    boxes2 = torch.tensor([[20., 20., 30., 30.]])

    iou = _compute_iou_matrix(boxes1, boxes2)

    assert torch.isclose(iou[0, 0], torch.tensor(0.0))


def test_partial_overlap():
    boxes1 = torch.tensor([[0., 0., 10., 10.]])
    boxes2 = torch.tensor([[5., 5., 15., 15.]])

    iou = _compute_iou_matrix(boxes1, boxes2)
    expected = 25.0 / 175.0

    assert torch.isclose(iou[0, 0], torch.tensor(expected))


def test_multiple_boxes():
    boxes1 = torch.tensor([
        [0., 0., 10., 10.],
        [20., 20., 30., 30.],
    ])

    boxes2 = torch.tensor([
        [0., 0., 10., 10.],
        [5., 5., 15., 15.],
        [100., 100., 120., 120.],
    ])

    iou = _compute_iou_matrix(boxes1, boxes2)

    assert iou.shape == (2, 3)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))
    assert torch.isclose(iou[0, 2], torch.tensor(0.0))


def test_iou_empty_inputs():
    boxes1 = torch.empty((0, 4))
    boxes2 = torch.tensor([[0., 0., 10., 10.]])

    iou = _compute_iou_matrix(boxes1, boxes2)

    assert iou.shape == (0, 1)


# ============================================================
# Matching tests
# ============================================================


def test_matching_true_positive():
    pred = {
        "boxes": torch.tensor([[0., 0., 10., 10.]]),
        "scores": torch.tensor([0.9]),
    }

    gt = {
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }

    result = _match_predictions(pred, gt)

    assert torch.allclose(result["scores"], torch.tensor([0.9]))
    assert result["tp"].tolist() == [1]
    assert result["fp"].tolist() == [0]
    assert result["matched_gt"] == {0}
    assert result["num_gt"] == 1


def test_matching_low_iou_is_false_positive():
    pred = {
        "boxes": torch.tensor([[0., 0., 10., 10.]]),
        "scores": torch.tensor([0.9]),
    }

    gt = {
        "boxes": torch.tensor([[20., 20., 30., 30.]])
    }

    result = _match_predictions(pred, gt)

    assert result["tp"].tolist() == [0]
    assert result["fp"].tolist() == [1]
    assert result["matched_gt"] == set()


def test_matching_iou_at_threshold_is_true_positive():
    """Current implementation follows the students' >= 0.5 convention."""

    # Two 10x10 boxes with an overlap producing IoU = 25/175 < 0.5
    # are not enough, so use identical boxes to keep this test explicit.
    pred = {
        "boxes": torch.tensor([[0., 0., 10., 10.]]),
        "scores": torch.tensor([0.9]),
    }

    gt = {
        "boxes": torch.tensor([[0., 0., 10., 20.]]),
    }

    result = _match_predictions(pred, gt, iou_threshold=1.0 / 2.0)

    # IoU = 100 / 200 = 0.5
    assert result["tp"].tolist() == [1]
    assert result["fp"].tolist() == [0]


def test_matching_duplicate_predictions():
    pred = {
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
            [1., 1., 9., 9.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    gt = {
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }

    result = _match_predictions(pred, gt)

    assert torch.allclose(result["scores"], torch.tensor([0.9, 0.8]))
    assert result["tp"].tolist() == [1, 0]
    assert result["fp"].tolist() == [0, 1]
    assert result["matched_gt"] == {0}


def test_matching_multiple_ground_truths():
    pred = {
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
            [20., 20., 30., 30.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    gt = {
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
            [20., 20., 30., 30.],
        ])
    }

    result = _match_predictions(pred, gt)

    assert result["tp"].tolist() == [1, 1]
    assert result["fp"].tolist() == [0, 0]
    assert result["matched_gt"] == {0, 1}


def test_matching_no_predictions():
    pred = {
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
    }

    gt = {
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }

    result = _match_predictions(pred, gt)

    assert result["scores"].numel() == 0
    assert result["tp"].numel() == 0
    assert result["fp"].numel() == 0
    assert result["matched_gt"] == set()
    assert result["num_gt"] == 1


def test_matching_no_ground_truths():
    pred = {
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
            [20., 20., 30., 30.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }

    gt = {
        "boxes": torch.empty((0, 4))
    }

    result = _match_predictions(pred, gt)

    assert result["tp"].tolist() == [0, 0]
    assert result["fp"].tolist() == [1, 1]
    assert result["matched_gt"] == set()
    assert result["num_gt"] == 0


# ============================================================
# Area helper tests
# ============================================================


def test_box_areas():
    boxes = torch.tensor([
        [0., 0., 10., 10.],
        [0., 0., 20., 5.],
    ])

    areas = _compute_box_areas(boxes)

    assert torch.equal(
        areas,
        torch.tensor([100., 100.]),
    )


def test_filter_targets_by_medium_area():
    targets = [{
        "boxes": torch.tensor([
            [0., 0., 32., 32.],       # 1024 -> medium
            [0., 0., 96., 96.],       # 9216 -> large boundary
            [0., 0., 20., 20.],       # 400 -> small
        ]),
    }]

    filtered = _filter_targets_by_area(
        targets,
        min_area=MEDIUM_AREA_MIN,
        max_area=MEDIUM_AREA_MAX,
    )

    assert len(filtered[0]["boxes"]) == 1
    assert torch.equal(
        filtered[0]["boxes"][0],
        torch.tensor([0., 0., 32., 32.]),
    )


def test_filter_targets_by_large_area():
    targets = [{
        "boxes": torch.tensor([
            [0., 0., 96., 96.],
            [0., 0., 100., 100.],
        ]),
    }]

    filtered = _filter_targets_by_area(
        targets,
        min_area=LARGE_AREA_MIN,
        max_area=None,
    )

    assert len(filtered[0]["boxes"]) == 2


# ============================================================
# AP tests
# ============================================================


def test_ap_perfect_detection_is_one():
    predictions = [{
        "boxes": torch.tensor([[0., 0., 10., 10.]]),
        "scores": torch.tensor([0.9]),
    }]

    targets = [{
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }]

    result = _compute_ap(predictions, targets)

    assert result["num_gt"] == 1
    assert result["num_pred"] == 1
    assert abs(result["AP"] - 1.0) < 1e-6


def test_ap_false_positive_before_true_positive_is_half():
    predictions = [{
        "boxes": torch.tensor([
            [20., 20., 30., 30.],
            [0., 0., 10., 10.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }]

    targets = [{
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }]

    result = _compute_ap(predictions, targets)

    assert result["num_gt"] == 1
    assert result["num_pred"] == 2
    assert abs(result["AP"] - 0.5) < 1e-6


def test_ap_is_global_over_multiple_images():
    predictions = [
        {
            "boxes": torch.tensor([[0., 0., 10., 10.]]),
            "scores": torch.tensor([0.9]),
        },
        {
            "boxes": torch.tensor([[20., 20., 30., 30.]]),
            "scores": torch.tensor([0.8]),
        },
    ]

    targets = [
        {
            "boxes": torch.tensor([[0., 0., 10., 10.]])
        },
        {
            "boxes": torch.tensor([[20., 20., 30., 30.]])
        },
    ]

    result = _compute_ap(predictions, targets)

    assert result["num_gt"] == 2
    assert result["num_pred"] == 2
    assert abs(result["AP"] - 1.0) < 1e-6


def test_ap_no_predictions_is_zero():
    predictions = [{
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
    }]

    targets = [{
        "boxes": torch.tensor([[0., 0., 10., 10.]])
    }]

    result = _compute_ap(predictions, targets)

    assert result["AP"] == 0.0
    assert result["num_gt"] == 1
    assert result["num_pred"] == 0


def test_ap_no_ground_truth_is_zero():
    predictions = [{
        "boxes": torch.tensor([[0., 0., 10., 10.]]),
        "scores": torch.tensor([0.9]),
    }]

    targets = [{
        "boxes": torch.empty((0, 4))
    }]

    result = _compute_ap(predictions, targets)

    assert result["AP"] == 0.0
    assert result["num_gt"] == 0
    assert result["num_pred"] == 1


# ============================================================
# AR tests
# ============================================================


def test_ar_perfect_detection_is_one():
    predictions = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
        ]),
        "scores": torch.tensor([0.9]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
        ])
    }]

    ar = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    assert abs(ar - 1.0) < 1e-6


def test_ar_no_predictions_is_zero():
    predictions = [{
        "boxes": torch.empty((0, 4)),
        "scores": torch.empty((0,)),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
        ])
    }]

    ar = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    assert ar == 0.0


def test_ar_top_k_limits_number_of_predictions():
    predictions = [{
        "boxes": torch.tensor([
            [20., 20., 30., 30.],  # high score FP
            [0., 0., 10., 10.],     # lower score TP
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
        ])
    }]

    ar_top1 = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=1,
    )

    ar_top2 = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=2,
    )

    assert ar_top1 == 0.0
    assert abs(ar_top2 - 1.0) < 1e-6


def test_ar_duplicate_predictions_count_gt_once():
    predictions = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
            [1., 1., 9., 9.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 10., 10.],
        ])
    }]

    ar = _compute_ar(
        predictions,
        targets,
        iou_threshold=0.5,
        max_dets=10,
    )

    assert abs(ar - 1.0) < 1e-6


# ============================================================
# Full metrics tests
# ============================================================


def test_compute_metrics_returns_paper_metrics():
    predictions = [{
        "boxes": torch.tensor([
            [0., 0., 32., 32.],
            [0., 0., 100., 100.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 32., 32.],
            [0., 0., 100., 100.],
        ])
    }]

    result = compute_metrics(
        predictions,
        targets,
    )

    expected_keys = {
        "AP",
        "AP_M",
        "AP_L",
        "AR@10",
        "AR_M",
        "AR_L",
    }

    assert set(result.keys()) == expected_keys

    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_compute_metrics_perfect_medium_and_large():
    predictions = [{
        "boxes": torch.tensor([
            [0., 0., 32., 32.],
            [0., 0., 100., 100.],
        ]),
        "scores": torch.tensor([0.9, 0.8]),
    }]

    targets = [{
        "boxes": torch.tensor([
            [0., 0., 32., 32.],
            [0., 0., 100., 100.],
        ])
    }]

    result = compute_metrics(
        predictions,
        targets,
    )

    assert abs(result["AP"] - 1.0) < 1e-6
    assert abs(result["AP_M"] - 1.0) < 1e-6
    # The area-filtered evaluation keeps all predictions, so the medium
    # prediction is a false positive for the large-only evaluation.
    assert abs(result["AP_L"] - 0.5) < 1e-6
    assert abs(result["AR@10"] - 1.0) < 1e-6
    assert abs(result["AR_M"] - 1.0) < 1e-6
    assert abs(result["AR_L"] - 1.0) < 1e-6
