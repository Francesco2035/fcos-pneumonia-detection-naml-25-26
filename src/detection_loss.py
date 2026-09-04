import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.detection import _utils as det_utils
from torchvision.ops import generalized_box_iou_loss


class DetectionLoss(nn.Module):
    """
    FCOS-like detection loss.

    Public interface intentionally unchanged.

    Predictions:
        {
            "P3": {
                "classification": Tensor[B, 1, H, W],  # logits
                "regression":     Tensor[B, 4, H, W],    # normalized LTRB
                "centerness":     Tensor[B, 1, H, W],    # logits
            },
            ...
            "P7": {...}
        }

    Targets:
        list[dict], one dict per image:
        [
            {
                "P3": {
                    "positive":   BoolTensor[H, W],
                    "ltrb":       FloatTensor[H, W, 4],
                    "centerness": FloatTensor[H, W],
                },
                ...
            }
        ]

    Losses:
        Classification:
            sigmoid focal loss over all locations.

        Regression:
            torchvision-style FCOS regression:
                prediction
                    -> ReLU
                    -> BoxLinearCoder.decode()
                    -> XYXY boxes
                    -> Generalized IoU loss

        Centerness:
            sigmoid focal loss over positive locations only.

        Final normalization:
            all losses are divided by the GLOBAL number of
            positive locations across all images and FPN levels.
    """

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

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        center_weight: float = 1.0,
        regression_weight: float = 1.0,
        centerness_weight: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                "alpha must be in [0, 1]."
            )

        if gamma < 0.0:
            raise ValueError(
                "gamma must be >= 0."
            )

        if (
            center_weight < 0.0
            or regression_weight < 0.0
            or centerness_weight < 0.0
        ):
            raise ValueError(
                "Loss weights must be non-negative."
            )

        if beta <= 0.0:
            raise ValueError(
                "SmoothL1 beta must be > 0."
            )

        self.alpha = alpha
        self.gamma = gamma

        self.center_weight = center_weight
        self.regression_weight = regression_weight
        self.centerness_weight = centerness_weight

        # Kept for API compatibility with the previous loss.
        # It is no longer used by the regression objective.
        self.beta = beta

        # Same coder used by torchvision FCOS.
        self.box_coder = det_utils.BoxLinearCoder(
            normalize_by_size=True
        )

    # =========================================================
    # Sigmoid focal loss
    # =========================================================

    def _focal_loss_sum(
        self,
        logits,
        targets,
    ):
        """
        Compute binary sigmoid focal loss over all locations.

        The per-location losses are summed here and normalized later
        using the global number of positive locations.
        """

        targets = targets.to(
            dtype=logits.dtype,
            device=logits.device,
        )

        # Binary cross-entropy for each location.
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # Convert logits to probabilities.
        probabilities = torch.sigmoid(logits)

        # Probability assigned to the correct class.
        p_t = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        # Balance positive and negative examples.
        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        # Down-weight easy examples and focus on hard ones.
        loss = (
            alpha_t
            * (1.0 - p_t).pow(self.gamma)
            * bce
        )

        # Sum per-location losses.
        # Global normalization is performed later.
        return loss.sum()

    # =========================================================
    # Graph-connected zero
    # =========================================================

    @staticmethod
    def _zero_loss(reference):
        """
        Return zero connected to the computation graph.
        """

        return reference.sum() * 0.0

    # =========================================================
    # FCOS locations / anchors
    # =========================================================

    @staticmethod
    def _build_fcos_anchors(
        height,
        width,
        stride,
        device,
        dtype,
    ):
        """
        Build a reference box for each location of an FPN feature map.

        Each feature-map location is mapped to the corresponding point
        in the original image using the FPN stride. A small box centered
        on that point is then created as the reference required by
        BoxLinearCoder for decoding LTRB predictions into XYXY boxes.
        """

        ys = torch.arange(
            height,
            device=device,
            dtype=dtype,
        )

        xs = torch.arange(
            width,
            device=device,
            dtype=dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        center_x = (
            grid_x + 0.5
        ) * float(stride)

        center_y = (
            grid_y + 0.5
        ) * float(stride)

        half_stride = (
            float(stride) / 2.0
        )

        x1 = center_x - half_stride
        y1 = center_y - half_stride
        x2 = center_x + half_stride
        y2 = center_y + half_stride

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

    # =========================================================
    # Decode TargetGenerator LTRB
    # =========================================================

    def _decode_target_ltrb(
        self,
        ltrb,
        positive,
        stride,
    ):
        """
        Decode the TargetGenerator's pixel-space LTRB targets
        into matched GT boxes in XYXY format.

        This is needed because TargetGenerator remains unchanged.
        """

        height, width = (
            positive.shape
        )

        anchors = (
            self._build_fcos_anchors(
                height,
                width,
                stride,
                ltrb.device,
                ltrb.dtype,
            )
        )

        positive_anchors = (
            anchors[positive]
        )

        positive_ltrb = (
            ltrb[positive]
        )

        center_x = (
            positive_anchors[:, 0]
            + positive_anchors[:, 2]
        ) * 0.5

        center_y = (
            positive_anchors[:, 1]
            + positive_anchors[:, 3]
        ) * 0.5

        x1 = (
            center_x
            - positive_ltrb[:, 0]
        )

        y1 = (
            center_y
            - positive_ltrb[:, 1]
        )

        x2 = (
            center_x
            + positive_ltrb[:, 2]
        )

        y2 = (
            center_y
            + positive_ltrb[:, 3]
        )

        return torch.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dim=-1,
        )

    # =========================================================
    # Regression loss
    # =========================================================

    def _giou_regression_loss(
        self,
        regression,
        ltrb,
        positive,
        stride,
    ):
        """
        Torchvision-style FCOS regression objective.

        Steps:

            raw regression output
                -> ReLU
                -> BoxLinearCoder.decode()
                -> predicted XYXY boxes
                -> Generalized IoU loss

        TargetGenerator remains unchanged and still returns
        pixel-space LTRB.
        """

        if not positive.any():
            return self._zero_loss(
                regression
            )

        # -----------------------------------------------------
        # Remove batch dimension
        # -----------------------------------------------------

        regression_pred = (
            regression[0]
            .permute(1, 2, 0)
        )

        height, width = (
            positive.shape
        )

        # -----------------------------------------------------
        # Build FCOS anchors
        # -----------------------------------------------------

        anchors = (
            self._build_fcos_anchors(
                height,
                width,
                stride,
                regression.device,
                regression.dtype,
            )
        )

        anchors_pos = (
            anchors[positive]
        )

        # -----------------------------------------------------
        # FCOS regression head convention:
        # non-negative distances.
        # -----------------------------------------------------

        regression_pred = F.relu(
            regression_pred
        )

        regression_pos = (
            regression_pred[
                positive
            ]
        )

        # -----------------------------------------------------
        # Decode normalized FCOS representation
        # -----------------------------------------------------

        predicted_boxes = (
            self.box_coder.decode(
                regression_pos,
                anchors_pos,
            )
        )

        # -----------------------------------------------------
        # Decode matched GT boxes from our TargetGenerator
        # -----------------------------------------------------

        target_boxes = (
            self._decode_target_ltrb(
                ltrb,
                positive,
                stride,
            )
        )

        # -----------------------------------------------------
        # Safety checks
        # -----------------------------------------------------

        if predicted_boxes.shape != target_boxes.shape:
            raise RuntimeError(
                "Predicted and target box shapes do not match: "
                f"{predicted_boxes.shape} vs "
                f"{target_boxes.shape}"
            )

        # -----------------------------------------------------
        # Generalized IoU loss
        # -----------------------------------------------------

        return generalized_box_iou_loss(
            predicted_boxes,
            target_boxes,
            reduction="sum",
        )

    # =========================================================
    # Raw losses for one image / one FPN level
    # =========================================================

    def _level_losses(
        self,
        predictions,
        level_targets,
        stride,
    ):
        """
        Compute raw losses for one image and one FPN level.

        No foreground normalization is performed here.
        """

        classification = predictions[
            "classification"
        ]

        regression = predictions[
            "regression"
        ]

        centerness = predictions[
            "centerness"
        ]

        positive = (
            level_targets[
                "positive"
            ]
            .bool()
            .to(
                device=classification.device
            )
        )

        ltrb = (
            level_targets[
                "ltrb"
            ]
            .to(
                device=classification.device,
                dtype=regression.dtype,
            )
        )

        centerness_target = (
            level_targets[
                "centerness"
            ]
            .to(
                device=classification.device,
                dtype=centerness.dtype,
            )
        )

        if classification.shape[0] != 1:
            raise ValueError(
                "DetectionLoss currently expects "
                "one image per target dictionary."
            )

        # -----------------------------------------------------
        # Remove batch / channel dimensions
        # -----------------------------------------------------

        center_logits = (
            classification[0, 0]
        )

        centerness_logits = (
            centerness[0, 0]
        )

        # -----------------------------------------------------
        # Classification
        # -----------------------------------------------------

        center_target = positive.to(
            dtype=center_logits.dtype
        )

        center_loss = (
            self._focal_loss_sum(
                center_logits,
                center_target,
            )
        )

        # -----------------------------------------------------
        # Positive count
        # -----------------------------------------------------

        num_positive = (
            positive.sum()
            .detach()
        )

        if positive.any():

            # ---------------------------------------------
            # Regression
            # ---------------------------------------------

            regression_loss = (
                self._giou_regression_loss(
                    regression=regression,
                    ltrb=ltrb,
                    positive=positive,
                    stride=stride,
                )
            )

            # ---------------------------------------------
            # Centerness
            # ---------------------------------------------

            center_logits_pos = (
                centerness_logits[
                    positive
                ]
            )

            center_target_pos = (
                centerness_target[
                    positive
                ]
            )

            centerness_loss = (
                self._focal_loss_sum(
                    center_logits_pos,
                    center_target_pos,
                )
            )

        else:

            regression_loss = (
                self._zero_loss(
                    regression
                )
            )

            centerness_loss = (
                self._zero_loss(
                    centerness
                )
            )

        # -----------------------------------------------------
        # Raw weighted total
        # -----------------------------------------------------

        total = (
            self.center_weight
            * center_loss
            +
            self.regression_weight
            * regression_loss
            +
            self.centerness_weight
            * centerness_loss
        )

        return {
            "center": center_loss,
            "regression": regression_loss,
            "centerness": centerness_loss,
            "total": total,
            "num_positive": num_positive,
        }

    # =========================================================
    # Full batch loss
    # =========================================================

    def forward(
        self,
        predictions,
        targets,
    ):
        """
        Public signature unchanged.

        Compute losses over the complete batch.
        """

        prediction_batch_size = (
            predictions[
                "P3"
            ][
                "classification"
            ].shape[0]
        )

        if len(targets) != prediction_batch_size:
            raise ValueError(
                "len(targets) must equal the "
                "prediction batch size."
            )

        batch_size = len(
            targets
        )

        # -----------------------------------------------------
        # Graph-connected zero
        # -----------------------------------------------------

        reference = (
            predictions[
                "P3"
            ][
                "classification"
            ]
        )

        zero = (
            reference.sum()
            * 0.0
        )

        # -----------------------------------------------------
        # Global raw losses
        # -----------------------------------------------------

        center_raw = zero.clone()
        regression_raw = zero.clone()
        centerness_raw = zero.clone()
        total_raw = zero.clone()

        # -----------------------------------------------------
        # Global foreground count
        # -----------------------------------------------------

        num_foreground = 0

        # -----------------------------------------------------
        # Per-level statistics
        # -----------------------------------------------------

        level_results = {
            level: []
            for level in self.LEVELS
        }

        # =====================================================
        # Iterate over images
        # =====================================================

        for b in range(
            batch_size
        ):

            for level in self.LEVELS:

                pred_level = (
                    predictions[level]
                )

                pred_image = {
                    "classification":
                        pred_level[
                            "classification"
                        ][
                            b:b + 1
                        ],

                    "regression":
                        pred_level[
                            "regression"
                        ][
                            b:b + 1
                        ],

                    "centerness":
                        pred_level[
                            "centerness"
                        ][
                            b:b + 1
                        ],
                }

                result = (
                    self._level_losses(
                        pred_image,
                        targets[b][level],
                        self.STRIDES[level],
                    )
                )

                level_results[
                    level
                ].append(
                    result
                )

                center_raw = (
                    center_raw
                    + result["center"]
                )

                regression_raw = (
                    regression_raw
                    + result["regression"]
                )

                centerness_raw = (
                    centerness_raw
                    + result["centerness"]
                )

                total_raw = (
                    total_raw
                    + result["total"]
                )

                num_foreground += int(
                    result[
                        "num_positive"
                    ].item()
                )

        # =====================================================
        # Global normalization
        # =====================================================

        normalizer = max(
            1,
            num_foreground,
        )

        normalizer_tensor = torch.tensor(
            normalizer,
            dtype=center_raw.dtype,
            device=center_raw.device,
        )

        center_total = (
            center_raw
            / normalizer_tensor
        )

        regression_total = (
            regression_raw
            / normalizer_tensor
        )

        centerness_total = (
            centerness_raw
            / normalizer_tensor
        )

        total = (
            total_raw
            / normalizer_tensor
        )

        # =====================================================
        # Normalize per-level statistics
        # =====================================================

        for level in self.LEVELS:

            for result in (
                level_results[level]
            ):

                result["center"] = (
                    result["center"]
                    / normalizer_tensor
                )

                result["regression"] = (
                    result["regression"]
                    / normalizer_tensor
                )

                result["centerness"] = (
                    result["centerness"]
                    / normalizer_tensor
                )

                result["total"] = (
                    result["total"]
                    / normalizer_tensor
                )

        # =====================================================
        # Return
        # =====================================================

        return {
            "total": total,
            "center": center_total,
            "regression": regression_total,
            "centerness": centerness_total,
            "levels": level_results,
        }