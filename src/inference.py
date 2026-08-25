import torch
import torch.nn as nn

from torchvision.models.detection import _utils as det_utils
from torchvision.ops import nms


class DetectionPostProcessor(nn.Module):
    """
    FCOS-like post-processing.

    Public interface unchanged.

    Regression:
        model output
            -> BoxLinearCoder.decode()
            -> XYXY boxes

    Score:
        sqrt(
            sigmoid(classification)
            *
            sigmoid(centerness)
        )

    This matches the torchvision FCOS post-processing
    convention found in the reference implementation.
    """

    def __init__(
        self,
        strides=(8, 16, 32, 64, 128),

        # -----------------------------------------------------
        # IMPORTANT:
        # Keep 0.0 for AP evaluation so that predictions are
        # not discarded before ranking.
        # -----------------------------------------------------
        score_threshold=0.1,

        nms_threshold=0.5,
    ):
        super().__init__()

        self.strides = strides

        self.score_th = score_threshold
        self.nms_th = nms_threshold

        self.levels = [
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
        ]

        # Same coder used by torchvision FCOS.
        self.box_coder = det_utils.BoxLinearCoder(
            normalize_by_size=True
        )

    # =========================================================
    # FCOS anchors
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
        Build one FCOS anchor per feature-map location.

        Anchor size = stride.

        Anchor center:
            ((x + 0.5) * stride,
             (y + 0.5) * stride)

        Output:
            [H, W, 4] in XYXY format.
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
    # Confidence score
    # =========================================================

    def _confidence_score(
        self,
        classification,
        centerness,
    ):
        """
        FCOS score:

            sqrt(
                sigmoid(classification)
                * sigmoid(centerness)
            )

        This matches the torchvision FCOS implementation.
        """

        classification_score = torch.sigmoid(
            classification
        )

        centerness_score = torch.sigmoid(
            centerness
        )

        score = torch.sqrt(
            classification_score
            * centerness_score
        )

        return score

    # =========================================================
    # Process one FPN level
    # =========================================================

    def _process_level(
        self,
        classification,
        regression,
        centerness,
        stride,
    ):
        """
        Process one FPN level for one image.

        Inputs:
            classification -> [1, 1, H, W]
            regression     -> [1, 4, H, W]
            centerness     -> [1, 1, H, W]

        Outputs:
            boxes  -> [N, 4]
            scores -> [N]
        """

        # -----------------------------------------------------
        # Remove batch/channel dimensions
        # -----------------------------------------------------

        classification = (
            classification[0, 0]
        )

        centerness = (
            centerness[0, 0]
        )

        regression = (
            regression[0]
        )

        H, W = classification.shape

        # -----------------------------------------------------
        # Confidence score
        # -----------------------------------------------------

        scores = self._confidence_score(
            classification,
            centerness,
        )

        # -----------------------------------------------------
        # Threshold
        # -----------------------------------------------------

        keep = (
            scores >= self.score_th
        )

        if not keep.any():

            device = regression.device

            return (
                torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=device,
                ),
            )

        # -----------------------------------------------------
        # FCOS anchors / locations
        # -----------------------------------------------------

        anchors = (
            self._build_fcos_anchors(
                H,
                W,
                stride,
                regression.device,
                regression.dtype,
            )
        )

        # -----------------------------------------------------
        # [4, H, W] -> [H, W, 4]
        # -----------------------------------------------------

        regression = (
            regression
            .permute(1, 2, 0)
        )

        # -----------------------------------------------------
        # Keep selected locations
        # -----------------------------------------------------

        selected_regression = (
            regression[keep]
        )

        selected_anchors = (
            anchors[keep]
        )

        selected_scores = (
            scores[keep]
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # The regression head is trained as a non-negative
        # FCOS distance representation.
        #
        # The training loss applies ReLU before BoxLinearCoder
        # decoding, so inference must do the same.
        # -----------------------------------------------------

        selected_regression = torch.relu(
            selected_regression
        )

        # -----------------------------------------------------
        # Decode normalized FCOS regression into XYXY boxes
        # -----------------------------------------------------

        boxes = self.box_coder.decode(
            selected_regression,
            selected_anchors,
        )

        return (
            boxes,
            selected_scores,
        )

    # =========================================================
    # Full post-processing
    # =========================================================

    def forward(
        self,
        predictions,
    ):
        """
        Public signature unchanged.

        Args:
            predictions:
                output dictionary from DetectionFramework.

        Returns:
            list of detections, one per image.

            Each entry:
                boxes  -> [N, 4]
                scores -> [N]
                labels -> [N]
        """

        batch_size = (
            predictions[
                "P3"
            ][
                "classification"
            ].shape[0]
        )

        outputs = []

        # =====================================================
        # Process each image
        # =====================================================

        for b in range(
            batch_size
        ):

            all_boxes = []
            all_scores = []

            # -------------------------------------------------
            # Process P3 ... P7
            # -------------------------------------------------

            for level, stride in zip(
                self.levels,
                self.strides,
            ):

                pred_level = (
                    predictions[level]
                )

                classification = (
                    pred_level[
                        "classification"
                    ][
                        b:b + 1
                    ]
                )

                regression = (
                    pred_level[
                        "regression"
                    ][
                        b:b + 1
                    ]
                )

                centerness = (
                    pred_level[
                        "centerness"
                    ][
                        b:b + 1
                    ]
                )

                boxes, scores = (
                    self._process_level(
                        classification,
                        regression,
                        centerness,
                        stride,
                    )
                )

                all_boxes.append(
                    boxes
                )

                all_scores.append(
                    scores
                )

            # -------------------------------------------------
            # Concatenate pyramid levels
            # -------------------------------------------------

            if len(all_boxes) == 0:

                device = (
                    predictions[
                        "P3"
                    ][
                        "classification"
                    ].device
                )

                final_boxes = torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                    device=device,
                )

                final_scores = torch.empty(
                    (0,),
                    dtype=torch.float32,
                    device=device,
                )

            else:

                final_boxes = torch.cat(
                    all_boxes,
                    dim=0,
                )

                final_scores = torch.cat(
                    all_scores,
                    dim=0,
                )

            # -------------------------------------------------
            # NMS
            # -------------------------------------------------

            if final_boxes.shape[0] > 0:

                keep = nms(
                    final_boxes,
                    final_scores,
                    self.nms_th,
                )

                final_boxes = (
                    final_boxes[keep]
                )

                final_scores = (
                    final_scores[keep]
                )

            # -------------------------------------------------
            # Single-class detector
            # -------------------------------------------------

            final_labels = torch.ones(
                final_boxes.shape[0],
                dtype=torch.long,
                device=final_boxes.device,
            )

            outputs.append(
                {
                    "boxes": final_boxes,
                    "scores": final_scores,
                    "labels": final_labels,
                }
            )

        return outputs

    # =========================================================
    # Callable interface
    # =========================================================

    def __call__(
        self,
        predictions,
    ):
        return self.forward(
            predictions
        )