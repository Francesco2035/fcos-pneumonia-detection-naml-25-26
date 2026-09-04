from pathlib import Path

import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
    VAL_RATIO,
    SEED,
    NMS_THRESHOLD,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_test_transforms,
)

from src.datasets.split import (
    create_train_val_split,
)

from src.models.detector import (
    DetectionFramework,
)

from src.inference import (
    DetectionPostProcessor,
)

from src.calibration import (
    DetectionThresholdCalibrator,
)

from src.analysis.geometry import (
    suppress_redundant_predictions,
    match_predictions_to_ground_truth,
)

from src.metrics import (
    compute_ap_curve,
    compute_metrics as compute_detection_metrics,
)

from src.analysis.io import (
    save_precision_recall_curve,
)


# ============================================================
# Configuration
# ============================================================

IOU_THRESHOLD = 0.50
OFFICIAL_SCORE_THRESHOLD = 0.10


class DetectionAnalyzer:
    """
    Runs quantitative analysis of a trained detector.

    The analyzer loads the detector checkpoint, builds the
    corresponding ResNet/FPN/head architecture, calibrates the
    detection threshold, performs validation inference, matches
    predictions to ground-truth boxes, and computes image- and
    box-level statistics.

    Visualization and file export are handled separately.
    """

    def __init__(
        self,
        checkpoint_path,
        backbone,
        resnet_depth,
        device,
        output_dir,
        max_detections=10,
        overlap_threshold=0.40,
        manual_threshold=None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.backbone_type = str(
            backbone
        )

        self.resnet_depth = int(
            resnet_depth
        )

        self.device = device

        self.output_dir = Path(
            output_dir
        )

        self.max_detections = int(
            max_detections
        )

        self.overlap_threshold = float(
            overlap_threshold
        )

        self.manual_threshold = (
            manual_threshold
        )

        self.tau = None

        self.visualization_threshold = (
            None
        )

        # --------------------------------------------------------
        # Validate configuration
        # --------------------------------------------------------

        if self.backbone_type not in (
            "imagenet",
            "chest_xray",
        ):
            raise ValueError(
                "Unsupported backbone: "
                f"{self.backbone_type}"
            )

        if self.resnet_depth not in (
            50,
            101,
        ):
            raise ValueError(
                "Unsupported ResNet depth: "
                f"{self.resnet_depth}. "
                "Supported values: 50, 101."
            )

        # --------------------------------------------------------
        # Output directories
        # --------------------------------------------------------

        (
            self.output_dir
            / "metrics"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        self.dataset = (
            RSNAPneumoniaDataset(
                dcm_path=TRAIN_DCM_PATH,
                csv_path=CSV_PATH,
                transform=get_test_transforms(
                    IMAGE_SIZE
                ),
            )
        )

        (
            _,
            self.val_indices,
        ) = create_train_val_split(
            self.dataset,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )

        self.val_loader = (
            self.dataset.get_dataloader(
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=VAL_NUM_WORKERS,
                indices=self.val_indices,
            )
        )

        print(
            f"[LOG] Validation images: "
            f"{len(self.val_indices)}"
        )

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------
        #
        # The complete detector checkpoint is loaded immediately
        # afterwards, therefore no external Chest-Xray checkpoint
        # is needed here.
        #
        # path_model=None only determines how the initial ResNet
        # structure is constructed. Its weights are overwritten
        # by the detector checkpoint below.

        print(
            "[LOG] Building detector:"
        )

        print(
            f"      backbone={self.backbone_type}"
        )

        print(
            f"      ResNet-{self.resnet_depth}"
        )

        self.model = (
            DetectionFramework(
                path_model=None,
                resnet_depth=self.resnet_depth,
            )
            .to(self.device)
        )

        # --------------------------------------------------------
        # Detector checkpoint
        # --------------------------------------------------------

        print(
            "[LOG] Loading detector checkpoint:"
        )

        print(
            f"      {self.checkpoint_path}"
        )

        if not self.checkpoint_path.is_file():

            raise FileNotFoundError(
                "Detector checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                "Invalid detector checkpoint: "
                "expected a dictionary."
            )

        if (
            "model_state_dict"
            not in checkpoint
        ):
            raise RuntimeError(
                "Invalid detector checkpoint: "
                "'model_state_dict' not found."
            )

        try:

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )

        except RuntimeError as error:

            raise RuntimeError(
                "Detector checkpoint is not compatible "
                "with the requested architecture.\n"
                f"Backbone: {self.backbone_type}\n"
                f"ResNet depth: {self.resnet_depth}\n"
                f"Checkpoint: {self.checkpoint_path}\n\n"
                f"Original error:\n{error}"
            ) from error

        del checkpoint

        self.model.eval()

        print(
            "[LOG] Detector checkpoint loaded."
        )

        # --------------------------------------------------------
        # Post-processors
        # --------------------------------------------------------

        # Standard post-processing used during analysis.
        self.official_postprocessor = (
            DetectionPostProcessor(
                score_threshold=(
                    OFFICIAL_SCORE_THRESHOLD
                ),
                nms_threshold=(
                    NMS_THRESHOLD
                ),
            )
        )

        # During calibration no score threshold is applied,
        # otherwise low-confidence detections would be removed
        # before the optimal threshold is searched.
        self.calibration_postprocessor = (
            DetectionPostProcessor(
                score_threshold=0.0,
                nms_threshold=(
                    NMS_THRESHOLD
                ),
            )
        )

    # =========================================================
    # Threshold calibration
    # =========================================================

    def calibrate_threshold(self):
        """
        Determine the operating threshold used for analysis.

        If a manual threshold is supplied, it is used directly.
        Otherwise, Youden's J criterion is used to select the
        threshold from validation predictions.
        """

        # -----------------------------------------------------
        # Manual threshold
        # -----------------------------------------------------

        if self.manual_threshold is not None:

            self.tau = float(
                self.manual_threshold
            )

            self.visualization_threshold = (
                self.tau
            )

            print(
                "[CALIBRATION] "
                f"Manual threshold = "
                f"{self.visualization_threshold:.4f}"
            )

            return

        # -----------------------------------------------------
        # Automatic calibration
        # -----------------------------------------------------

        print()
        print(
            "=" * 80
        )

        print(
            "CALIBRATING DETECTION THRESHOLD"
        )

        print(
            "=" * 80
        )

        calibrator = (
            DetectionThresholdCalibrator(
                model=self.model,
                postprocessor=(
                    self.calibration_postprocessor
                ),
                device=self.device,
                criterion="youden",
                visualization_rule=False,
                verbose=True,
            )
        )

        result = calibrator.calibrate(
            dataloader=self.val_loader,
            num_thresholds=101,
            save_path=(
                self.output_dir
                / "metrics"
                / "threshold_calibration.csv"
            ),
        )

        self.tau = float(
            result.threshold
        )

        self.visualization_threshold = (
            self.tau
        )

        print()
        print(
            "[LOG] tau* = "
            f"{self.tau:.4f}"
        )

    # =========================================================
    # Prediction filtering
    # =========================================================

    def filter_predictions(
        self,
        detections,
    ):
        """
        Filter detections used for analysis and visualization.

        First applies the selected score threshold and then
        removes highly redundant overlapping predictions.
        """

        if self.visualization_threshold is None:

            raise RuntimeError(
                "Threshold has not been calibrated. "
                "Call calibrate_threshold() first."
            )

        boxes = detections[
            "boxes"
        ]

        scores = detections[
            "scores"
        ]

        # Score threshold.
        keep = (
            scores
            >= self.visualization_threshold
        )

        boxes = boxes[
            keep
        ]

        scores = scores[
            keep
        ]

        # Remove redundant boxes for visualization.
        (
            boxes,
            scores,
        ) = suppress_redundant_predictions(
            boxes=boxes,
            scores=scores,
            overlap_threshold=(
                self.overlap_threshold
            ),
            max_detections=(
                self.max_detections
            ),
        )

        return {
            "boxes": boxes,
            "scores": scores,
        }

    # =========================================================
    # Image-level classification
    # =========================================================

    @staticmethod
    def classify_image(
        gt_boxes,
        detections,
    ):
        """
        Classify an image as TP, TN, FP or FN.

        This is an image-level classification based only on
        whether ground-truth and predicted boxes are present.
        """

        gt_positive = (
            len(gt_boxes)
            > 0
        )

        predicted_positive = (
            len(
                detections[
                    "boxes"
                ]
            )
            > 0
        )

        if (
            gt_positive
            and predicted_positive
        ):
            return "TP"

        if (
            gt_positive
            and not predicted_positive
        ):
            return "FN"

        if (
            not gt_positive
            and predicted_positive
        ):
            return "FP"

        return "TN"

    # =========================================================
    # Validation
    # =========================================================

    @torch.no_grad()
    def collect_validation_results(self):
        """
        Run a complete validation pass.

        Predictions are filtered, matched to ground-truth boxes,
        and stored together with the corresponding images.

        No visualization is performed here.
        """

        results = []

        # Image-level statistics.
        image_tp = 0
        image_tn = 0
        image_fp = 0
        image_fn = 0

        # Box-level statistics.
        box_tp = 0
        box_fp = 0
        box_fn = 0

        total_images = 0
        total_gt_boxes = 0
        total_pred_boxes = 0

        matched_ious = []

        # Raw post-NMS predictions are stored separately for the
        # post-training Precision-Recall curve. They are kept before any
        # score threshold or visualization-only redundancy filtering.
        raw_predictions = []
        raw_targets = []

        self.model.eval()

        total_batches = len(
            self.val_loader
        )

        print()
        print(
            "=" * 80
        )

        print(
            "FULL VALIDATION ANALYSIS"
        )

        print(
            "=" * 80
        )

        for (
            batch_index,
            (
                images,
                targets,
            ),
        ) in enumerate(
            self.val_loader,
            start=1,
        ):

            # -------------------------------------------------
            # Move input to device
            # -------------------------------------------------

            images = images.to(
                self.device
            )

            # -------------------------------------------------
            # Detector forward pass
            # -------------------------------------------------

            predictions = (
                self.model(
                    images
                )
            )

            # -------------------------------------------------
            # Raw post-processing for PR / AP analysis
            # -------------------------------------------------

            raw_detections = (
                self.calibration_postprocessor(
                    predictions
                )
            )

            for sample_index, raw_detection in enumerate(
                raw_detections
            ):
                raw_predictions.append(
                    {
                        "boxes": (
                            raw_detection["boxes"]
                            .detach()
                            .cpu()
                        ),
                        "scores": (
                            raw_detection["scores"]
                            .detach()
                            .cpu()
                        ),
                    }
                )

                raw_targets.append(
                    {
                        "boxes": (
                            targets[sample_index]["boxes"]
                            .detach()
                            .cpu()
                        ),
                    }
                )

            # -------------------------------------------------
            # Standard post-processing
            # -------------------------------------------------

            detections = (
                self.official_postprocessor(
                    predictions
                )
            )

            # -------------------------------------------------
            # Process each image
            # -------------------------------------------------

            for (
                sample_index,
                detection,
            ) in enumerate(
                detections
            ):

                target = targets[
                    sample_index
                ]

                # Ground-truth boxes are kept on CPU
                # for analysis and visualization.
                gt_boxes = (
                    target[
                        "boxes"
                    ]
                    .detach()
                    .cpu()
                    .float()
                )

                # -------------------------------------------------
                # Map validation batch position to dataset index
                # -------------------------------------------------

                position = (
                    (
                        batch_index - 1
                    )
                    * BATCH_SIZE
                    + sample_index
                )

                if (
                    position
                    >= len(
                        self.val_indices
                    )
                ):
                    continue

                dataset_index = (
                    self.val_indices[
                        position
                    ]
                )

                patient_id = (
                    self.dataset
                    .image_paths[
                        dataset_index
                    ]
                    .stem
                )

                # -------------------------------------------------
                # Filter predictions
                # -------------------------------------------------

                visual_detections = (
                    self.filter_predictions(
                        detection
                    )
                )

                gt_list = (
                    gt_boxes.tolist()
                )

                pred_boxes = (
                    visual_detections[
                        "boxes"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                pred_scores = (
                    visual_detections[
                        "scores"
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                # -------------------------------------------------
                # Image-level classification
                # -------------------------------------------------

                category = (
                    self.classify_image(
                        gt_boxes,
                        visual_detections,
                    )
                )

                if category == "TP":

                    image_tp += 1

                elif category == "TN":

                    image_tn += 1

                elif category == "FP":

                    image_fp += 1

                elif category == "FN":

                    image_fn += 1

                # -------------------------------------------------
                # Box-level matching
                # -------------------------------------------------

                (
                    current_tp,
                    current_fp,
                    current_fn,
                    current_ious,
                ) = (
                    match_predictions_to_ground_truth(
                        gt_boxes=gt_list,
                        pred_boxes=pred_boxes,
                        iou_threshold=(
                            IOU_THRESHOLD
                        ),
                    )
                )

                box_tp += current_tp
                box_fp += current_fp
                box_fn += current_fn

                matched_ious.extend(
                    current_ious
                )

                # -------------------------------------------------
                # Per-image statistics
                # -------------------------------------------------

                max_score = (
                    max(pred_scores)
                    if pred_scores
                    else 0.0
                )

                best_iou = (
                    max(current_ious)
                    if current_ious
                    else 0.0
                )

                results.append(
                    {
                        "dataset_index": int(
                            dataset_index
                        ),

                        "patient_id": str(
                            patient_id
                        ),

                        "category": (
                            category
                        ),

                        "num_gt_boxes": len(
                            gt_list
                        ),

                        "num_predictions": len(
                            pred_boxes
                        ),

                        "max_score": float(
                            max_score
                        ),

                        "best_matched_iou": float(
                            best_iou
                        ),

                        "box_tp": int(
                            current_tp
                        ),

                        "box_fp": int(
                            current_fp
                        ),

                        "box_fn": int(
                            current_fn
                        ),

                        # Stored for later visualization.
                        "image": (
                            images[
                                sample_index
                            ]
                            .detach()
                            .cpu()
                        ),

                        "gt_boxes": (
                            gt_boxes
                        ),

                        "detections": (
                            visual_detections
                        ),
                    }
                )

                total_images += 1

                total_gt_boxes += (
                    len(gt_list)
                )

                total_pred_boxes += (
                    len(pred_boxes)
                )

            # -----------------------------------------------------
            # Progress
            # -----------------------------------------------------

            if (
                batch_index % 100 == 0
                or batch_index == total_batches
            ):

                progress = (
                    100.0
                    * batch_index
                    / total_batches
                )

                print(
                    "[ANALYSIS] "
                    f"batch={batch_index}/"
                    f"{total_batches} "
                    f"progress={progress:.1f}%"
                )

        return {
            "results": results,

            "image_tp": (
                image_tp
            ),

            "image_tn": (
                image_tn
            ),

            "image_fp": (
                image_fp
            ),

            "image_fn": (
                image_fn
            ),

            "box_tp": (
                box_tp
            ),

            "box_fp": (
                box_fp
            ),

            "box_fn": (
                box_fn
            ),

            "total_images": (
                total_images
            ),

            "total_gt_boxes": (
                total_gt_boxes
            ),

            "total_pred_boxes": (
                total_pred_boxes
            ),

            "matched_ious": (
                matched_ious
            ),

            "raw_predictions": (
                raw_predictions
            ),

            "raw_targets": (
                raw_targets
            ),
        }

    # =========================================================
    # Metrics
    # =========================================================

    @staticmethod
    def _safe_divide(
        numerator,
        denominator,
    ):
        """
        Divide safely and return zero when denominator is zero.
        """

        if denominator == 0:

            return 0.0

        return (
            numerator
            / denominator
        )

    @classmethod
    def _compute_binary_metrics(
        cls,
        tp,
        tn,
        fp,
        fn,
    ):
        """
        Compute binary classification metrics from TP/TN/FP/FN.
        """

        precision = (
            cls._safe_divide(
                tp,
                tp + fp,
            )
        )

        recall = (
            cls._safe_divide(
                tp,
                tp + fn,
            )
        )

        specificity = (
            cls._safe_divide(
                tn,
                tn + fp,
            )
        )

        accuracy = (
            cls._safe_divide(
                tp + tn,
                tp + tn + fp + fn,
            )
        )

        f1 = (
            cls._safe_divide(
                2.0
                * precision
                * recall,
                precision + recall,
            )
        )

        youden_j = (
            recall
            + specificity
            - 1.0
        )

        return {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),

            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "specificity": float(
                specificity
            ),

            "f1": float(
                f1
            ),

            "youden_j": float(
                youden_j
            ),
        }

    def compute_metrics(
        self,
        analysis,
    ):
        """
        Compute image-level and box-level analysis metrics.
        """

        # -----------------------------------------------------
        # Box-level metrics
        # -----------------------------------------------------

        box_metrics = (
            self._compute_binary_metrics(
                tp=analysis[
                    "box_tp"
                ],

                tn=0,

                fp=analysis[
                    "box_fp"
                ],

                fn=analysis[
                    "box_fn"
                ],
            )
        )

        # -----------------------------------------------------
        # Image-level metrics
        # -----------------------------------------------------

        image_metrics = (
            self._compute_binary_metrics(
                tp=analysis[
                    "image_tp"
                ],

                tn=analysis[
                    "image_tn"
                ],

                fp=analysis[
                    "image_fp"
                ],

                fn=analysis[
                    "image_fn"
                ],
            )
        )

        # -----------------------------------------------------
        # Mean IoU of successful matches
        # -----------------------------------------------------

        mean_best_iou = (
            self._safe_divide(
                sum(
                    analysis[
                        "matched_ious"
                    ]
                ),

                len(
                    analysis[
                        "matched_ious"
                    ]
                ),
            )
        )

        # -----------------------------------------------------
        # Average number of predictions
        # -----------------------------------------------------

        mean_predictions = (
            self._safe_divide(
                analysis[
                    "total_pred_boxes"
                ],

                analysis[
                    "total_images"
                ],
            )
        )

        # -----------------------------------------------------
        # Standard post-training detection metrics
        # -----------------------------------------------------
        # These metrics are computed from the raw post-NMS predictions
        # before the analysis confidence threshold is applied.
        #
        # Returns: AP, AP_M, AP_L, AR@10, AR_M and AR_L.

        detection_metrics = compute_detection_metrics(
            predictions=analysis["raw_predictions"],
            targets=analysis["raw_targets"],
        )

        # -----------------------------------------------------
        # Precision-Recall curve / AP
        # -----------------------------------------------------

        ap_curve = compute_ap_curve(
            predictions=analysis["raw_predictions"],
            targets=analysis["raw_targets"],
            iou_threshold=IOU_THRESHOLD,
        )

        save_precision_recall_curve(
            precisions=ap_curve["precisions"],
            recalls=ap_curve["recalls"],
            ap=ap_curve["AP"],
            output_dir=self.output_dir,
        )

        # Sanity check: both implementations should produce the same
        # global AP because they operate on the same raw predictions.
        if abs(
            float(detection_metrics["AP"])
            - float(ap_curve["AP"])
        ) > 1e-8:
            print(
                "[WARNING] AP mismatch between compute_metrics and "
                "compute_ap_curve: "
                f"{float(detection_metrics['AP']):.8f} vs "
                f"{float(ap_curve['AP']):.8f}"
            )

        # -----------------------------------------------------
        # Combined result
        # -----------------------------------------------------

        return {
            "experiment": (
                self.checkpoint_path
                .parent
                .name
            ),

            "checkpoint": str(
                self.checkpoint_path
            ),

            "backbone": (
                self.backbone_type
            ),

            "resnet_depth": int(
                self.resnet_depth
            ),

            "image_size": int(
                IMAGE_SIZE
            ),

            "official_score_threshold": float(
                OFFICIAL_SCORE_THRESHOLD
            ),

            "nms_threshold": float(
                NMS_THRESHOLD
            ),

            "visualization_threshold": float(
                self.visualization_threshold
            ),

            "evaluation_iou_threshold": float(
                IOU_THRESHOLD
            ),

            "visualization_overlap_threshold": float(
                self.overlap_threshold
            ),

            "num_validation_images": int(
                analysis[
                    "total_images"
                ]
            ),

            "total_gt_boxes": int(
                analysis[
                    "total_gt_boxes"
                ]
            ),

            "total_visualization_predictions": int(
                analysis[
                    "total_pred_boxes"
                ]
            ),

            "mean_predictions_per_image": float(
                mean_predictions
            ),

            "mean_best_matched_iou": float(
                mean_best_iou
            ),

            "box_level": (
                box_metrics
            ),

            "image_level": (
                image_metrics
            ),

            "precision_recall": {
                "AP": float(
                    ap_curve["AP"]
                ),
                "num_gt": int(
                    ap_curve["num_gt"]
                ),
                "num_pred": int(
                    ap_curve["num_pred"]
                ),
            },

            "detection_metrics": {
                "AP": float(
                    detection_metrics["AP"]
                ),
                "AP_M": float(
                    detection_metrics["AP_M"]
                ),
                "AP_L": float(
                    detection_metrics["AP_L"]
                ),
                "AR@10": float(
                    detection_metrics["AR@10"]
                ),
                "AR_M": float(
                    detection_metrics["AR_M"]
                ),
                "AR_L": float(
                    detection_metrics["AR_L"]
                ),
            },
        }