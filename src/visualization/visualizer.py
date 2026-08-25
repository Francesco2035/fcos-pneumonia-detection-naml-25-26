from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from torchvision.transforms import v2

from src.config import (
    CSV_PATH,
    RESNET50_CHEST_XRAY_CHECKPOINT,
    TRAIN_DCM_PATH,
)
from src.models.detector import DetectionFramework
from src.inference import DetectionPostProcessor


class DetectionVisualizer:

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
        checkpoint_path,
        csv_path=CSV_PATH,
        backbone="chest_xray",
        device=None,
        image_size=512,
        score_threshold=0.30,
        max_detections=10,
        nms_threshold=0.50,
        output_dir="visualizations",
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.csv_path = Path(
            csv_path
        )

        self.backbone = backbone
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.output_dir = Path(
            output_dir
        )

        self.device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self.backbone not in {
            "chest_xray",
            "imagenet",
        }:
            raise ValueError(
                "backbone must be 'chest_xray' or 'imagenet'"
            )

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"CSV not found:\n"
                f"{self.csv_path}"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # Same validation preprocessing
        # =====================================================

        self.transform = v2.Compose([
            v2.Resize(
                (image_size, image_size)
            ),
            v2.Grayscale(
                num_output_channels=3
            ),
            v2.ToDtype(
                torch.float32,
                scale=True,
            ),
        ])

        # =====================================================
        # Ground-truth CSV
        # =====================================================

        self.df = pd.read_csv(
            self.csv_path
        )

        self._build_patient_index()

        # =====================================================
        # Backbone
        # =====================================================

        if self.backbone == "chest_xray":

            path_model = (
                RESNET50_CHEST_XRAY_CHECKPOINT
            )

            if not Path(
                path_model
            ).is_file():
                raise FileNotFoundError(
                    "Chest-Xray backbone checkpoint "
                    f"not found:\n{path_model}"
                )

            print(
                "[LOG] Backbone: "
                "Chest-Xray pretrained ResNet-50"
            )

        else:

            path_model = None

            print(
                "[LOG] Backbone: "
                "ImageNet pretrained ResNet-50"
            )

        # =====================================================
        # Detector
        # =====================================================

        self.model = DetectionFramework(
            path_model=path_model
        ).to(self.device)

        print(
            f"[LOG] Loading detector checkpoint:\n"
            f"      {self.checkpoint_path}"
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        del checkpoint

        self.model.eval()

        print(
            "[LOG] Detector weights loaded."
        )

        # =====================================================
        # Postprocessor
        #
        # Keep threshold 0 here because this is the same
        # post-processing base used for evaluation.
        # We apply visualization threshold afterward.
        # =====================================================

        self.postprocessor = (
            DetectionPostProcessor(
                score_threshold=0.0,
                nms_threshold=nms_threshold,
            )
        )

    # =========================================================
    # Dataset index
    # =========================================================

    def _build_patient_index(self):

        grouped = (
            self.df
            .groupby("patientId")["Target"]
            .max()
        )

        self.positive_ids = sorted(
            grouped[
                grouped == 1
            ].index.astype(str).tolist()
        )

        self.negative_ids = sorted(
            grouped[
                grouped == 0
            ].index.astype(str).tolist()
        )

    # =========================================================
    # Automatically select a sample
    # =========================================================

    def find_sample(
        self,
        kind,
    ):

        if kind == "positive":
            patient_ids = self.positive_ids
        elif kind == "negative":
            patient_ids = self.negative_ids
        else:
            raise ValueError(
                "kind must be 'positive' or 'negative'"
            )

        train_dir = Path(
            TRAIN_DCM_PATH
        )

        for patient_id in patient_ids:

            path = (
                train_dir
                / f"{patient_id}.dcm"
            )

            if path.is_file():
                return path

        raise FileNotFoundError(
            f"No DICOM found for {kind} sample."
        )

    # =========================================================
    # Load DICOM
    # =========================================================

    def load_dicom(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        dicom = pydicom.dcmread(
            str(image_path)
        )

        image = dicom.pixel_array

        image = np.asarray(
            image,
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # Display image normalization
        # -----------------------------------------------------

        min_value = float(
            image.min()
        )

        max_value = float(
            image.max()
        )

        if max_value > min_value:

            display_image = (
                image - min_value
            ) / (
                max_value - min_value
            )

        else:

            display_image = np.zeros_like(
                image
            )

        original_height, original_width = (
            image.shape
        )

        # -----------------------------------------------------
        # Detector input
        # -----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0)

        tensor = v2.ToImage()(
            tensor
        )

        tensor = self.transform(
            tensor
        )

        tensor = tensor.unsqueeze(
            0
        ).to(
            self.device
        )

        return (
            display_image,
            tensor,
            original_width,
            original_height,
        )

    # =========================================================
    # Ground truth
    # =========================================================

    def get_ground_truth(
        self,
        patient_id,
    ):

        rows = self.df[
            self.df["patientId"].astype(str)
            == str(patient_id)
        ]

        boxes = []

        for _, row in rows.iterrows():

            if int(row["Target"]) != 1:
                continue

            x1 = float(row["x"])
            y1 = float(row["y"])

            x2 = (
                x1
                + float(row["width"])
            )

            y2 = (
                y1
                + float(row["height"])
            )

            boxes.append(
                [x1, y1, x2, y2]
            )

        if not boxes:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

        return torch.tensor(
            boxes,
            dtype=torch.float32,
        )

    # =========================================================
    # Forward
    # =========================================================

    @torch.no_grad()
    def forward(
        self,
        image,
    ):

        self.model.eval()

        return self.model(
            image
        )

    # =========================================================
    # Backbone
    # =========================================================

    @torch.no_grad()
    def get_backbone_features(
        self,
        image,
    ):

        C2, C3, C4, C5 = (
            self.model.fpn.backbone(
                image
            )
        )

        return {
            "C2": C2,
            "C3": C3,
            "C4": C4,
            "C5": C5,
        }

    # =========================================================
    # FPN
    # =========================================================

    @torch.no_grad()
    def get_fpn_features(
        self,
        image,
    ):

        P3, P4, P5, P6, P7 = (
            self.model.fpn(
                image
            )
        )

        return {
            "P3": P3,
            "P4": P4,
            "P5": P5,
            "P6": P6,
            "P7": P7,
        }

    # =========================================================
    # Decode
    # =========================================================

    @torch.no_grad()
    def decode_predictions(
        self,
        raw_predictions,
    ):

        results = self.postprocessor(
            raw_predictions
        )

        return results[0]

    # =========================================================
    # Visualization-only selection
    # =========================================================

    def select_display_predictions(
        self,
        detections,
    ):

        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["labels"]

        # -----------------------------------------------------
        # Confidence threshold
        # -----------------------------------------------------

        keep = (
            scores
            >= self.score_threshold
        )

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # -----------------------------------------------------
        # Top-K
        # -----------------------------------------------------

        if len(scores) > self.max_detections:

            order = (
                scores.argsort(
                    descending=True
                )[
                    : self.max_detections
                ]
            )

            boxes = boxes[order]
            scores = scores[order]
            labels = labels[order]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    # =========================================================
    # IoU
    # =========================================================

    @staticmethod
    def compute_iou(
        box_a,
        box_b,
    ):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            iw * ih
        )

        area_a = (
            max(0.0, ax2 - ax1)
            * max(0.0, ay2 - ay1)
        )

        area_b = (
            max(0.0, bx2 - bx1)
            * max(0.0, by2 - by1)
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection / union
        )

    # =========================================================
    # Final image with GT + predictions
    # =========================================================

    def plot_detection_result(
        self,
        display_image,
        original_width,
        original_height,
        gt_boxes,
        detections,
        patient_id,
        has_pneumonia,
        output_path,
    ):

        fig, ax = plt.subplots(
            figsize=(12, 12)
        )

        ax.imshow(
            display_image,
            cmap="gray",
        )

        # -----------------------------------------------------
        # GT
        # -----------------------------------------------------

        for index, box in enumerate(
            gt_boxes.tolist()
        ):

            x1, y1, x2, y2 = box

            rectangle = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=3,
                edgecolor="lime",
            )

            ax.add_patch(
                rectangle
            )

            ax.text(
                x1,
                max(0, y1 - 8),
                f"GT #{index + 1} | Pneumonia",
                color="lime",
                fontsize=10,
                backgroundcolor="black",
            )

        # -----------------------------------------------------
        # Predictions
        # -----------------------------------------------------

        pred_boxes = (
            detections["boxes"]
            .detach()
            .cpu()
            .clone()
        )

        pred_scores = (
            detections["scores"]
            .detach()
            .cpu()
        )

        scale_x = (
            original_width
            / float(self.image_size)
        )

        scale_y = (
            original_height
            / float(self.image_size)
        )

        if len(pred_boxes) > 0:

            pred_boxes[:, [0, 2]] *= (
                scale_x
            )

            pred_boxes[:, [1, 3]] *= (
                scale_y
            )

        for index, (
            box,
            score,
        ) in enumerate(
            zip(
                pred_boxes.tolist(),
                pred_scores.tolist(),
            )
        ):

            x1, y1, x2, y2 = box

            best_iou = 0.0

            if len(gt_boxes) > 0:

                best_iou = max(
                    self.compute_iou(
                        box,
                        gt,
                    )
                    for gt in gt_boxes.tolist()
                )

            status = (
                "TP"
                if best_iou >= 0.5
                else "FP"
            )

            rectangle = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                edgecolor="red",
            )

            ax.add_patch(
                rectangle
            )

            ax.text(
                x1,
                y2,
                (
                    f"Pneumonia "
                    f"{score:.2f}\n"
                    f"IoU={best_iou:.2f} "
                    f"{status}"
                ),
                color="red",
                fontsize=9,
                backgroundcolor="white",
            )

        pneumonia_text = (
            "YES"
            if has_pneumonia
            else "NO"
        )

        ax.set_title(
            (
                f"{patient_id} | "
                f"Ground truth: "
                f"Pneumonia = "
                f"{pneumonia_text}\n"
                f"Green = GT | "
                f"Red = prediction | "
                f"Threshold = "
                f"{self.score_threshold:.2f}"
            ),
            fontsize=13,
        )

        ax.axis("off")

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.show()

        plt.close(
            fig
        )

    # =========================================================
    # Backbone plot
    # =========================================================

    @staticmethod
    def activation_map(
        feature,
    ):

        return (
            feature[0]
            .abs()
            .mean(dim=0)
            .detach()
            .cpu()
        )

    def plot_backbone(
        self,
        features,
        output_path,
    ):

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4),
        )

        for ax, (
            name,
            feature,
        ) in zip(
            axes,
            features.items(),
        ):

            activation = (
                self.activation_map(
                    feature
                )
            )

            ax.imshow(
                activation,
                cmap="gray",
            )

            ax.set_title(
                f"{name}\n"
                f"{tuple(feature.shape)}"
            )

            ax.axis("off")

        fig.suptitle(
            "Backbone feature hierarchy"
        )

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

    # =========================================================
    # FPN plot
    # =========================================================

    def plot_fpn(
        self,
        features,
        output_path,
    ):

        fig, axes = plt.subplots(
            1,
            5,
            figsize=(20, 4),
        )

        for ax, level in zip(
            axes,
            self.LEVELS,
        ):

            feature = features[
                level
            ]

            activation = (
                self.activation_map(
                    feature
                )
            )

            ax.imshow(
                activation,
                cmap="gray",
            )

            ax.set_title(
                f"{level}\n"
                f"stride={self.STRIDES[level]}\n"
                f"{tuple(feature.shape)}"
            )

            ax.axis("off")

        fig.suptitle(
            "FPN feature pyramid"
        )

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

    # =========================================================
    # Head outputs
    # =========================================================

    def plot_heads(
        self,
        predictions,
        output_path,
    ):

        fig, axes = plt.subplots(
            5,
            3,
            figsize=(15, 20),
        )

        for row, level in enumerate(
            self.LEVELS
        ):

            pred = predictions[
                level
            ]

            classification = (
                torch.sigmoid(
                    pred[
                        "classification"
                    ][0, 0]
                )
                .detach()
                .cpu()
            )

            centerness = (
                torch.sigmoid(
                    pred[
                        "centerness"
                    ][0, 0]
                )
                .detach()
                .cpu()
            )

            regression = (
                pred[
                    "regression"
                ][0]
                .abs()
                .mean(dim=0)
                .detach()
                .cpu()
            )

            axes[row, 0].imshow(
                classification,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[row, 0].set_title(
                f"{level} classification"
            )

            axes[row, 1].imshow(
                centerness,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            axes[row, 1].set_title(
                f"{level} centerness"
            )

            axes[row, 2].imshow(
                regression,
                cmap="gray",
            )

            axes[row, 2].set_title(
                f"{level} regression"
            )

            for col in range(3):
                axes[row, col].axis(
                    "off"
                )

        fig.suptitle(
            "Detection head outputs"
        )

        plt.tight_layout()

        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

    # =========================================================
    # One sample
    # =========================================================

    @torch.no_grad()
    def visualize_sample(
        self,
        image_path,
        output_dir,
    ):

        image_path = Path(
            image_path
        )

        patient_id = (
            image_path.stem
        )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        gt_boxes = (
            self.get_ground_truth(
                patient_id
            )
        )

        has_pneumonia = (
            len(gt_boxes) > 0
        )

        print()
        print("=" * 80)

        print(
            f"Patient:       {patient_id}"
        )

        print(
            "Ground truth:  "
            f"Pneumonia = "
            f"{'YES' if has_pneumonia else 'NO'}"
        )

        print(
            f"DICOM:         {image_path}"
        )

        print("=" * 80)

        (
            display_image,
            image,
            original_width,
            original_height,
        ) = self.load_dicom(
            image_path
        )

        # -----------------------------------------------------
        # Feature extraction
        # -----------------------------------------------------

        print(
            "[LOG] Backbone..."
        )

        backbone_features = (
            self.get_backbone_features(
                image
            )
        )

        print(
            "[LOG] FPN..."
        )

        fpn_features = (
            self.get_fpn_features(
                image
            )
        )

        print(
            "[LOG] Detection heads..."
        )

        predictions = self.forward(
            image
        )

        print(
            "[LOG] Decode + NMS..."
        )

        detections = (
            self.decode_predictions(
                predictions
            )
        )

        print(
            f"[LOG] Predictions after NMS: "
            f"{len(detections['boxes'])}"
        )

        # -----------------------------------------------------
        # Display filtering
        # -----------------------------------------------------

        display_detections = (
            self.select_display_predictions(
                detections
            )
        )

        print(
            f"[LOG] Predictions shown after "
            f"threshold={self.score_threshold:.2f}: "
            f"{len(display_detections['boxes'])}"
        )

        print(
            f"[LOG] Ground-truth boxes: "
            f"{len(gt_boxes)}"
        )

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        self.plot_detection_result(
            display_image=display_image,
            original_width=original_width,
            original_height=original_height,
            gt_boxes=gt_boxes,
            detections=display_detections,
            patient_id=patient_id,
            has_pneumonia=has_pneumonia,
            output_path=(
                output_dir
                / "detections.png"
            ),
        )

        # -----------------------------------------------------
        # Backbone
        # -----------------------------------------------------

        self.plot_backbone(
            backbone_features,
            output_dir / "backbone.png",
        )

        # -----------------------------------------------------
        # FPN
        # -----------------------------------------------------

        self.plot_fpn(
            fpn_features,
            output_dir / "fpn.png",
        )

        # -----------------------------------------------------
        # Heads
        # -----------------------------------------------------

        self.plot_heads(
            predictions,
            output_dir / "heads.png",
        )

        print()
        print(
            f"[SAVED] {output_dir}"
        )

    # =========================================================
    # Automatically visualize positive and negative
    # =========================================================

    def visualize_samples(
        self,
    ):

        positive_path = self.find_sample(
            "positive"
        )

        negative_path = self.find_sample(
            "negative"
        )

        self.visualize_sample(
            positive_path,
            self.output_dir / "positive",
        )

        self.visualize_sample(
            negative_path,
            self.output_dir / "negative",
        )