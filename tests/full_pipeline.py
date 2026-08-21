import torch

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.models.detector import DetectionFramework
from src.models.target_generator import TargetGenerator
from src.datasets.transforms import get_test_transforms


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_IMAGES = 100

IMAGE_SIZE = 224



DICOM_PATH = (
"data/rsna-pneumonia-detection-challenge/"
"stage_2_train_images"
)

CSV_PATH = (
"data/rsna-pneumonia-detection-challenge/"
"stage_2_train_labels.csv"
)

CHECKPOINT_PATH = (
"checkpoints/resnet50_scratch_chest_xray_best.pth"
)

MODEL_PATH = None


# ============================================================
# CHECK UTILITY
# ============================================================

def check(condition, message):

    if not condition:
        raise AssertionError(message)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    print()
    print("Device:", DEVICE)
    print("Images to test:", NUM_IMAGES)

    # ========================================================
    # 1. DATASET
    # ========================================================

    print()
    print("[1] Creating dataset...")

    dataset = RSNAPneumoniaDataset(
        dcm_path=DICOM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            image_size=IMAGE_SIZE
        ),
    )

    print("Dataset size:", len(dataset))

    check(
        len(dataset) > 0,
        "Dataset is empty."
    )

    # ========================================================
    # 2. DATALOADER
    # ========================================================

    print()
    print("[2] Creating DataLoader...")

    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # ========================================================
    # 3. DETECTION FRAMEWORK
    # ========================================================

    print()
    print("[3] Creating DetectionFramework...")

    model = DetectionFramework(
        path_model=MODEL_PATH
    )

    model = model.to(DEVICE)

    model.eval()

    # ========================================================
    # 4. TARGET GENERATOR
    # ========================================================

    print()
    print("[4] Creating TargetGenerator...")

    target_generator = TargetGenerator()

    # ========================================================
    # FPN CONFIGURATION
    # ========================================================

    levels = {
        "P3": {
            "stride": 8,
        },

        "P4": {
            "stride": 16,
        },

        "P5": {
            "stride": 32,
        },

        "P6": {
            "stride": 64,
        },

        "P7": {
            "stride": 128,
        },
    }

    # ========================================================
    # STATISTICS
    # ========================================================

    images_tested = 0

    images_with_gt = 0
    images_without_gt = 0

    total_gt_boxes = 0

    level_positive_images = {
        level: 0
        for level in levels
    }

    level_positive_points = {
        level: 0
        for level in levels
    }

    # ========================================================
    # 5. TEST IMAGES
    # ========================================================

    print()
    print(
        f"[5] Testing {NUM_IMAGES} images..."
    )

    with torch.no_grad():

        for batch_index, (images, targets) in enumerate(
            dataloader
        ):

            if images_tested >= NUM_IMAGES:
                break

            # ------------------------------------------------
            # Images
            # ------------------------------------------------

            check(
                images.ndim == 4,
                f"Expected images [B,C,H,W], got {images.shape}"
            )

            check(
                images.shape[0] == 1,
                f"Expected batch size 1, got {images.shape[0]}"
            )

            check(
                images.shape[1] == 3,
                f"Expected 3 image channels, got {images.shape[1]}"
            )

            check(
                images.shape[2] == IMAGE_SIZE,
                f"Expected image height {IMAGE_SIZE}, "
                f"got {images.shape[2]}"
            )

            check(
                images.shape[3] == IMAGE_SIZE,
                f"Expected image width {IMAGE_SIZE}, "
                f"got {images.shape[3]}"
            )

            # ------------------------------------------------
            # Move image to device
            # ------------------------------------------------

            images = images.to(DEVICE)

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            check(
                len(targets) == 1,
                f"Expected one target, got {len(targets)}"
            )

            target = targets[0]

            boxes = target["boxes"]

            # ------------------------------------------------
            # Ground-truth boxes
            # ------------------------------------------------

            check(
                boxes.ndim == 2,
                f"Expected boxes [N,4], got {boxes.shape}"
            )

            check(
                boxes.shape[1] == 4,
                f"Expected 4 box coordinates, got {boxes.shape}"
            )

            num_boxes = boxes.shape[0]

            total_gt_boxes += num_boxes

            if num_boxes > 0:
                images_with_gt += 1
            else:
                images_without_gt += 1

            # ------------------------------------------------
            # Convert boxes to regular tensor
            # ------------------------------------------------

            boxes = boxes.to(
                dtype=torch.float32,
                device=DEVICE,
            )

            # ------------------------------------------------
            # Model forward
            # ------------------------------------------------

            predictions = model(images)

            # ------------------------------------------------
            # Statistics for this image
            # ------------------------------------------------

            image_positive_points = 0
            image_levels = []

            # =================================================
            # Check every FPN level
            # =================================================

            for level, config in levels.items():

                stride = config["stride"]

                prediction = predictions[level]

                # ------------------------------------------------
                # Predictions
                # ------------------------------------------------

                classification = prediction[
                    "classification"
                ]

                center = prediction[
                    "centerness"
                ]

                scale = prediction[
                    "regression"
                ]

                # ------------------------------------------------
                # Prediction shape checks
                # ------------------------------------------------

                check(
                    classification.ndim == 4,
                    (
                        f"{level}: classification must be "
                        f"[B,C,H,W], got "
                        f"{classification.shape}"
                    )
                )

                check(
                    center.ndim == 4,
                    (
                        f"{level}: center must be "
                        f"[B,C,H,W], got "
                        f"{center.shape}"
                    )
                )

                check(
                    scale.ndim == 4,
                    (
                        f"{level}: scale must be "
                        f"[B,C,H,W], got "
                        f"{scale.shape}"
                    )
                )

                # ------------------------------------------------
                # Number of channels
                # ------------------------------------------------

                check(
                    classification.shape[1] == 1,
                    (
                        f"{level}: classification must "
                        f"have 1 channel, got "
                        f"{classification.shape[1]}"
                    )
                )

                check(
                    center.shape[1] == 1,
                    (
                        f"{level}: center must "
                        f"have 1 channel, got "
                        f"{center.shape[1]}"
                    )
                )

                check(
                    scale.shape[1] == 4,
                    (
                        f"{level}: scale must "
                        f"have 4 channels, got "
                        f"{scale.shape[1]}"
                    )
                )

                # ------------------------------------------------
                # Feature-map shape
                # ------------------------------------------------

                height = classification.shape[2]
                width = classification.shape[3]

                check(
                    center.shape[2:] == (height, width),
                    (
                        f"{level}: classification/center "
                        f"shape mismatch."
                    )
                )

                check(
                    scale.shape[2:] == (height, width),
                    (
                        f"{level}: classification/scale "
                        f"shape mismatch."
                    )
                )

                # =================================================
                # TARGET GENERATION
                # =================================================

                targets_level = (
                    target_generator.generate_targets(
                        label_boxes=boxes,
                        feature_shape=(height, width),
                        stride=stride,
                        device=DEVICE,
                    )
                )

                positive = targets_level[
                    "positive"
                ]

                ltrb = targets_level[
                    "ltrb"
                ]

                centerness = targets_level[
                    "centerness"
                ]

                # ------------------------------------------------
                # Target shapes
                # ------------------------------------------------

                check(
                    positive.shape == (height, width),
                    (
                        f"{level}: positive shape mismatch. "
                        f"Got {positive.shape}"
                    )
                )

                check(
                    ltrb.shape == (
                        height,
                        width,
                        4,
                    ),
                    (
                        f"{level}: ltrb shape mismatch. "
                        f"Got {ltrb.shape}"
                    )
                )

                check(
                    centerness.shape == (
                        height,
                        width,
                    ),
                    (
                        f"{level}: centerness shape mismatch. "
                        f"Got {centerness.shape}"
                    )
                )

                # =================================================
                # PREDICTION / TARGET COMPATIBILITY
                # =================================================

                check(
                    classification.shape[-2:]
                    == positive.shape,
                    (
                        f"{level}: classification/positive "
                        f"spatial mismatch."
                    )
                )

                check(
                    center.shape[-2:]
                    == centerness.shape,
                    (
                        f"{level}: center/centerness "
                        f"spatial mismatch."
                    )
                )

                check(
                    scale.shape[-2:]
                    == ltrb.shape[:2],
                    (
                        f"{level}: scale/ltrb "
                        f"spatial mismatch."
                    )
                )

                # ------------------------------------------------
                # Count positive locations
                # ------------------------------------------------

                num_positive = (
                    positive.sum().item()
                )

                image_positive_points += (
                    num_positive
                )

                if num_positive > 0:

                    image_levels.append(level)

                    level_positive_images[
                        level
                    ] += 1

                    level_positive_points[
                        level
                    ] += num_positive

                # ------------------------------------------------
                # Target validity checks
                # ------------------------------------------------

                positive_ltrb = ltrb[
                    positive
                ]

                positive_centerness = centerness[
                    positive
                ]

                if positive_ltrb.numel() > 0:

                    check(
                        torch.all(
                            positive_ltrb >= 0
                        ),
                        f"{level}: negative LTRB target."
                    )

                if positive_centerness.numel() > 0:

                    check(
                        torch.all(
                            positive_centerness >= 0
                        ),
                        (
                            f"{level}: centerness "
                            f"below zero."
                        )
                    )

                    check(
                        torch.all(
                            positive_centerness <= 1
                        ),
                        (
                            f"{level}: centerness "
                            f"above one."
                        )
                    )

                # ------------------------------------------------
                # Background targets
                # ------------------------------------------------

                background = ~positive

                if background.any():

                    background_ltrb = ltrb[
                        background
                    ]

                    background_centerness = (
                        centerness[
                            background
                        ]
                    )

                    check(
                        torch.all(
                            background_ltrb == 0
                        ),
                        (
                            f"{level}: background "
                            f"LTRB is not zero."
                        )
                    )

                    check(
                        torch.all(
                            background_centerness == 0
                        ),
                        (
                            f"{level}: background "
                            f"centerness is not zero."
                        )
                    )

            # ====================================================
            # PRINT IMAGE RESULT
            # ====================================================

            print(
                f"[{images_tested + 1:03d}/{NUM_IMAGES}] "
                f"GT boxes={num_boxes:2d} | "
                f"positive locations="
                f"{image_positive_points:3d} | "
                f"levels={image_levels}"
            )

            images_tested += 1

    # ========================================================
    # 6. DATASET STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(
        "Images tested:      ",
        images_tested
    )

    print(
        "Images with GT:     ",
        images_with_gt
    )

    print(
        "Images without GT:  ",
        images_without_gt
    )

    print(
        "Total GT boxes:     ",
        total_gt_boxes
    )

    if images_with_gt > 0:

        print(
            "Average boxes/positive image:",
            round(
                total_gt_boxes / images_with_gt,
                2,
            )
        )

    # ========================================================
    # 7. FPN STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("FPN LEVEL STATISTICS")
    print("=" * 60)

    for level, config in levels.items():

        print()
        print(level)

        print(
            "  stride:",
            config["stride"]
        )

        print(
            "  images with positives:",
            level_positive_images[level]
        )

        print(
            "  total positive points:",
            level_positive_points[level]
        )

    # ========================================================
    # 8. FINAL CHECKS
    # ========================================================

    check(
        images_tested == NUM_IMAGES,
        (
            f"Expected to test {NUM_IMAGES} images, "
            f"but tested {images_tested}."
        )
    )

    check(
        images_with_gt + images_without_gt
        == images_tested,
        "Image statistics are inconsistent."
    )

    print()
    print("=" * 60)
    print("MULTI-IMAGE FULL PIPELINE TEST PASSED!")
    print("=" * 60)

    print()
    print("Pipeline verified:")

    print(
        """
  Dataset
    ↓
  Transform
    ↓
  DataLoader
    ↓
  Ground-truth boxes
    ↓
  DetectionFramework
    ↓
  FPN P3-P7
    ↓
  Detection heads
    ↓
  Classification predictions
  Center predictions
  LTRB predictions
    ↓
  TargetGenerator
    ↓
  Positive / negative assignment
    ↓
  Regression range assignment
    ↓
  LTRB targets
    ↓
  Centerness targets
    ↓
  Prediction / target compatibility
"""
    )


if __name__ == "__main__":
    main()