import torch

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import get_test_transforms
from src.models.detector import DetectionFramework
from src.models.target_generator import TargetGenerator


# =========================================================
# Configuration
# =========================================================

IMAGE_SIZE = 224

STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64,
    "P7": 128,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

# Number of images to test
NUM_IMAGES = 2000


# =========================================================
# Utility
# =========================================================

def check(condition, message):

    if not condition:
        raise AssertionError(message)


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("MULTI-IMAGE FULL PIPELINE TEST")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Images to test: {NUM_IMAGES}")


    # =====================================================
    # 1. Dataset
    # =====================================================

    print("\n[1] Creating dataset...")

    transform = get_test_transforms(
        image_size=IMAGE_SIZE
    )

    dataset = RSNAPneumoniaDataset(
        dcm_path=DICOM_PATH,
        csv_path=CSV_PATH,
        transform=transform,
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    check(
        len(dataset) > 0,
        "Dataset is empty."
    )


    # =====================================================
    # 2. DataLoader
    # =====================================================

    print("\n[2] Creating DataLoader...")

    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


    # =====================================================
    # 3. Detection Framework
    # =====================================================

    print(
        "\n[3] Creating DetectionFramework..."
    )

    model = DetectionFramework(
        path_model=CHECKPOINT_PATH
    )

    model = model.to(DEVICE)

    model.eval()


    # =====================================================
    # 4. Target Generator
    # =====================================================

    print(
        "\n[4] Creating TargetGenerator..."
    )

    target_generator = TargetGenerator()


    # =====================================================
    # Statistics
    # =====================================================

    images_tested = 0

    positive_images = 0
    negative_images = 0

    total_gt_boxes = 0

    total_positive_locations = {
        "P3": 0,
        "P4": 0,
        "P5": 0,
        "P6": 0,
        "P7": 0,
    }

    images_with_positive_locations = {
        "P3": 0,
        "P4": 0,
        "P5": 0,
        "P6": 0,
        "P7": 0,
    }

    # Number of GT boxes assigned to each level.
    #
    # This is useful to understand whether the
    # regression ranges are behaving as expected.
    level_box_assignment = {
        "P3": 0,
        "P4": 0,
        "P5": 0,
        "P6": 0,
        "P7": 0,
    }


    # =====================================================
    # 5. Iterate over images
    # =====================================================

    print(
        f"\n[5] Testing {NUM_IMAGES} images..."
    )

    for images, targets in dataloader:

        if images_tested >= NUM_IMAGES:
            break

        # -------------------------------------------------
        # Batch
        # -------------------------------------------------

        check(
            images.shape[0] == 1,
            "This test expects batch_size=1."
        )

        check(
            images.ndim == 4,
            f"Expected [B,C,H,W], got {images.shape}"
        )

        check(
            images.shape[1] == 3,
            f"Expected 3 channels, got {images.shape[1]}"
        )

        check(
            images.shape[2] == IMAGE_SIZE
            and images.shape[3] == IMAGE_SIZE,
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"got {images.shape[2:]}"
        )


        # -------------------------------------------------
        # Ground truth
        # -------------------------------------------------

        target = targets[0]

        boxes = target["boxes"]

        labels = target["labels"]

        check(
            boxes.ndim == 2
            and boxes.shape[1] == 4,
            f"Expected boxes [N,4], got {boxes.shape}"
        )

        check(
            labels.ndim == 1,
            f"Expected labels [N], got {labels.shape}"
        )

        check(
            len(boxes) == len(labels),
            "Number of boxes and labels differ."
        )


        # -------------------------------------------------
        # Convert BoundingBoxes -> Tensor
        # -------------------------------------------------

        gt_boxes = boxes.as_subclass(
            torch.Tensor
        ).to(DEVICE)


        # -------------------------------------------------
        # Check GT coordinates
        # -------------------------------------------------

        if len(gt_boxes) > 0:

            x1 = gt_boxes[:, 0]
            y1 = gt_boxes[:, 1]
            x2 = gt_boxes[:, 2]
            y2 = gt_boxes[:, 3]

            check(
                torch.all(x2 >= x1),
                "Some boxes have x2 < x1."
            )

            check(
                torch.all(y2 >= y1),
                "Some boxes have y2 < y1."
            )

            check(
                torch.all(x1 >= 0)
                and torch.all(y1 >= 0),
                "Some boxes have negative coordinates."
            )

            check(
                torch.all(x2 <= IMAGE_SIZE)
                and torch.all(y2 <= IMAGE_SIZE),
                "Some boxes exceed image dimensions."
            )


        # -------------------------------------------------
        # Image statistics
        # -------------------------------------------------

        num_gt_boxes = len(gt_boxes)

        total_gt_boxes += num_gt_boxes

        if num_gt_boxes > 0:

            positive_images += 1

        else:

            negative_images += 1


        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        images_device = images.to(DEVICE)

        with torch.no_grad():

            outputs = model(
                images_device
            )


        # -------------------------------------------------
        # Generate targets
        # -------------------------------------------------

        image_positive_locations = 0

        image_positive_levels = []


        for level, stride in STRIDES.items():

            prediction = outputs[level]

            _, _, height, width = (
                prediction["center"].shape
            )


            # =============================================
            # Target generation
            # =============================================

            targets_level = (
                target_generator.generate_targets(
                    label_boxes=gt_boxes,
                    feature_shape=(height, width),
                    stride=stride,
                    device=DEVICE,
                )
            )


            positive = (
                targets_level["positive"]
            )

            ltrb = (
                targets_level["ltrb"]
            )

            centerness = (
                targets_level["centerness"]
            )


            # =============================================
            # Target shape checks
            # =============================================

            check(
                positive.shape
                == (height, width),
                f"{level}: positive shape mismatch."
            )

            check(
                ltrb.shape
                == (height, width, 4),
                f"{level}: LTRB shape mismatch."
            )

            check(
                centerness.shape
                == (height, width),
                f"{level}: centerness shape mismatch."
            )


            # =============================================
            # Prediction / target compatibility
            # =============================================

            center = (
                prediction["center"]
            )

            scale = (
                prediction["scale"]
            )

            check(
                center.shape[-2:]
                == positive.shape,
                f"{level}: center/positive mismatch."
            )

            check(
                scale.shape[-2:]
                == positive.shape,
                f"{level}: scale/positive mismatch."
            )


            # =============================================
            # Positive locations
            # =============================================

            num_positive = (
                positive.sum().item()
            )

            total_positive_locations[level] += (
                num_positive
            )


            if num_positive > 0:

                images_with_positive_locations[level] += 1

                image_positive_locations += (
                    num_positive
                )

                image_positive_levels.append(
                    level
                )


                # =========================================
                # Validate LTRB
                # =========================================

                positive_ltrb = (
                    ltrb[positive]
                )

                check(
                    torch.all(
                        positive_ltrb >= 0
                    ),
                    f"{level}: negative LTRB found."
                )


                # =========================================
                # Validate centerness
                # =========================================

                positive_centerness = (
                    centerness[positive]
                )

                check(
                    torch.all(
                        positive_centerness >= 0
                    ),
                    f"{level}: centerness < 0."
                )

                check(
                    torch.all(
                        positive_centerness <= 1
                    ),
                    f"{level}: centerness > 1."
                )


        # -------------------------------------------------
        # Positive / negative consistency
        # -------------------------------------------------

        if num_gt_boxes == 0:

            check(
                image_positive_locations == 0,
                (
                    f"Image {images_tested} has no GT "
                    f"boxes but TargetGenerator produced "
                    f"positive locations."
                )
            )

        else:

            check(
                image_positive_locations > 0,
                (
                    f"Image {images_tested} has "
                    f"{num_gt_boxes} GT boxes but "
                    f"TargetGenerator produced zero "
                    f"positive locations."
                )
            )


        # -------------------------------------------------
        # Count level assignments
        #
        # A level is considered used by the image if
        # at least one positive location exists there.
        # -------------------------------------------------

        for level in image_positive_levels:

            level_box_assignment[level] += 1


        # -------------------------------------------------
        # Print progress
        # -------------------------------------------------

        images_tested += 1

        print(
            f"[{images_tested:03d}/{NUM_IMAGES}] "
            f"GT boxes={num_gt_boxes:2d} | "
            f"positive locations="
            f"{image_positive_locations:3d} | "
            f"levels="
            f"{image_positive_levels}"
        )


    # =====================================================
    # 6. Dataset statistics
    # =====================================================

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(
        f"Images tested:       {images_tested}"
    )

    print(
        f"Images with GT:      {positive_images}"
    )

    print(
        f"Images without GT:   {negative_images}"
    )

    print(
        f"Total GT boxes:      {total_gt_boxes}"
    )

    if positive_images > 0:

        average_boxes = (
            total_gt_boxes
            / positive_images
        )

        print(
            f"Average boxes/positive image: "
            f"{average_boxes:.2f}"
        )


    # =====================================================
    # 7. FPN level statistics
    # =====================================================

    print("\n" + "=" * 60)
    print("FPN LEVEL STATISTICS")
    print("=" * 60)

    for level in STRIDES:

        print(
            f"\n{level}"
        )

        print(
            f"  stride:                 "
            f"{STRIDES[level]}"
        )

        print(
            f"  images with positives:  "
            f"{images_with_positive_locations[level]}"
        )

        print(
            f"  total positive points:  "
            f"{total_positive_locations[level]}"
        )


    # =====================================================
    # 8. Regression-range distribution
    # =====================================================

    print("\n" + "=" * 60)
    print("REGRESSION RANGE DISTRIBUTION")
    print("=" * 60)

    print(
        "\nImages receiving positive locations "
        "at each FPN level:"
    )

    for level in STRIDES:

        print(
            f"  {level}: "
            f"{level_box_assignment[level]} "
            f"/ {images_tested}"
        )


    # =====================================================
    # 9. Final checks
    # =====================================================

    check(
        images_tested > 0,
        "No images were tested."
    )

    check(
        positive_images > 0,
        "None of the tested images contains GT boxes."
    )

    check(
        negative_images > 0,
        "None of the tested images is a negative image."
    )

    total_positive_points = sum(
        total_positive_locations.values()
    )

    check(
        total_positive_points > 0,
        (
            "TargetGenerator produced zero "
            "positive locations over the entire test."
        )
    )


    # =====================================================
    # 10. Summary
    # =====================================================

    print("\n" + "=" * 60)
    print("MULTI-IMAGE FULL PIPELINE TEST PASSED!")
    print("=" * 60)

    print("\nPipeline verified:")

    print("  Dataset")
    print("    ↓")
    print("  Transform")
    print("    ↓")
    print("  DataLoader")
    print("    ↓")
    print("  Ground-truth boxes")
    print("    ↓")
    print("  DetectionFramework")
    print("    ↓")
    print("  FPN P3-P7")
    print("    ↓")
    print("  Detection heads")
    print("    ↓")
    print("  TargetGenerator")
    print("    ↓")
    print("  Positive / negative assignment")
    print("    ↓")
    print("  Regression range assignment")
    print("    ↓")
    print("  LTRB targets")
    print("    ↓")
    print("  Centerness targets")
    print("    ↓")
    print("  Prediction / target compatibility")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()