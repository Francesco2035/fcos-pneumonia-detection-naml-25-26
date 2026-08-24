import torch

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.models.target_generator import (
    TargetGenerator,
)


# =========================================================
# OLD IMPLEMENTATION
# =========================================================

def old_generate_targets(
    generator,
    label_boxes,
    feature_shape,
    stride,
    device,
):
    """
    Original non-vectorized implementation.

    Used only to verify that the new vectorized
    implementation produces identical targets.
    """

    height, width = feature_shape

    positive = torch.zeros(
        (height, width),
        dtype=torch.bool,
        device=device,
    )

    ltrb = torch.zeros(
        (height, width, 4),
        dtype=torch.float32,
        device=device,
    )

    centerness = torch.zeros(
        (height, width),
        dtype=torch.float32,
        device=device,
    )

    # ---------------------------------------------------------
    # Original location-by-location implementation
    # ---------------------------------------------------------

    for y in range(height):

        for x in range(width):

            location = generator._convert_location(
                x,
                y,
                stride,
            )

            matching_boxes = (
                generator._find_matching_boxes(
                    label_boxes,
                    location,
                    stride,
                )
            )

            if len(matching_boxes) == 0:
                continue

            selected_box = (
                generator._select_box(
                    matching_boxes
                )
            )

            location_ltrb = (
                generator._calculate_ltrb(
                    selected_box,
                    location,
                )
            )

            location_centerness = (
                generator._calculate_centerness(
                    location_ltrb
                )
            )

            positive[y, x] = True

            ltrb[y, x] = torch.tensor(
                location_ltrb,
                dtype=torch.float32,
                device=device,
            )

            centerness[y, x] = (
                location_centerness
            )

    return {
        "positive": positive,
        "ltrb": ltrb,
        "centerness": centerness,
    }


# =========================================================
# TARGET COMPARISON
# =========================================================

def compare_targets(
    old_targets,
    new_targets,
):
    """
    Compare old and new target dictionaries.
    """

    # ---------------------------------------------------------
    # Positive mask
    # ---------------------------------------------------------

    positive_equal = torch.equal(
        old_targets["positive"],
        new_targets["positive"],
    )

    # ---------------------------------------------------------
    # LTRB
    # ---------------------------------------------------------

    ltrb_difference = torch.abs(
        old_targets["ltrb"]
        - new_targets["ltrb"]
    )

    max_ltrb_difference = (
        ltrb_difference.max().item()
        if ltrb_difference.numel() > 0
        else 0.0
    )

    ltrb_equal = torch.allclose(
        old_targets["ltrb"],
        new_targets["ltrb"],
        rtol=1e-5,
        atol=1e-5,
    )

    # ---------------------------------------------------------
    # Centerness
    # ---------------------------------------------------------

    centerness_difference = torch.abs(
        old_targets["centerness"]
        - new_targets["centerness"]
    )

    max_centerness_difference = (
        centerness_difference.max().item()
        if centerness_difference.numel() > 0
        else 0.0
    )

    centerness_equal = torch.allclose(
        old_targets["centerness"],
        new_targets["centerness"],
        rtol=1e-5,
        atol=1e-5,
    )

    return {
        "positive_equal": positive_equal,
        "ltrb_equal": ltrb_equal,
        "centerness_equal": centerness_equal,
        "max_ltrb_difference": max_ltrb_difference,
        "max_centerness_difference": (
            max_centerness_difference
        ),
    }


# =========================================================
# TEST
# =========================================================

def test_target_generator():

    print()
    print("=" * 60)
    print("TargetGenerator equivalence test")
    print("=" * 60)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] Device: {device}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=None,
    )

    generator = TargetGenerator()

    # ---------------------------------------------------------
    # Find positive images
    # ---------------------------------------------------------

    positive_indices = []

    for index in range(len(dataset)):

        patient_id = (
            dataset.image_paths[index].stem
        )

        boxes = dataset.annotations[
            patient_id
        ]["boxes"]

        if len(boxes) > 0:

            positive_indices.append(
                index
            )

        if len(positive_indices) >= 100:
            break

    if len(positive_indices) == 0:
        raise RuntimeError(
            "No positive images found."
        )

    print(
        f"[TEST] Positive images selected: "
        f"{len(positive_indices)}"
    )

    # ---------------------------------------------------------
    # All FPN levels
    # ---------------------------------------------------------

    levels = {
        8: "P3",
        16: "P4",
        32: "P5",
        64: "P6",
        128: "P7",
    }

    # ---------------------------------------------------------
    # Global statistics
    # ---------------------------------------------------------

    total_tests = 0
    failed_tests = 0

    max_ltrb_difference = 0.0
    max_centerness_difference = 0.0

    # =========================================================
    # Test each image
    # =========================================================

    for image_number, dataset_index in enumerate(
        positive_indices,
        start=1,
    ):

        patient_id = (
            dataset.image_paths[
                dataset_index
            ].stem
        )

        boxes = torch.tensor(
            dataset.annotations[
                patient_id
            ]["boxes"],
            dtype=torch.float32,
            device=device,
        )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        if (
            image_number % 10 == 0
            or image_number == len(positive_indices)
        ):
            print(
                f"[TEST] "
                f"Image {image_number}/"
                f"{len(positive_indices)}"
            )

        # =====================================================
        # Test all levels
        # =====================================================

        for stride, level_name in levels.items():

            total_tests += 1

            # -------------------------------------------------
            # Feature map dimensions
            # -------------------------------------------------

            height = IMAGE_SIZE // stride
            width = IMAGE_SIZE // stride

            feature_shape = (
                height,
                width,
            )

            # -------------------------------------------------
            # NEW vectorized implementation
            # -------------------------------------------------

            new_targets = (
                generator.generate_targets(
                    label_boxes=boxes,
                    feature_shape=feature_shape,
                    stride=stride,
                    device=device,
                )
            )

            # -------------------------------------------------
            # OLD implementation
            # -------------------------------------------------

            old_targets = (
                old_generate_targets(
                    generator=generator,
                    label_boxes=boxes,
                    feature_shape=feature_shape,
                    stride=stride,
                    device=device,
                )
            )

            # -------------------------------------------------
            # Compare
            # -------------------------------------------------

            result = compare_targets(
                old_targets,
                new_targets,
            )

            max_ltrb_difference = max(
                max_ltrb_difference,
                result["max_ltrb_difference"],
            )

            max_centerness_difference = max(
                max_centerness_difference,
                result[
                    "max_centerness_difference"
                ],
            )

            passed = (
                result["positive_equal"]
                and result["ltrb_equal"]
                and result["centerness_equal"]
            )

            if not passed:

                failed_tests += 1

                print()
                print(
                    "[FAIL]"
                )

                print(
                    f"Image: {dataset_index}"
                )

                print(
                    f"Patient ID: {patient_id}"
                )

                print(
                    f"Level: {level_name}"
                )

                print(
                    f"Positive equal: "
                    f"{result['positive_equal']}"
                )

                print(
                    f"LTRB equal: "
                    f"{result['ltrb_equal']}"
                )

                print(
                    f"Centerness equal: "
                    f"{result['centerness_equal']}"
                )

                print(
                    f"Max LTRB difference: "
                    f"{result['max_ltrb_difference']:.10f}"
                )

                print(
                    f"Max centerness difference: "
                    f"{result['max_centerness_difference']:.10f}"
                )

    # =========================================================
    # Final summary
    # =========================================================

    passed_tests = (
        total_tests
        - failed_tests
    )

    print()
    print("=" * 60)
    print("TargetGenerator equivalence summary")
    print("=" * 60)

    print(
        f"Positive images tested: "
        f"{len(positive_indices)}"
    )

    print(
        f"Levels tested:          "
        f"{len(levels)}"
    )

    print(
        f"Total comparisons:      "
        f"{total_tests}"
    )

    print(
        f"Passed:                 "
        f"{passed_tests}"
    )

    print(
        f"Failed:                 "
        f"{failed_tests}"
    )

    print(
        f"Max LTRB difference:     "
        f"{max_ltrb_difference:.10f}"
    )

    print(
        f"Max centerness diff:     "
        f"{max_centerness_difference:.10f}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Assertion
    # ---------------------------------------------------------

    if failed_tests > 0:

        raise AssertionError(
            "TargetGenerator equivalence test FAILED."
        )

    print(
        "TargetGenerator equivalence test PASSED."
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    test_target_generator()