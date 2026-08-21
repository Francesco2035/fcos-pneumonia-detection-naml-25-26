import torch

from src.models.target_generator import TargetGenerator


def test_target_generator_all_levels():

    generator = TargetGenerator()

    # =========================================================
    # Configurazione FPN
    # =========================================================

    levels = {
        "P3": {
            "shape": (28, 28),
            "stride": 8,
        },

        "P4": {
            "shape": (14, 14),
            "stride": 16,
        },

        "P5": {
            "shape": (7, 7),
            "stride": 32,
        },

        "P6": {
            "shape": (4, 4),
            "stride": 64,
        },

        "P7": {
            "shape": (2, 2),
            "stride": 128,
        },
    }

    print("\n==============================")
    print("TARGET GENERATOR TEST")
    print("==============================")

    # =========================================================
    # Test generale dei livelli
    # =========================================================

    boxes = torch.tensor(
        [
            [40.0, 40.0, 100.0, 100.0],
            [120.0, 120.0, 200.0, 200.0],
            [10.0, 10.0, 250.0, 250.0],
            [10.0, 10.0, 600.0, 600.0],
        ],
        dtype=torch.float32,
    )

    for level_name, config in levels.items():

        feature_shape = config["shape"]
        stride = config["stride"]

        print(f"\n----- {level_name} -----")

        targets = generator.generate_targets(
            label_boxes=boxes,
            feature_shape=feature_shape,
            stride=stride,
        )

        positive = targets["positive"]
        ltrb = targets["ltrb"]
        centerness = targets["centerness"]

        # -----------------------------------------------------
        # Shape checks
        # -----------------------------------------------------

        height, width = feature_shape

        assert positive.shape == (
            height,
            width,
        )

        assert ltrb.shape == (
            height,
            width,
            4,
        )

        assert centerness.shape == (
            height,
            width,
        )

        # -----------------------------------------------------
        # Positive locations
        # -----------------------------------------------------

        num_positive = positive.sum().item()

        print(
            "positive locations:",
            num_positive,
        )

        # -----------------------------------------------------
        # Centerness
        # -----------------------------------------------------

        positive_centerness = centerness[positive]

        if positive_centerness.numel() > 0:

            assert torch.all(
                positive_centerness >= 0
            )

            assert torch.all(
                positive_centerness <= 1
            )

        # -----------------------------------------------------
        # LTRB
        # -----------------------------------------------------

        positive_ltrb = ltrb[positive]

        if positive_ltrb.numel() > 0:

            assert torch.all(
                positive_ltrb >= 0
            )

        # -----------------------------------------------------
        # Background
        # -----------------------------------------------------

        background = ~positive

        if background.any():

            assert torch.all(
                ltrb[background] == 0
            )

            assert torch.all(
                centerness[background] == 0
            )

        print("shape checks: OK")
        print("centerness range: OK")
        print("LTRB positivity: OK")
        print("background targets: OK")

    # =========================================================
    # P3 known location test
    # =========================================================

    print("\n----- P3 known location test -----")

    p3_boxes = torch.tensor(
        [
            [40.0, 40.0, 100.0, 100.0],
        ],
        dtype=torch.float32,
    )

    targets_p3 = generator.generate_targets(
        label_boxes=p3_boxes,
        feature_shape=(28, 28),
        stride=8,
    )

    x = 7
    y = 7

    # Location:
    #
    # (7 + 0.5) * 8 = 60
    #

    assert targets_p3["positive"][y, x]

    expected_ltrb = torch.tensor(
        [20.0, 20.0, 40.0, 40.0]
    )

    assert torch.allclose(
        targets_p3["ltrb"][y, x],
        expected_ltrb,
    )

    expected_centerness = torch.tensor(0.5)

    assert torch.allclose(
        targets_p3["centerness"][y, x],
        expected_centerness,
        atol=1e-6,
    )

    print(
        "LTRB:",
        targets_p3["ltrb"][y, x],
    )

    print(
        "centerness:",
        targets_p3["centerness"][y, x].item(),
    )

    print("P3 known location: OK")

    # =========================================================
    # P6 specific test
    # =========================================================
    #
    # P6:
    #
    # stride = 64
    # regression range = [256, 512)
    #
    # Scegliamo la location (1, 1):
    #
    # x = (1 + 0.5) * 64 = 96
    # y = (1 + 0.5) * 64 = 96
    #
    # Box:
    #
    # [0, 0, 400, 400]
    #
    # LTRB:
    #
    # l = 96
    # t = 96
    # r = 304
    # b = 304
    #
    # max(LTRB) = 304
    #
    # quindi:
    #
    # 256 <= 304 < 512
    #
    # -> deve essere positivo su P6.
    # =========================================================

    print("\n----- P6 regression range test -----")

    p6_boxes = torch.tensor(
        [
            [0.0, 0.0, 400.0, 400.0],
        ],
        dtype=torch.float32,
    )

    targets_p6 = generator.generate_targets(
        label_boxes=p6_boxes,
        feature_shape=(4, 4),
        stride=64,
    )

    x = 1
    y = 1

    assert targets_p6["positive"][y, x]

    expected_ltrb = torch.tensor(
        [96.0, 96.0, 304.0, 304.0]
    )

    assert torch.allclose(
        targets_p6["ltrb"][y, x],
        expected_ltrb,
    )

    print(
        "location:",
        (x, y),
    )

    print(
        "LTRB:",
        targets_p6["ltrb"][y, x],
    )

    print(
        "max LTRB:",
        targets_p6["ltrb"][y, x].max().item(),
    )

    print("P6 regression range: OK")

    # =========================================================
    # P7 specific test
    # =========================================================
    #
    # P7:
    #
    # stride = 128
    # regression range = [512, infinity)
    #
    # Scegliamo la location (1, 1):
    #
    # x = (1 + 0.5) * 128 = 192
    # y = (1 + 0.5) * 128 = 192
    #
    # Box:
    #
    # [0, 0, 800, 800]
    #
    # LTRB:
    #
    # l = 192
    # t = 192
    # r = 608
    # b = 608
    #
    # max(LTRB) = 608
    #
    # quindi:
    #
    # 608 >= 512
    #
    # -> deve essere positivo su P7.
    # =========================================================

    print("\n----- P7 regression range test -----")

    p7_boxes = torch.tensor(
        [
            [0.0, 0.0, 800.0, 800.0],
        ],
        dtype=torch.float32,
    )

    targets_p7 = generator.generate_targets(
        label_boxes=p7_boxes,
        feature_shape=(2, 2),
        stride=128,
    )

    x = 1
    y = 1

    assert targets_p7["positive"][y, x]

    expected_ltrb = torch.tensor(
        [192.0, 192.0, 608.0, 608.0]
    )

    assert torch.allclose(
        targets_p7["ltrb"][y, x],
        expected_ltrb,
    )

    print(
        "location:",
        (x, y),
    )

    print(
        "LTRB:",
        targets_p7["ltrb"][y, x],
    )

    print(
        "max LTRB:",
        targets_p7["ltrb"][y, x].max().item(),
    )

    print("P7 regression range: OK")

    # =========================================================
    # Test P6 NON deve essere positivo con box troppo piccolo
    # =========================================================
    #
    # P6 richiede:
    #
    # max(LTRB) >= 256
    #
    # Usiamo un box con max(LTRB) < 256.
    # =========================================================

    print("\n----- P6 rejection test -----")

    small_box = torch.tensor(
        [
            [0.0, 0.0, 200.0, 200.0],
        ],
        dtype=torch.float32,
    )

    targets_p6_rejected = generator.generate_targets(
        label_boxes=small_box,
        feature_shape=(4, 4),
        stride=64,
    )

    assert targets_p6_rejected["positive"].sum() == 0

    print(
        "P6 correctly rejects small box: OK"
    )

    # =========================================================
    # Test P7 NON deve essere positivo con box troppo piccolo
    # =========================================================

    print("\n----- P7 rejection test -----")

    targets_p7_rejected = generator.generate_targets(
        label_boxes=small_box,
        feature_shape=(2, 2),
        stride=128,
    )

    assert targets_p7_rejected["positive"].sum() == 0

    print(
        "P7 correctly rejects small box: OK"
    )

    # =========================================================
    # Overlapping boxes test
    # =========================================================

    print("\n----- Overlapping boxes test -----")

    overlapping_boxes = torch.tensor(
        [
            [40.0, 40.0, 140.0, 140.0],
            [60.0, 60.0, 110.0, 110.0],
        ],
        dtype=torch.float32,
    )

    targets_overlap = generator.generate_targets(
        label_boxes=overlapping_boxes,
        feature_shape=(28, 28),
        stride=8,
    )

    x = 10
    y = 10

    # Location = (84, 84)
    #
    # Entrambi i box la contengono.
    # Deve essere scelto quello piccolo:
    #
    # [60, 60, 110, 110]
    #
    # LTRB:
    #
    # [24, 24, 26, 26]

    assert targets_overlap["positive"][y, x]

    expected_ltrb = torch.tensor(
        [24.0, 24.0, 26.0, 26.0]
    )

    assert torch.allclose(
        targets_overlap["ltrb"][y, x],
        expected_ltrb,
    )

    print(
        "selected LTRB:",
        targets_overlap["ltrb"][y, x],
    )

    print("overlapping boxes: OK")

    # =========================================================
    # Fine
    # =========================================================

    print("\n==============================")
    print("ALL TESTS PASSED")
    print("==============================")


if __name__ == "__main__":
    test_target_generator_all_levels()