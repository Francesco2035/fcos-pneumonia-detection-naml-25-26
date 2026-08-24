import os

import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    VAL_NUM_WORKERS,
    VAL_RATIO,
    SEED,
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)

from src.datasets.split import (
    create_train_val_split,
)

from src.models.detector import (
    DetectionFramework,
)

from src.models.target_generator import (
    TargetGenerator,
)


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "checkpoints/exp3/best.pt"

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


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "Classification / centerness score diagnostics"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[TEST] Device: {device}"
    )

    print(
        f"[TEST] Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    train_dataset = (
        RSNAPneumoniaDataset(
            dcm_path=TRAIN_DCM_PATH,
            csv_path=CSV_PATH,
            transform=get_train_transforms(
                IMAGE_SIZE
            ),
        )
    )

    val_dataset = (
        RSNAPneumoniaDataset(
            dcm_path=TRAIN_DCM_PATH,
            csv_path=CSV_PATH,
            transform=get_test_transforms(
                IMAGE_SIZE
            ),
        )
    )

    # Reproduce the exact validation split.
    _, val_indices = (
        create_train_val_split(
            train_dataset,
            val_ratio=VAL_RATIO,
            seed=SEED,
        )
    )

    val_loader = (
        val_dataset.get_dataloader(
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=VAL_NUM_WORKERS,
            indices=val_indices,
        )
    )

    print(
        f"[TEST] Validation images: "
        f"{len(val_indices)}"
    )

    print(
        f"[TEST] Validation batches: "
        f"{len(val_loader)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = DetectionFramework(
        path_model=(
            RESNET50_CHEST_XRAY_CHECKPOINT
        ),
    ).to(device)

    load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    model.eval()

    print(
        "[TEST] Checkpoint loaded."
    )

    # ========================================================
    # TARGET GENERATOR
    # ========================================================

    target_generator = (
        TargetGenerator()
    )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    stats = {}

    for level in LEVELS:

        stats[level] = {
            "positive_count": 0,
            "negative_count": 0,

            "positive_cls_sum": 0.0,
            "negative_cls_sum": 0.0,

            "positive_ctr_sum": 0.0,
            "negative_ctr_sum": 0.0,

            "positive_score_sum": 0.0,
            "negative_score_sum": 0.0,

            "max_positive_cls": 0.0,
            "max_negative_cls": 0.0,

            "max_positive_ctr": 0.0,
            "max_negative_ctr": 0.0,

            "max_positive_score": 0.0,
            "max_negative_score": 0.0,

            "positive_cls_above_01": 0,
            "positive_cls_above_05": 0,
            "positive_score_above_01": 0,
            "positive_score_above_05": 0,
        }

    # ========================================================
    # VALIDATION
    # ========================================================

    with torch.no_grad():

        for batch_idx, (
            images,
            dataset_targets,
        ) in enumerate(
            val_loader,
            start=1,
        ):

            images = images.to(
                device
            )

            predictions = (
                model(images)
            )

            batch_size = (
                predictions[
                    "P3"
                ][
                    "classification"
                ].shape[0]
            )

            # ------------------------------------------------
            # Images in the batch
            # ------------------------------------------------

            for b in range(
                batch_size
            ):

                boxes = (
                    dataset_targets[b][
                        "boxes"
                    ].to(device)
                )

                # --------------------------------------------
                # Process every FPN level
                # --------------------------------------------

                for level in LEVELS:

                    classification = (
                        predictions[level][
                            "classification"
                        ][
                            b,
                            0,
                        ]
                    )

                    centerness = (
                        predictions[level][
                            "centerness"
                        ][
                            b,
                            0,
                        ]
                    )

                    _, _, H, W = (
                        predictions[level][
                            "classification"
                        ][
                            b:b + 1
                        ].shape
                    )

                    stride = STRIDES[
                        level
                    ]

                    # ----------------------------------------
                    # Generate targets using EXACT API
                    # of the current TargetGenerator.
                    # ----------------------------------------

                    targets = (
                        target_generator.generate_targets(
                            label_boxes=boxes,
                            feature_shape=(
                                H,
                                W,
                            ),
                            stride=stride,
                            device=device,
                        )
                    )

                    positive = (
                        targets[
                            "positive"
                        ]
                        .bool()
                    )

                    # ----------------------------------------
                    # Convert logits -> probabilities
                    # ----------------------------------------

                    cls_prob = torch.sigmoid(
                        classification
                    )

                    ctr_prob = torch.sigmoid(
                        centerness
                    )

                    final_score = (
                        cls_prob
                        * ctr_prob
                    )

                    s = stats[level]

                    # ----------------------------------------
                    # Positive locations
                    # ----------------------------------------

                    if positive.any():

                        cls_pos = (
                            cls_prob[
                                positive
                            ]
                        )

                        ctr_pos = (
                            ctr_prob[
                                positive
                            ]
                        )

                        score_pos = (
                            final_score[
                                positive
                            ]
                        )

                        s[
                            "positive_count"
                        ] += (
                            cls_pos.numel()
                        )

                        s[
                            "positive_cls_sum"
                        ] += (
                            cls_pos.sum().item()
                        )

                        s[
                            "positive_ctr_sum"
                        ] += (
                            ctr_pos.sum().item()
                        )

                        s[
                            "positive_score_sum"
                        ] += (
                            score_pos.sum().item()
                        )

                        s[
                            "max_positive_cls"
                        ] = max(
                            s[
                                "max_positive_cls"
                            ],
                            cls_pos.max().item(),
                        )

                        s[
                            "max_positive_ctr"
                        ] = max(
                            s[
                                "max_positive_ctr"
                            ],
                            ctr_pos.max().item(),
                        )

                        s[
                            "max_positive_score"
                        ] = max(
                            s[
                                "max_positive_score"
                            ],
                            score_pos.max().item(),
                        )

                        s[
                            "positive_cls_above_01"
                        ] += int(
                            (
                                cls_pos >= 0.1
                            ).sum().item()
                        )

                        s[
                            "positive_cls_above_05"
                        ] += int(
                            (
                                cls_pos >= 0.5
                            ).sum().item()
                        )

                        s[
                            "positive_score_above_01"
                        ] += int(
                            (
                                score_pos >= 0.1
                            ).sum().item()
                        )

                        s[
                            "positive_score_above_05"
                        ] += int(
                            (
                                score_pos >= 0.5
                            ).sum().item()
                        )

                    # ----------------------------------------
                    # Negative locations
                    # ----------------------------------------

                    negative = ~positive

                    if negative.any():

                        cls_neg = (
                            cls_prob[
                                negative
                            ]
                        )

                        ctr_neg = (
                            ctr_prob[
                                negative
                            ]
                        )

                        score_neg = (
                            final_score[
                                negative
                            ]
                        )

                        s[
                            "negative_count"
                        ] += (
                            cls_neg.numel()
                        )

                        s[
                            "negative_cls_sum"
                        ] += (
                            cls_neg.sum().item()
                        )

                        s[
                            "negative_ctr_sum"
                        ] += (
                            ctr_neg.sum().item()
                        )

                        s[
                            "negative_score_sum"
                        ] += (
                            score_neg.sum().item()
                        )

                        s[
                            "max_negative_cls"
                        ] = max(
                            s[
                                "max_negative_cls"
                            ],
                            cls_neg.max().item(),
                        )

                        s[
                            "max_negative_ctr"
                        ] = max(
                            s[
                                "max_negative_ctr"
                            ],
                            ctr_neg.max().item(),
                        )

                        s[
                            "max_negative_score"
                        ] = max(
                            s[
                                "max_negative_score"
                            ],
                            score_neg.max().item(),
                        )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx % 100 == 0
                or batch_idx == len(
                    val_loader
                )
            ):

                print(
                    f"[TEST] "
                    f"batch={batch_idx}/"
                    f"{len(val_loader)}"
                )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "Score component summary"
    )
    print("=" * 70)

    for level in LEVELS:

        s = stats[level]

        print()
        print(level)

        # ----------------------------------------------------
        # Positive locations
        # ----------------------------------------------------

        if s[
            "positive_count"
        ] > 0:

            n = s[
                "positive_count"
            ]

            pos_cls_mean = (
                s[
                    "positive_cls_sum"
                ] / n
            )

            pos_ctr_mean = (
                s[
                    "positive_ctr_sum"
                ] / n
            )

            pos_score_mean = (
                s[
                    "positive_score_sum"
                ] / n
            )

            print(
                "  POSITIVE"
            )

            print(
                f"    count:               "
                f"{n}"
            )

            print(
                f"    cls mean:            "
                f"{pos_cls_mean:.8f}"
            )

            print(
                f"    cls max:             "
                f"{s['max_positive_cls']:.8f}"
            )

            print(
                f"    centerness mean:     "
                f"{pos_ctr_mean:.8f}"
            )

            print(
                f"    centerness max:      "
                f"{s['max_positive_ctr']:.8f}"
            )

            print(
                f"    final score mean:    "
                f"{pos_score_mean:.8f}"
            )

            print(
                f"    final score max:     "
                f"{s['max_positive_score']:.8f}"
            )

            print(
                f"    cls >= 0.1:          "
                f"{s['positive_cls_above_01']}"
                f" / {n}"
                f" ({100.0 * s['positive_cls_above_01'] / n:.2f}%)"
            )

            print(
                f"    cls >= 0.5:          "
                f"{s['positive_cls_above_05']}"
                f" / {n}"
                f" ({100.0 * s['positive_cls_above_05'] / n:.2f}%)"
            )

            print(
                f"    final score >= 0.1:  "
                f"{s['positive_score_above_01']}"
                f" / {n}"
                f" ({100.0 * s['positive_score_above_01'] / n:.2f}%)"
            )

            print(
                f"    final score >= 0.5:  "
                f"{s['positive_score_above_05']}"
                f" / {n}"
                f" ({100.0 * s['positive_score_above_05'] / n:.2f}%)"
            )

        else:

            print(
                "  POSITIVE: none"
            )

        # ----------------------------------------------------
        # Negative locations
        # ----------------------------------------------------

        if s[
            "negative_count"
        ] > 0:

            n = s[
                "negative_count"
            ]

            neg_cls_mean = (
                s[
                    "negative_cls_sum"
                ] / n
            )

            neg_ctr_mean = (
                s[
                    "negative_ctr_sum"
                ] / n
            )

            neg_score_mean = (
                s[
                    "negative_score_sum"
                ] / n
            )

            print()
            print(
                "  NEGATIVE"
            )

            print(
                f"    count:               "
                f"{n}"
            )

            print(
                f"    cls mean:            "
                f"{neg_cls_mean:.8f}"
            )

            print(
                f"    cls max:             "
                f"{s['max_negative_cls']:.8f}"
            )

            print(
                f"    centerness mean:     "
                f"{neg_ctr_mean:.8f}"
            )

            print(
                f"    centerness max:      "
                f"{s['max_negative_ctr']:.8f}"
            )

            print(
                f"    final score mean:    "
                f"{neg_score_mean:.8f}"
            )

            print(
                f"    final score max:     "
                f"{s['max_negative_score']:.8f}"
            )

    print()
    print("=" * 70)
    print(
        "Interpretation"
    )
    print("=" * 70)

    print(
        "1. Positive classification much higher "
        "than negative -> classification is working."
    )

    print(
        "2. Positive centerness much higher "
        "than negative -> centerness is working."
    )

    print(
        "3. Positive final score still very low -> "
        "the product may be the bottleneck."
    )

    print(
        "4. Positive classification itself very low -> "
        "classification training is the bottleneck."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()