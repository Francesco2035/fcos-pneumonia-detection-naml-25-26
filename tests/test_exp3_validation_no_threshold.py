import os
import time

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
    NMS_THRESHOLD,
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

from src.inference import (
    DetectionPostProcessor,
)

from src.metrics import (
    compute_metrics,
)


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/exp3/best.pt"
)

SCORE_THRESHOLD = 0.0


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
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

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
        "exp3 validation diagnostics "
        "WITHOUT score threshold"
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

    print(
        f"[TEST] Score threshold: "
        f"{SCORE_THRESHOLD}"
    )

    print(
        f"[TEST] NMS threshold: "
        f"{NMS_THRESHOLD}"
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

    print(
        f"[TEST] Dataset size: "
        f"{len(train_dataset)}"
    )

    print(
        f"[TEST] Validation images: "
        f"{len(val_indices)}"
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
    # POSTPROCESSOR
    # ========================================================

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=0.0,
            nms_threshold=NMS_THRESHOLD,
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    all_predictions = []
    all_targets = []

    total_images = 0
    images_with_detections = 0
    total_detections = 0

    max_detection_score = 0.0

    all_scores = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for batch_idx, (
            images,
            targets,
        ) in enumerate(
            val_loader,
            start=1,
        ):

            images = images.to(
                device
            )

            predictions = model(
                images
            )

            detections = (
                postprocessor(
                    predictions
                )
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            for detection in detections:

                total_images += 1

                num_detections = (
                    detection[
                        "boxes"
                    ].shape[0]
                )

                total_detections += (
                    num_detections
                )

                if num_detections > 0:

                    images_with_detections += 1

                    scores = (
                        detection[
                            "scores"
                        ]
                        .detach()
                        .cpu()
                    )

                    all_scores.extend(
                        scores.tolist()
                    )

                    max_detection_score = max(
                        max_detection_score,
                        scores.max().item(),
                    )

                all_predictions.append(
                    {
                        "boxes":
                            detection[
                                "boxes"
                            ].cpu(),

                        "scores":
                            detection[
                                "scores"
                            ].cpu(),

                        "labels":
                            detection[
                                "labels"
                            ].cpu(),
                    }
                )

            # ------------------------------------------------
            # Targets
            # ------------------------------------------------

            for target in targets:

                all_targets.append(
                    {
                        "boxes":
                            target[
                                "boxes"
                            ].cpu(),

                        "labels":
                            target[
                                "labels"
                            ].cpu(),
                    }
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

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                progress = (
                    100.0
                    * batch_idx
                    / len(val_loader)
                )

                print(
                    f"[VAL] "
                    f"batch={batch_idx}/"
                    f"{len(val_loader)} "
                    f"progress={progress:.1f}% "
                    f"time="
                    f"{elapsed / 60.0:.2f} min"
                )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    average_detections = (
        total_detections
        / max(total_images, 1)
    )

    detection_ratio = (
        images_with_detections
        / max(total_images, 1)
    )

    print()
    print("=" * 70)
    print(
        "Detection diagnostics"
    )
    print("-" * 70)

    print(
        f"Images evaluated:       "
        f"{total_images}"
    )

    print(
        f"Images with detections: "
        f"{images_with_detections} "
        f"({100.0 * detection_ratio:.2f}%)"
    )

    print(
        f"Total detections:       "
        f"{total_detections}"
    )

    print(
        f"Average detections/img: "
        f"{average_detections:.4f}"
    )

    print(
        f"Maximum detection score: "
        f"{max_detection_score:.8f}"
    )

    print(
        f"Validation time:        "
        f"{elapsed / 60.0:.2f} min"
    )

    # --------------------------------------------------------
    # Top 20 scores
    # --------------------------------------------------------

    if all_scores:

        all_scores.sort(
            reverse=True
        )

        print()
        print(
            "Top 20 detection scores:"
        )

        for index, score in enumerate(
            all_scores[:20],
            start=1,
        ):

            print(
                f"  {index:02d}: "
                f"{score:.8f}"
            )

    else:

        print()
        print(
            "Top 20 detection scores:"
        )

        print(
            "  No detections."
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = compute_metrics(
        all_predictions,
        all_targets,
    )

    print()
    print("=" * 70)
    print(
        "Metrics WITHOUT score threshold"
    )
    print("-" * 70)

    print(
        f"AP:     "
        f"{metrics['AP']:.10f}"
    )

    print(
        f"AP_M:   "
        f"{metrics['AP_M']:.10f}"
    )

    print(
        f"AP_L:   "
        f"{metrics['AP_L']:.10f}"
    )

    print(
        f"AR@10:  "
        f"{metrics['AR@10']:.10f}"
    )

    print(
        f"AR_M:   "
        f"{metrics['AR_M']:.10f}"
    )

    print(
        f"AR_L:   "
        f"{metrics['AR_L']:.10f}"
    )

    print("=" * 70)

    print(
        "[TEST] No model parameters were modified."
    )

    print(
        "[TEST] No checkpoint was saved."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()