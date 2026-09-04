import argparse
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

import pretrain

mp.set_sharing_strategy(
    "file_system"
)


# ============================================================
# Project imports
# ============================================================

from src.config import (
    IMAGE_SIZE,
    CSV_PATH,
    TRAIN_DCM_PATH,
    BATCH_SIZE,
    TRAIN_NUM_WORKERS,
    VAL_NUM_WORKERS,
    VAL_RATIO,
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    USE_SCHEDULER,
    LR_STEP_SIZE,
    LR_GAMMA,
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    LOG_SCALARS,
    LOG_HISTOGRAMS,
    LOG_GRADIENTS,
    LOG_HPARAMS,
    HISTOGRAM_EVERY_N_EPOCHS,
    GRADIENT_EVERY_N_STEPS,
    NUM_EPOCHS,
    EXPERIMENTS_DIR,
)

from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)

from src.models.detector import (
    DetectionFramework,
)

from src.detection_loss import (
    DetectionLoss,
)

from src.models.target_generator import (
    TargetGenerator,
)

from src.inference import (
    DetectionPostProcessor,
)

from src.evaluate import (
    DetectionEvaluator,
)

from src.train import (
    Trainer,
)

from src.train_v2 import (
    TrainerV2,
)

from src.analysis.analyzer import (
    DetectionAnalyzer,
)

from src.analysis.visualizer import (
    DetectionVisualizer,
)

from src.analysis.io import (
    save_per_image_results,
    save_metrics,
)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "FCOS-like pneumonia detector."
        )
    )

    # ========================================================
    # Mode
    # ========================================================

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "pretrain",
            "train",
            "analyze",
            "visualize",
        ],
        default="train",
        help=(
            "Execution mode."
        ),
    )

    # ========================================================
    # Backbone
    # ========================================================

    parser.add_argument(
        "--backbone",
        type=str,
        choices=[
            "imagenet",
            "chest_xray",
        ],
        default="imagenet",
        help=(
            "Backbone initialization."
        ),
    )

    parser.add_argument(
        "--resnet-depth",
        type=int,
        choices=[
            50,
            101,
        ],
        default=50,
        help=(
            "ResNet depth."
        ),
    )

    # ========================================================
    # Training
    # ========================================================

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Experiment name used for training."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=(
            "Number of training epochs."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=(
            "Base learning rate."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Training/validation batch size."
        ),
    )

    parser.add_argument(
        "--freeze-resnet",
        type=int,
        default=0,
        help=(
            "Number of initial epochs during which "
            "the ResNet backbone remains frozen."
        ),
    )

    # ========================================================
    # Trainer selection
    # ========================================================

    parser.add_argument(
        "--trainer",
        type=str,
        choices=[
            "standard",
            "v2",
        ],
        default="standard",
        help=(
            "Training implementation."
        ),
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help=(
            "Warmup duration used by TrainerV2."
        ),
    )

    parser.add_argument(
        "--backbone-lr-factor",
        type=float,
        default=0.1,
        help=(
            "Backbone learning-rate factor used "
            "after unfreezing."
        ),
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help=(
            "EMA decay used by TrainerV2."
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help=(
            "Adam weight decay."
        ),
    )

    # ========================================================
    # Training checkpoints
    # ========================================================

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from the experiment's "
            "last checkpoint."
        ),
    )

    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help=(
            "Load detector weights only."
        ),
    )

    parser.add_argument(
        "--load-backbone-weights",
        type=str,
        default=None,
        help=(
            "Load backbone weights only."
        ),
    )

    # ========================================================
    # Pretraining
    # ========================================================

    parser.add_argument(
        "--pretrain-data-dir",
        type=str,
        default=(
            "/home/legion/shared/Projects/"
            "NAML_25-26/data/chest_xray"
        ),
        help=(
            "Root directory containing the "
            "Chest-Xray train/val/test folders."
        ),
    )

    parser.add_argument(
        "--pretrain-architecture",
        type=int,
        choices=[
            50,
            101,
        ],
        default=50,
        help=(
            "ResNet architecture used for pretraining."
        ),
    )

    parser.add_argument(
        "--pretrain-image-size",
        type=int,
        default=512,
        help=(
            "Image size used during pretraining."
        ),
    )

    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default=None,
        help=(
            "Optional existing ResNet checkpoint "
            "to fine-tune."
        ),
    )

    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=10,
        help=(
            "Number of pretraining epochs."
        ),
    )

    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=16,
        help=(
            "Pretraining batch size."
        ),
    )

    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=1e-5,
        help=(
            "Pretraining learning rate."
        ),
    )

    parser.add_argument(
        "--pretrain-weight-decay",
        type=float,
        default=1e-4,
        help=(
            "Pretraining weight decay."
        ),
    )

    parser.add_argument(
        "--pretrain-num-workers",
        type=int,
        default=8,
        help=(
            "Number of DataLoader workers used "
            "during pretraining."
        ),
    )

    parser.add_argument(
        "--pretrain-seed",
        type=int,
        default=42,
        help=(
            "Random seed used during pretraining."
        ),
    )

    parser.add_argument(
        "--pretrain-freeze-epochs",
        type=int,
        default=0,
        help=(
            "Number of initial epochs for which "
            "the pretraining backbone is frozen."
        ),
    )

    parser.add_argument(
        "--pretrain-output-dir",
        type=str,
        default="checkpoints/pretrain",
        help=(
            "Output directory for pretraining checkpoints."
        ),
    )

    parser.add_argument(
        "--pretrain-amp",
        action="store_true",
        help=(
            "Use CUDA automatic mixed precision "
            "during pretraining."
        ),
    )

    # ========================================================
    # Analysis
    # ========================================================

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Detector checkpoint used in analyze mode."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for analysis results."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Manual visualization threshold. "
            "If omitted, Youden calibration is used."
        ),
    )

    parser.add_argument(
        "--visualization-mode",
        type=str,
        choices=[
            "youden",
            "no_th",
        ],
        default="youden",
        help=(
            "Visualization mode. "
            "'youden' uses the model-specific tau* stored by a previous "
            "analysis; 'no_th' uses threshold 0.10 with NMS, redundancy "
            "suppression, and the detection cap disabled."
        ),
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=10,
        help=(
            "Maximum number of detections retained "
            "for visualization."
        ),
    )

    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.40,
        help=(
            "Overlap threshold for the visualization "
            "redundancy filter."
        ),
    )

    parser.add_argument(
        "--num-flow-images",
        type=int,
        default=6,
        help=(
            "Number of feature-flow examples."
        ),
    )

    return parser.parse_args()


# ============================================================
# Argument validation
# ============================================================

def validate_args(args):

    # --------------------------------------------------------
    # Pretraining
    # --------------------------------------------------------

    if args.mode == "pretrain":

        if args.pretrain_epochs < 1:
            raise ValueError(
                "--pretrain-epochs must be >= 1."
            )

        if args.pretrain_batch_size < 1:
            raise ValueError(
                "--pretrain-batch-size must be >= 1."
            )

        if args.pretrain_lr <= 0:
            raise ValueError(
                "--pretrain-lr must be > 0."
            )

        if args.pretrain_weight_decay < 0:
            raise ValueError(
                "--pretrain-weight-decay must be >= 0."
            )

        if args.pretrain_num_workers < 0:
            raise ValueError(
                "--pretrain-num-workers must be >= 0."
            )

        if args.pretrain_image_size < 1:
            raise ValueError(
                "--pretrain-image-size must be >= 1."
            )

        if args.pretrain_freeze_epochs < 0:
            raise ValueError(
                "--pretrain-freeze-epochs must be >= 0."
            )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    elif args.mode == "train":

        if args.experiment is None:

            raise ValueError(
                "--experiment is required "
                "in train mode."
            )

        if args.epochs < 1:

            raise ValueError(
                "--epochs must be >= 1."
            )

        if args.lr <= 0:

            raise ValueError(
                "--lr must be > 0."
            )

        if args.batch_size < 1:

            raise ValueError(
                "--batch-size must be >= 1."
            )

        if args.freeze_resnet < 0:

            raise ValueError(
                "--freeze-resnet must be >= 0."
            )

        if args.warmup_epochs < 0:

            raise ValueError(
                "--warmup-epochs must be >= 0."
            )

        if args.backbone_lr_factor <= 0:

            raise ValueError(
                "--backbone-lr-factor must be > 0."
            )

        if not (
            0.0
            < args.ema_decay
            < 1.0
        ):

            raise ValueError(
                "--ema-decay must be in (0, 1)."
            )

        if (
            args.resume
            and args.load_weights is not None
        ):

            raise ValueError(
                "--resume and --load-weights "
                "cannot be used together."
            )

        if (
            args.resume
            and args.load_backbone_weights
            is not None
        ):

            raise ValueError(
                "--resume and "
                "--load-backbone-weights "
                "cannot be used together."
            )

        if (
            args.load_weights is not None
            and args.load_backbone_weights
            is not None
        ):

            raise ValueError(
                "--load-weights and "
                "--load-backbone-weights "
                "cannot be used together."
            )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    elif args.mode in ("analyze", "visualize"):

        if args.checkpoint is None:

            raise ValueError(
                "--checkpoint is required "
                "in analyze mode."
            )

        if not os.path.isfile(
            args.checkpoint
        ):

            raise FileNotFoundError(
                "Detector checkpoint not found:\n"
                f"{args.checkpoint}"
            )

        if args.threshold is not None:

            if not (
                0.0
                <= args.threshold
                <= 1.0
            ):

                raise ValueError(
                    "--threshold must be in [0, 1]."
                )

        if args.max_detections < 1:

            raise ValueError(
                "--max-detections must be >= 1."
            )

        if args.overlap_threshold < 0:

            raise ValueError(
                "--overlap-threshold must be >= 0."
            )

        if args.num_flow_images < 0:

            raise ValueError(
                "--num-flow-images must be >= 0."
            )


# ============================================================
# Device
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Model information
# ============================================================

def _count_parameters(
    module,
):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def _count_trainable_parameters(
    module,
):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def count_model_parameters(
    model,
):

    backbone = (
        model.fpn.backbone
    )

    backbone_parameters = (
        _count_parameters(
            backbone
        )
    )

    backbone_trainable = (
        _count_trainable_parameters(
            backbone
        )
    )

    fpn_parameters = 0
    fpn_trainable = 0

    for (
        name,
        parameter,
    ) in model.fpn.named_parameters():

        if name.startswith(
            "backbone."
        ):
            continue

        fpn_parameters += (
            parameter.numel()
        )

        if parameter.requires_grad:

            fpn_trainable += (
                parameter.numel()
            )

    heads = {}
    heads_trainable = {}

    head_modules = {
        "P3": model.head3,
        "P4": model.head4,
        "P5": model.head5,
        "P6": model.head6,
        "P7": model.head7,
    }

    for (
        level,
        head,
    ) in head_modules.items():

        heads[level] = (
            _count_parameters(
                head
            )
        )

        heads_trainable[level] = (
            _count_trainable_parameters(
                head
            )
        )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "backbone": backbone_parameters,
        "backbone_trainable": backbone_trainable,
        "fpn": fpn_parameters,
        "fpn_trainable": fpn_trainable,
        "heads": heads,
        "heads_trainable": heads_trainable,
    }


def print_model_information(
    model,
):

    counts = count_model_parameters(
        model
    )

    print()
    print(
        "=" * 60
    )

    print(
        "Model parameters"
    )

    print(
        "=" * 60
    )

    print(
        f"Backbone: "
        f"{counts['backbone']:,} "
        f"({counts['backbone'] / 1e6:.2f} M) "
        f"trainable="
        f"{counts['backbone_trainable']:,}"
    )

    print(
        f"FPN:      "
        f"{counts['fpn']:,} "
        f"({counts['fpn'] / 1e6:.2f} M) "
        f"trainable="
        f"{counts['fpn_trainable']:,}"
    )

    print()
    print(
        "Detection heads:"
    )

    for level in (
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
    ):

        parameters = (
            counts["heads"][level]
        )

        trainable = (
            counts["heads_trainable"][level]
        )

        print(
            f"  {level}: "
            f"{parameters:,} "
            f"({parameters / 1e6:.2f} M) "
            f"trainable={trainable:,}"
        )

    print(
        "-" * 60
    )

    print(
        f"Total:    "
        f"{counts['total']:,} "
        f"({counts['total'] / 1e6:.2f} M)"
    )

    print(
        f"Trainable: "
        f"{counts['trainable']:,} "
        f"({counts['trainable'] / 1e6:.2f} M)"
    )

    print(
        "=" * 60
    )


# ============================================================
# Backbone initialization
# ============================================================

def resolve_backbone_path(
    backbone,
):

    if backbone == "imagenet":

        return None

    if backbone == "chest_xray":

        from src.config import (
            RESNET50_CHEST_XRAY_CHECKPOINT,
        )

        path = (
            RESNET50_CHEST_XRAY_CHECKPOINT
        )

        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                "Chest-Xray pretrained backbone "
                f"not found:\n{path}"
            )

        return path

    raise ValueError(
        f"Unsupported backbone: "
        f"{backbone}"
    )


# ============================================================
# Pretraining
# ============================================================

def run_pretraining(
    args,
):
    """
    Adapt the main CLI arguments to the pretraining module.

    The actual pretraining implementation remains in pretrain.py.
    """

    from argparse import Namespace

    pretrain_args = Namespace(
        architecture=(
            args.pretrain_architecture
        ),

        data_dir=(
            args.pretrain_data_dir
        ),

        image_size=(
            args.pretrain_image_size
        ),

        weights=(
            args.pretrain_weights
        ),

        epochs=(
            args.pretrain_epochs
        ),

        batch_size=(
            args.pretrain_batch_size
        ),

        lr=(
            args.pretrain_lr
        ),

        weight_decay=(
            args.pretrain_weight_decay
        ),

        num_workers=(
            args.pretrain_num_workers
        ),

        seed=(
            args.pretrain_seed
        ),

        freeze_backbone_epochs=(
            args.pretrain_freeze_epochs
        ),

        output_dir=(
            args.pretrain_output_dir
        ),

        amp=(
            args.pretrain_amp
        ),
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CHEST-XRAY PRETRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"[LOG] Architecture: "
        f"ResNet-{args.pretrain_architecture}"
    )

    print(
        f"[LOG] Data: "
        f"{args.pretrain_data_dir}"
    )

    print(
        f"[LOG] Output: "
        f"{args.pretrain_output_dir}"
    )

    print(
        "=" * 70
    )

    pretrain.run_pretraining(
        pretrain_args
    )


# ============================================================
# Training
# ============================================================

def run_training(
    args,
    device,
):

    print()
    print(
        "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    # ---------------------------------------------------------
    # Experiment directories
    # ---------------------------------------------------------

    experiment_dir = (
        Path(
            EXPERIMENTS_DIR
        )
        / args.experiment
    )

    checkpoint_dir = (
        experiment_dir
    )

    log_dir = (
        experiment_dir
        / "tensorboard"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    resume_checkpoint = None

    if args.resume:

        resume_checkpoint = (
            checkpoint_dir
            / "last.pt"
        )

        if not resume_checkpoint.is_file():

            raise FileNotFoundError(
                "Resume requested but checkpoint "
                f"does not exist:\n"
                f"{resume_checkpoint}"
            )

        resume_checkpoint = str(
            resume_checkpoint
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print(
        "[LOG] Creating datasets..."
    )

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

    # ---------------------------------------------------------
    # Backbone
    # ---------------------------------------------------------

    path_model = (
        resolve_backbone_path(
            args.backbone
        )
    )

    print()
    print(
        "[LOG] Backbone:"
    )

    print(
        f"      {args.backbone}"
    )

    print(
        f"      ResNet-{args.resnet_depth}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print()
    print(
        "[LOG] Creating detector..."
    )

    model = (
        DetectionFramework(
            path_model=path_model,
            resnet_depth=args.resnet_depth,
        )
        .to(device)
    )

    # ---------------------------------------------------------
    # Load detector weights
    # ---------------------------------------------------------

    if args.load_weights is not None:

        checkpoint_path = Path(
            args.load_weights
        )

        if not checkpoint_path.is_file():

            raise FileNotFoundError(
                "Detector checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if (
            not isinstance(
                checkpoint,
                dict,
            )
            or
            "model_state_dict"
            not in checkpoint
        ):

            raise RuntimeError(
                "Invalid detector checkpoint."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        del checkpoint

        print(
            "[LOG] Detector weights loaded."
        )

    print_model_information(
        model
    )

    # ---------------------------------------------------------
    # Loss / target generation
    # ---------------------------------------------------------

    criterion = (
        DetectionLoss()
    )

    target_generator = (
        TargetGenerator()
    )

    # ---------------------------------------------------------
    # Postprocessor
    # ---------------------------------------------------------

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=(
                SCORE_THRESHOLD
            ),
            nms_threshold=(
                NMS_THRESHOLD
            ),
        )
    )

    # ---------------------------------------------------------
    # Evaluator
    # ---------------------------------------------------------

    evaluator = (
        DetectionEvaluator(
            model=model,
            postprocessor=postprocessor,
            device=device,
        )
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    effective_weight_decay = (
        WEIGHT_DECAY
        if args.weight_decay is None
        else args.weight_decay
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            effective_weight_decay
        ),
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    scheduler = None

    if USE_SCHEDULER:

        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=LR_STEP_SIZE,
                gamma=LR_GAMMA,
            )
        )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    common_trainer_args = {
        "resume": args.resume,
        "resume_checkpoint": (
            resume_checkpoint
        ),

        "train_dataset": train_dataset,
        "val_dataset": val_dataset,

        "model": model,
        "criterion": criterion,
        "target_generator": target_generator,
        "postprocessor": postprocessor,
        "evaluator": evaluator,

        "optimizer": optimizer,
        "scheduler": scheduler,

        "device": device,

        "batch_size": args.batch_size,
        "val_ratio": VAL_RATIO,
        "seed": SEED,

        "train_num_workers": (
            TRAIN_NUM_WORKERS
        ),

        "val_num_workers": (
            VAL_NUM_WORKERS
        ),

        "log_dir": str(log_dir),
        "checkpoint_dir": str(
            checkpoint_dir
        ),

        "log_scalars": LOG_SCALARS,
        "log_histograms": LOG_HISTOGRAMS,
        "log_gradients": LOG_GRADIENTS,
        "log_hparams": LOG_HPARAMS,

        "histogram_every_n_epochs": (
            HISTOGRAM_EVERY_N_EPOCHS
        ),

        "gradient_every_n_steps": (
            GRADIENT_EVERY_N_STEPS
        ),

        "freeze_resnet_epochs": (
            args.freeze_resnet
        ),
    }

    if args.trainer == "v2":

        print()
        print(
            "[LOG] Using TrainerV2."
        )

        trainer = TrainerV2(
            **common_trainer_args,

            base_lr=args.lr,

            backbone_lr_factor=(
                args.backbone_lr_factor
            ),

            warmup_epochs=(
                args.warmup_epochs
            ),

            ema_decay=(
                args.ema_decay
            ),

            use_ema=True,
        )

    else:

        print()
        print(
            "[LOG] Using standard Trainer."
        )

        trainer = Trainer(
            **common_trainer_args
        )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    trainer.train(
        num_epochs=args.epochs
    )


# ============================================================
# Analysis
# ============================================================

def run_analysis(
    args,
    device,
):

    print()
    print(
        "=" * 70
    )

    print(
        "DETECTOR ANALYSIS"
    )

    print(
        "=" * 70
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint
    )

    if args.output is not None:

        output_dir = Path(
            args.output
        )

    else:

        # Usually:
        #
        # experiments/<experiment>/last.pt
        #
        # becomes:
        #
        # visualization/<experiment>
        #
        experiment_name = (
            checkpoint_path
            .parent
            .name
        )

        output_dir = (
            Path("visualization")
            / experiment_name
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[LOG] Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"[LOG] Backbone: "
        f"{args.backbone}"
    )

    print(
        f"[LOG] ResNet: "
        f"{args.resnet_depth}"
    )

    print(
        f"[LOG] Output: "
        f"{output_dir}"
    )

    # ---------------------------------------------------------
    # Analyzer
    # ---------------------------------------------------------

    analyzer = (
        DetectionAnalyzer(
            checkpoint_path=(
                checkpoint_path
            ),

            backbone=args.backbone,

            resnet_depth=(
                args.resnet_depth
            ),

            device=device,

            output_dir=(
                output_dir
            ),

            max_detections=(
                args.max_detections
            ),

            overlap_threshold=(
                args.overlap_threshold
            ),

            manual_threshold=(
                args.threshold
            ),
        )
    )

    # ---------------------------------------------------------
    # Threshold calibration
    # ---------------------------------------------------------

    analyzer.calibrate_threshold()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    analysis = (
        analyzer.collect_validation_results()
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = (
        analyzer.compute_metrics(
            analysis
        )
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    save_per_image_results(
        results=analysis[
            "results"
        ],
        output_dir=output_dir,
    )

    save_metrics(
        metrics=metrics,
        tau=analyzer.tau,
        visualization_threshold=(
            analyzer.visualization_threshold
        ),
        output_dir=output_dir,
    )

    # ---------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------

    visualizer = (
        DetectionVisualizer(
            output_dir=output_dir,
            num_flow_images=(
                args.num_flow_images
            ),
        )
    )

    visualizer.save_confusion_matrix(
        metrics
    )

    visualizer.save_feature_flow_examples(
        model=analyzer.model,
        results=analysis[
            "results"
        ],
    )

    visualizer.save_all_prediction_images(
        results=analysis[
            "results"
        ],
        threshold=(
            analyzer.visualization_threshold
        ),
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ANALYSIS COMPLETED"
    )

    print(
        f"Results saved to: "
        f"{output_dir}"
    )

    print(
        "=" * 70
    )


# ============================================================
# Visualization-only mode
# ============================================================

def run_visualization(
    args,
    device,
):
    """
    Visualization-only execution.

    youden:
        Reads the previously computed model-specific tau* from metrics.json
        and generates the same clean prediction images as normal analysis,
        without recalibrating or computing metrics.

    no_th:
        Uses a fixed 0.10 score threshold, disables NMS and redundancy
        suppression, and shows all retained candidate detections.
    """

    print()
    print("=" * 70)
    print("DETECTOR VISUALIZATION ONLY")
    print("=" * 70)

    checkpoint_path = Path(
        args.checkpoint
    )

    if args.output is not None:
        output_dir = Path(
            args.output
        )
    else:
        experiment_name = (
            checkpoint_path.parent.name
        )

        output_dir = (
            Path("visualization")
            / experiment_name
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analyzer = DetectionAnalyzer(
        checkpoint_path=checkpoint_path,
        backbone=args.backbone,
        resnet_depth=args.resnet_depth,
        device=device,
        output_dir=output_dir,
        max_detections=args.max_detections,
        overlap_threshold=args.overlap_threshold,
        manual_threshold=None,
    )

    # ---------------------------------------------------------
    # Dense / no-th visualization
    # ---------------------------------------------------------

    if args.visualization_mode == "no_th":

        visualization_threshold = 0.10

        visualization_dir = (
            output_dir
            / "no_th"
        )

        results = (
            analyzer.collect_low_threshold_visualization_results(
                threshold=visualization_threshold
            )
        )

    # ---------------------------------------------------------
    # Clean / Youden visualization
    # ---------------------------------------------------------

    else:

        metrics_path = (
            output_dir
            / "metrics"
            / "metrics.json"
        )

        if not metrics_path.is_file():
            raise FileNotFoundError(
                "metrics.json not found. "
                "Run the normal analyze mode first so tau* is stored."
                f"\nExpected: {metrics_path}"
            )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(
                file
            )

        visualization_threshold = float(
            metrics[
                "visualization_threshold"
            ]
        )

        analyzer.visualization_threshold = (
            visualization_threshold
        )

        # Reuse the standard analysis validation path so the normal
        # filtering behaviour is exactly preserved. No calibration is run.
        analysis = (
            analyzer.collect_validation_results()
        )

        results = analysis[
            "results"
        ]

        visualization_dir = (
            output_dir
            / "youden"
        )

    print()
    print(
        "[LOG] Visualization mode: "
        f"{args.visualization_mode}"
    )

    print(
        "[LOG] Visualization threshold: "
        f"{visualization_threshold:.3f}"
    )

    print(
        "[LOG] Output: "
        f"{visualization_dir}"
    )

    visualizer = DetectionVisualizer(
        output_dir=visualization_dir,
        num_flow_images=0,
    )

    visualizer.save_all_prediction_images(
        results=results,
        threshold=visualization_threshold,
    )

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETED")
    print(
        f"Images saved to: {visualization_dir}"
    )
    print("=" * 70)




# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    validate_args(
        args
    )

    device = get_device()

    print()
    print(
        f"[LOG] Device: {device}"
    )

    if args.mode == "pretrain":

        run_pretraining(
            args
        )

    elif args.mode == "train":

        run_training(
            args=args,
            device=device,
        )

    elif args.mode == "analyze":

        run_analysis(
            args=args,
            device=device,
        )

    elif args.mode == "visualize":

        run_visualization(
            args=args,
            device=device,
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()