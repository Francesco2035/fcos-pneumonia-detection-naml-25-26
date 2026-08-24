import argparse
import os

import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

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
    RESNET50_CHEST_XRAY_CHECKPOINT,
)

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset

from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)

from src.models.detector import DetectionFramework
from src.models.target_generator import TargetGenerator
from src.detection_loss import DetectionLoss
from src.inference import DetectionPostProcessor
from src.evaluate import DetectionEvaluator
from src.train import Trainer


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FCOS-like pneumonia detector."
    )

    # ---------------------------------------------------------
    # Experiment
    # ---------------------------------------------------------

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name, e.g. exp1.",
    )

    # ---------------------------------------------------------
    # Backbone
    # ---------------------------------------------------------

    parser.add_argument(
        "--backbone",
        type=str,
        choices=[
            "imagenet",
            "chest_xray",
        ],
        default="imagenet",
        help=(
            "Backbone initialization: "
            "ImageNet pretrained or chest-X-ray pretrained."
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs. Default: {NUM_EPOCHS}",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate. Default: {LEARNING_RATE}",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training/validation batch size. Default: {BATCH_SIZE}",
    )

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from the last checkpoint "
            "of the selected experiment."
        ),
    )

    # ---------------------------------------------------------
    # Load only weights
    # ---------------------------------------------------------

    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help=(
            "Load only model weights from a checkpoint. "
            "Optimizer, scheduler, epoch, global step and "
            "best metric are NOT restored."
        ),
    )

    return parser.parse_args()


# ============================================================
# Parameter helpers
# ============================================================

def _count_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def _count_trainable_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def count_model_parameters(model):
    backbone = model.fpn.backbone

    backbone_parameters = _count_parameters(
        backbone
    )

    backbone_trainable = _count_trainable_parameters(
        backbone
    )

    fpn_parameters = 0
    fpn_trainable = 0

    for name, parameter in model.fpn.named_parameters():

        if name.startswith("backbone."):
            continue

        fpn_parameters += parameter.numel()

        if parameter.requires_grad:
            fpn_trainable += parameter.numel()

    heads = {}
    heads_trainable = {}

    head_modules = {
        "P3": model.head3,
        "P4": model.head4,
        "P5": model.head5,
        "P6": model.head6,
        "P7": model.head7,
    }

    for level, head in head_modules.items():

        heads[level] = _count_parameters(
            head
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


# ============================================================
# Print model information
# ============================================================

def print_model_information(model):

    counts = count_model_parameters(
        model
    )

    print()
    print("=" * 60)
    print("Model parameters")
    print("=" * 60)

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
    print("Detection heads:")

    for level in (
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
    ):

        parameters = counts["heads"][level]
        trainable = counts["heads_trainable"][level]

        print(
            f"  {level}: "
            f"{parameters:,} "
            f"({parameters / 1e6:.2f} M) "
            f"trainable={trainable:,}"
        )

    print("-" * 60)

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

    print("=" * 60)
    print()


# ============================================================
# Load model weights only
# ============================================================

def load_model_weights_only(
    model,
    checkpoint_path,
    device,
):
    """
    Load only model weights.

    Does NOT restore:
        - optimizer
        - scheduler
        - epoch
        - global step
        - best metric
    """

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Weights checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print(
        "[LOG] Loading ONLY model weights from:"
    )
    print(
        f"      {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain "
            f"'model_state_dict':\n{checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    del checkpoint

    print(
        "[LOG] Model weights loaded successfully."
    )
    print(
        "[LOG] Optimizer/scheduler/training state "
        "will start fresh."
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ---------------------------------------------------------
    # Argument validation
    # ---------------------------------------------------------

    if args.resume and args.load_weights is not None:
        raise ValueError(
            "--resume and --load-weights "
            "cannot be used together."
        )

    if args.epochs < 1:
        raise ValueError(
            f"--epochs must be >= 1, got {args.epochs}"
        )

    if args.lr <= 0:
        raise ValueError(
            f"--lr must be > 0, got {args.lr}"
        )

    if args.batch_size < 1:
        raise ValueError(
            f"--batch-size must be >= 1, "
            f"got {args.batch_size}"
        )

    # ========================================================
    # Experiment directories
    # ========================================================

    experiment_dir = os.path.join(
        EXPERIMENTS_DIR,
        args.experiment,
    )

    checkpoint_dir = experiment_dir

    log_dir = os.path.join(
        experiment_dir,
        "tensorboard",
    )

    os.makedirs(
        experiment_dir,
        exist_ok=True,
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    last_checkpoint = os.path.join(
        checkpoint_dir,
        "last.pt",
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"[LOG] Device: {device}"
    )

    # ========================================================
    # Effective configuration
    # ========================================================

    print()
    print("=" * 60)
    print("Training configuration")
    print("=" * 60)

    print(
        f"Experiment:      {args.experiment}"
    )

    print(
        f"Backbone:        {args.backbone}"
    )

    print(
        f"Epochs:          {args.epochs}"
    )

    print(
        f"Batch size:      {args.batch_size}"
    )

    print(
        f"Learning rate:   {args.lr:.2e}"
    )

    print(
        f"Weight decay:    {WEIGHT_DECAY:.2e}"
    )

    print(
        f"Scheduler:       "
        f"{'StepLR' if USE_SCHEDULER else 'disabled'}"
    )

    if USE_SCHEDULER:
        print(
            f"LR step size:   {LR_STEP_SIZE}"
        )
        print(
            f"LR gamma:       {LR_GAMMA}"
        )

    print("=" * 60)

    # ========================================================
    # Resume
    # ========================================================

    if args.resume:

        if not os.path.isfile(
            last_checkpoint
        ):
            raise FileNotFoundError(
                "Resume requested but checkpoint "
                f"does not exist:\n{last_checkpoint}"
            )

        resume_checkpoint = last_checkpoint

        print()
        print(
            f"[LOG] Resuming experiment: "
            f"{args.experiment}"
        )

        print(
            f"[LOG] Resume checkpoint: "
            f"{resume_checkpoint}"
        )

    else:

        resume_checkpoint = None

        if args.load_weights is not None:

            print()
            print(
                f"[LOG] Starting fine-tuning experiment: "
                f"{args.experiment}"
            )

            print(
                f"[LOG] Initial weights: "
                f"{args.load_weights}"
            )

        else:

            print()
            print(
                f"[LOG] Starting new experiment: "
                f"{args.experiment}"
            )

    # ========================================================
    # Backbone selection
    # ========================================================

    if args.backbone == "imagenet":

        path_model = None

        print(
            "[LOG] Backbone initialization: "
            "ImageNet pretrained ResNet-50"
        )

    elif args.backbone == "chest_xray":

        path_model = (
            RESNET50_CHEST_XRAY_CHECKPOINT
        )

        if not os.path.isfile(
            path_model
        ):
            raise FileNotFoundError(
                "Chest-X-ray pretrained backbone "
                f"not found:\n{path_model}"
            )

        print(
            "[LOG] Backbone initialization: "
            "Chest-X-ray pretrained ResNet-50"
        )

        print(
            f"[LOG] Backbone checkpoint: "
            f"{path_model}"
        )

    else:
        raise ValueError(
            f"Unsupported backbone: {args.backbone}"
        )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("[LOG] Creating datasets...")

    train_dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_train_transforms(
            IMAGE_SIZE
        ),
    )

    val_dataset = RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=get_test_transforms(
            IMAGE_SIZE
        ),
    )

    print(
        f"[LOG] Dataset size: "
        f"{len(train_dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print("[LOG] Creating model...")

    model = DetectionFramework(
        path_model=path_model,
    ).to(device)

    # --------------------------------------------------------
    # Load only model weights
    # --------------------------------------------------------

    if args.load_weights is not None:

        load_model_weights_only(
            model=model,
            checkpoint_path=args.load_weights,
            device=device,
        )

    print_model_information(
        model
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = DetectionLoss()

    # ========================================================
    # Target generator
    # ========================================================

    target_generator = TargetGenerator()

    # ========================================================
    # Postprocessor
    # ========================================================

    postprocessor = DetectionPostProcessor(
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )

    # ========================================================
    # Evaluator
    # ========================================================

    evaluator = DetectionEvaluator(
        model=model,
        postprocessor=postprocessor,
        device=device,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = None

    if USE_SCHEDULER:

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=LR_STEP_SIZE,
            gamma=LR_GAMMA,
        )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = Trainer(
        resume=args.resume,
        resume_checkpoint=resume_checkpoint,

        train_dataset=train_dataset,
        val_dataset=val_dataset,

        model=model,
        criterion=criterion,
        target_generator=target_generator,
        postprocessor=postprocessor,
        evaluator=evaluator,

        optimizer=optimizer,
        scheduler=scheduler,

        device=device,

        batch_size=args.batch_size,
        val_ratio=VAL_RATIO,
        seed=SEED,

        train_num_workers=TRAIN_NUM_WORKERS,
        val_num_workers=VAL_NUM_WORKERS,

        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,

        log_scalars=LOG_SCALARS,
        log_histograms=LOG_HISTOGRAMS,
        log_gradients=LOG_GRADIENTS,
        log_hparams=LOG_HPARAMS,

        histogram_every_n_epochs=HISTOGRAM_EVERY_N_EPOCHS,
        gradient_every_n_steps=GRADIENT_EVERY_N_STEPS,
    )

    # ========================================================
    # Training
    # ========================================================

    trainer.train(
        num_epochs=args.epochs
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()