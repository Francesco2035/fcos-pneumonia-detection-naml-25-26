import argparse
import os

import torch
import torch.multiprocessing as mp

mp.set_sharing_strategy(
    "file_system"
)

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


# ============================================================
# Argument parsing
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train FCOS-like pneumonia detector."
        )
    )

    # ---------------------------------------------------------
    # Experiment
    # ---------------------------------------------------------

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=(
            "Experiment name, e.g. exp1."
        ),
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
            "ImageNet pretrained or "
            "chest-Xray pretrained."
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=(
            f"Number of training epochs. "
            f"Default: {NUM_EPOCHS}"
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=(
            f"Learning rate. "
            f"Default: {LEARNING_RATE}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            f"Training/validation batch size. "
            f"Default: {BATCH_SIZE}"
        ),
    )

    # ---------------------------------------------------------
    # ResNet freeze
    # ---------------------------------------------------------

    parser.add_argument(
        "--freeze-resnet",
        type=int,
        default=0,
        help=(
            "Freeze the ResNet backbone for the "
            "first N epochs. Default: 0."
        ),
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
    # Load complete detector weights only
    # ---------------------------------------------------------

    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help=(
            "Load only model weights from a detector "
            "checkpoint. Optimizer, scheduler, epoch, "
            "global step and best metric are NOT restored."
        ),
    )

    # ---------------------------------------------------------
    # Load ResNet backbone weights only
    # ---------------------------------------------------------

    parser.add_argument(
        "--load-backbone-weights",
        type=str,
        default=None,
        help=(
            "Load a ResNet classification checkpoint "
            "as the detector backbone. The checkpoint "
            "is passed directly to the detector Backbone."
        ),
    )

    return parser.parse_args()


# ============================================================
# Parameter helpers
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

    backbone = model.fpn.backbone

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


# ============================================================
# Print model information
# ============================================================

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

    print()


# ============================================================
# Load complete detector weights only
# ============================================================

def load_model_weights_only(
    model,
    checkpoint_path,
    device,
):
    """
    Load only model weights from a complete detector checkpoint.

    Does NOT restore:
        - optimizer
        - scheduler
        - epoch
        - global step
        - best metric
    """

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "Weights checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print(
        "[LOG] Loading ONLY detector model weights from:"
    )

    print(
        f"      {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict':\n"
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    del checkpoint

    print(
        "[LOG] Detector model weights "
        "loaded successfully."
    )

    print(
        "[LOG] Optimizer/scheduler/training state "
        "will start fresh."
    )


# ============================================================
# Prepare custom ResNet checkpoint for resume
# ============================================================

def prepare_resume_backbone_checkpoint(
    resume_checkpoint,
    output_path,
    device,
):
    """
    Build a temporary ResNet-only checkpoint from the complete
    FCOS resume checkpoint.

    The current Backbone implementation expects a checkpoint
    whose model_state_dict contains keys such as:

        conv1.weight
        bn1.weight
        layer1.0.conv1.weight
        ...
        fc.weight
        fc.bias

    The complete FCOS checkpoint contains these same weights
    under:

        fpn.backbone.model.*

    This helper extracts only those parameters and strips the
    prefix so that the existing Backbone class can construct
    the correct custom ResNet architecture.

    The temporary checkpoint is used ONLY to construct the model.
    The actual resume is still performed from the complete
    exp11/last.pt checkpoint by Trainer.
    """

    print()
    print(
        "[RESUME] Preparing ResNet architecture "
        "from the complete resume checkpoint..."
    )

    checkpoint = torch.load(
        resume_checkpoint,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Resume checkpoint must be a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Resume checkpoint does not contain "
            "'model_state_dict'."
        )

    state_dict = (
        checkpoint[
            "model_state_dict"
        ]
    )

    backbone_prefix = (
        "fpn.backbone.model."
    )

    backbone_state_dict = {}

    for (
        key,
        value,
    ) in state_dict.items():

        if key.startswith(
            backbone_prefix
        ):

            new_key = key[
                len(backbone_prefix):
            ]

            backbone_state_dict[
                new_key
            ] = value.cpu()

    del checkpoint

    if len(
        backbone_state_dict
    ) == 0:

        raise RuntimeError(
            "Could not extract ResNet backbone "
            "weights from resume checkpoint.\n"
            f"Expected keys starting with: "
            f"{backbone_prefix}"
        )

    required_keys = (
        "conv1.weight",
        "bn1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
        "layer4.0.conv1.weight",
    )

    missing_required = [
        key
        for key in required_keys
        if key not in backbone_state_dict
    ]

    if missing_required:

        raise RuntimeError(
            "Resume checkpoint does not contain the "
            "expected custom ResNet-50 structure.\n"
            f"Missing keys: {missing_required}"
        )

    torch.save(
        {
            "model_state_dict": (
                backbone_state_dict
            )
        },
        output_path,
    )

    print(
        "[RESUME] Extracted "
        f"{len(backbone_state_dict)} "
        "ResNet parameters."
    )

    print(
        "[RESUME] Temporary backbone checkpoint:"
    )

    print(
        f"         {output_path}"
    )

    return output_path


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ---------------------------------------------------------
    # Argument validation
    # ---------------------------------------------------------

    if args.resume and (
        args.load_weights is not None
        or
        args.load_backbone_weights is not None
    ):
        raise ValueError(
            "--resume cannot be used together with "
            "--load-weights or "
            "--load-backbone-weights."
        )

    if (
        args.load_weights is not None
        and
        args.load_backbone_weights is not None
    ):
        raise ValueError(
            "--load-weights and "
            "--load-backbone-weights "
            "cannot be used together."
        )

    if args.epochs < 1:
        raise ValueError(
            f"--epochs must be >= 1, "
            f"got {args.epochs}"
        )

    if args.lr <= 0:
        raise ValueError(
            f"--lr must be > 0, "
            f"got {args.lr}"
        )

    if args.batch_size < 1:
        raise ValueError(
            f"--batch-size must be >= 1, "
            f"got {args.batch_size}"
        )

    if args.freeze_resnet < 0:
        raise ValueError(
            f"--freeze-resnet must be >= 0, "
            f"got {args.freeze_resnet}"
        )

    # ========================================================
    # Experiment directories
    # ========================================================

    experiment_dir = os.path.join(
        EXPERIMENTS_DIR,
        args.experiment,
    )

    checkpoint_dir = (
        experiment_dir
    )

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

    # Temporary checkpoint used only when resuming.
    resume_backbone_checkpoint = os.path.join(
        checkpoint_dir,
        ".resume_backbone.pt",
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
    print(
        "=" * 60
    )

    print(
        "Training configuration"
    )

    print(
        "=" * 60
    )

    print(
        f"Experiment:          "
        f"{args.experiment}"
    )

    print(
        f"Backbone mode:       "
        f"{args.backbone}"
    )

    print(
        f"Epochs:              "
        f"{args.epochs}"
    )

    print(
        f"Batch size:          "
        f"{args.batch_size}"
    )

    print(
        f"Learning rate:       "
        f"{args.lr:.2e}"
    )

    print(
        f"Weight decay:        "
        f"{WEIGHT_DECAY:.2e}"
    )

    print(
        f"Scheduler:           "
        f"{'StepLR' if USE_SCHEDULER else 'disabled'}"
    )

    if USE_SCHEDULER:

        print(
            f"LR step size:       "
            f"{LR_STEP_SIZE}"
        )

        print(
            f"LR gamma:           "
            f"{LR_GAMMA}"
        )

    print(
        f"Freeze ResNet:       "
        f"{args.freeze_resnet} epochs"
    )

    if args.load_backbone_weights:

        print(
            f"Backbone checkpoint: "
            f"{args.load_backbone_weights}"
        )

    elif args.load_weights:

        print(
            f"Detector checkpoint: "
            f"{args.load_weights}"
        )

    print(
        "=" * 60
    )

    # ========================================================
    # Resume
    # ========================================================

    if args.resume:

        if not os.path.isfile(
            last_checkpoint
        ):
            raise FileNotFoundError(
                "Resume requested but checkpoint "
                f"does not exist:\n"
                f"{last_checkpoint}"
            )

        resume_checkpoint = (
            last_checkpoint
        )

        print()
        print(
            f"[LOG] Resuming experiment: "
            f"{args.experiment}"
        )

        print(
            f"[LOG] Resume checkpoint: "
            f"{resume_checkpoint}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We do NOT use:
        #
        #   RESNET50_CHEST_XRAY_CHECKPOINT
        #
        # and we do NOT use:
        #
        #   args.load_backbone_weights
        #
        # for resume.
        #
        # The complete exp11/last.pt is the source of truth.
        #
        # We temporarily extract the ResNet parameters from
        # last.pt only because the current Backbone class needs
        # a checkpoint in order to construct the custom ResNet.
        # ----------------------------------------------------

        path_model = (
            prepare_resume_backbone_checkpoint(
                resume_checkpoint=(
                    resume_checkpoint
                ),
                output_path=(
                    resume_backbone_checkpoint
                ),
                device=device,
            )
        )

        print(
            "[RESUME] No external ResNet checkpoint "
            "will be used."
        )

        print(
            "[RESUME] The complete model state will "
            "be restored from last.pt."
        )

    else:

        resume_checkpoint = None

        # ====================================================
        # Backbone selection for a NEW training run
        # ====================================================

        if (
            args.load_backbone_weights
            is not None
        ):

            path_model = (
                args.load_backbone_weights
            )

            if not os.path.isfile(
                path_model
            ):
                raise FileNotFoundError(
                    "Backbone checkpoint not found:\n"
                    f"{path_model}"
                )

            print()
            print(
                "[LOG] Starting fine-tuning "
                "from custom ResNet backbone."
            )

            print(
                f"[LOG] Initial backbone weights: "
                f"{path_model}"
            )

            print()
            print(
                "[LOG] Backbone initialization:"
            )

            print(
                "      Custom Chest-Xray "
                "ResNet-50"
            )

            print(
                "[LOG] Backbone checkpoint:"
            )

            print(
                f"      {path_model}"
            )

        elif args.backbone == "imagenet":

            path_model = None

            print()
            print(
                "[LOG] Starting new experiment:"
                f" {args.experiment}"
            )

            print()
            print(
                "[LOG] Backbone initialization:"
            )

            print(
                "      ImageNet pretrained "
                "ResNet-50"
            )

        elif args.backbone == "chest_xray":

            path_model = (
                RESNET50_CHEST_XRAY_CHECKPOINT
            )

            if not os.path.isfile(
                path_model
            ):
                raise FileNotFoundError(
                    "Chest-Xray pretrained backbone "
                    "not found:\n"
                    f"{path_model}"
                )

            print()
            print(
                "[LOG] Starting new experiment:"
                f" {args.experiment}"
            )

            print()
            print(
                "[LOG] Backbone initialization:"
            )

            print(
                "      Chest-Xray pretrained "
                "ResNet-50"
            )

            print(
                "[LOG] Backbone checkpoint:"
            )

            print(
                f"      {path_model}"
            )

        else:

            raise ValueError(
                f"Unsupported backbone: "
                f"{args.backbone}"
            )

    # ========================================================
    # Dataset
    # ========================================================

    print()
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

    print(
        f"[LOG] Dataset size: "
        f"{len(train_dataset)}"
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "[LOG] Creating model..."
    )

    model = (
        DetectionFramework(
            path_model=path_model,
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Load complete detector weights only
    #
    # This is NOT used for --resume.
    # For resume, Trainer restores the entire checkpoint.
    # --------------------------------------------------------

    if (
        not args.resume
        and
        args.load_weights
        is not None
    ):

        load_model_weights_only(
            model=model,
            checkpoint_path=args.load_weights,
            device=device,
        )

    # --------------------------------------------------------
    # For resume:
    #
    # At this point the architecture exists and has been
    # initialized using only a temporary ResNet extracted from
    # last.pt.
    #
    # Trainer will now restore the COMPLETE state from:
    #
    #     checkpoints/<experiment>/last.pt
    #
    # including model, optimizer, scheduler, epoch,
    # global step and best metric.
    # --------------------------------------------------------

    if args.resume:

        print()
        print(
            "[RESUME] Model architecture created."
        )

        print(
            "[RESUME] Complete checkpoint state will "
            "now be restored from:"
        )

        print(
            f"         {resume_checkpoint}"
        )

    print_model_information(
        model
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = (
        DetectionLoss()
    )

    # ========================================================
    # Target generator
    # ========================================================

    target_generator = (
        TargetGenerator()
    )

    # ========================================================
    # Postprocessor
    # ========================================================

    postprocessor = (
        DetectionPostProcessor(
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
        )
    )

    # ========================================================
    # Evaluator
    # ========================================================

    evaluator = (
        DetectionEvaluator(
            model=model,
            postprocessor=postprocessor,
            device=device,
        )
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
        )
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = None

    if USE_SCHEDULER:

        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=LR_STEP_SIZE,
                gamma=LR_GAMMA,
            )
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

        histogram_every_n_epochs=(
            HISTOGRAM_EVERY_N_EPOCHS
        ),

        gradient_every_n_steps=(
            GRADIENT_EVERY_N_STEPS
        ),

        freeze_resnet_epochs=(
            args.freeze_resnet
        ),
    )

    # ========================================================
    # Training
    # ========================================================

    trainer.train(
        num_epochs=args.epochs
    )

    # ========================================================
    # Cleanup
    # ========================================================

    if (
        args.resume
        and
        os.path.isfile(
            resume_backbone_checkpoint
        )
    ):

        try:
            os.remove(
                resume_backbone_checkpoint
            )

            print(
                "[RESUME] Temporary backbone "
                "checkpoint removed."
            )

        except OSError as exc:

            print(
                "[RESUME] Warning: could not remove "
                "temporary backbone checkpoint:"
            )

            print(
                f"         {exc}"
            )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()