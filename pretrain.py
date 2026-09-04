from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.models.resnet import (
    ResNet50,
    ResNet101,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(
    seed: int,
):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Argument parsing
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Pretrain ResNet-50 or ResNet-101 on the "
            "Chest-Xray pneumonia classification dataset."
        )
    )

    # ---------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------

    parser.add_argument(
        "--architecture",
        type=int,
        choices=[
            50,
            101,
        ],
        default=50,
        help=(
            "ResNet architecture: 50 or 101. "
            "Default: 50."
        ),
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=str,
        default=(
            "/home/legion/shared/Projects/"
            "NAML_25-26/data/chest_xray"
        ),
        help=(
            "Root directory containing train/val/test."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help=(
            "Input image size."
        ),
    )

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Optional existing ResNet checkpoint "
            "to fine-tune."
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Number of training epochs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Training batch size."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help=(
            "Learning rate."
        ),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help=(
            "Adam weight decay."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help=(
            "Number of DataLoader workers."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed."
        ),
    )

    # ---------------------------------------------------------
    # Freezing
    # ---------------------------------------------------------

    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help=(
            "Freeze convolutional backbone for the "
            "first N epochs."
        ),
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/pretrain",
        help=(
            "Output directory for checkpoints."
        ),
    )

    # ---------------------------------------------------------
    # Mixed precision
    # ---------------------------------------------------------

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA automatic mixed precision."
        ),
    )

    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================

def build_datasets(
    data_dir,
    image_size,
):
    """
    Build train, validation and test datasets.

    Training uses mild augmentation.
    Validation and test use deterministic transforms.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.RandomHorizontalFlip(
                p=0.5,
            ),

            transforms.RandomRotation(
                degrees=7,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=os.path.join(
            data_dir,
            "train",
        ),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(
            data_dir,
            "val",
        ),
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(
            data_dir,
            "test",
        ),
        transform=eval_transform,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ============================================================
# DataLoaders
# ============================================================

def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
):
    """
    Build DataLoaders for train, validation and test sets.
    """

    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:

        common_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# Model
# ============================================================

def build_model(
    device,
    architecture,
    weights_path=None,
):
    """
    Build a ResNet-50 or ResNet-101 with a binary
    classification head.

    Classes:

        0 = NORMAL
        1 = PNEUMONIA
    """

    # ---------------------------------------------------------
    # Select architecture
    # ---------------------------------------------------------

    if architecture == 50:

        model = ResNet50(
            img_channels=3,
            num_classes=2,
        ).to(device)

    elif architecture == 101:

        model = ResNet101(
            img_channels=3,
            num_classes=2,
        ).to(device)

    else:

        raise ValueError(
            f"Unsupported architecture: "
            f"{architecture}. Use 50 or 101."
        )

    # ---------------------------------------------------------
    # Optional checkpoint loading
    # ---------------------------------------------------------

    if weights_path is not None:

        if not os.path.isfile(
            weights_path
        ):

            raise FileNotFoundError(
                "Weights checkpoint not found:\n"
                f"{weights_path}"
            )

        print()
        print(
            "[LOG] Loading existing ResNet checkpoint:"
        )

        print(
            f"      {weights_path}"
        )

        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=False,
        )

        # Accept common checkpoint formats.
        if isinstance(
            checkpoint,
            dict,
        ):

            if (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            elif (
                "state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "state_dict"
                    ]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        # -----------------------------------------------------
        # Try exact loading first.
        # -----------------------------------------------------

        try:

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            print(
                "[LOG] Full checkpoint loaded "
                "successfully."
            )

        except RuntimeError as error:

            print()
            print(
                "[WARN] Strict checkpoint loading failed."
            )

            print(
                "[WARN] Loading only compatible parameters."
            )

            current_state = (
                model.state_dict()
            )

            compatible = {}
            skipped = []

            for (
                key,
                value,
            ) in state_dict.items():

                if (
                    key in current_state
                    and
                    current_state[
                        key
                    ].shape == value.shape
                ):

                    compatible[key] = value

                else:

                    skipped.append(
                        key
                    )

            missing_parameters = [
                key
                for key in current_state
                if key not in compatible
            ]

            model.load_state_dict(
                compatible,
                strict=False,
            )

            print(
                "[LOG] Compatible parameters loaded: "
                f"{len(compatible)}"
            )

            print(
                "[LOG] Skipped parameters: "
                f"{len(skipped)}"
            )

            print(
                "[LOG] Missing parameters: "
                f"{len(missing_parameters)}"
            )

            print(
                "[LOG] Original loading error:"
            )

            print(
                str(error)
            )

        del checkpoint

    return model


# ============================================================
# Freeze / unfreeze
# ============================================================

def set_backbone_trainable(
    model,
    trainable,
):
    """
    Freeze or unfreeze the convolutional ResNet backbone.

    The final fully-connected classification head remains
    trainable.
    """

    # Initial convolution.
    for parameter in (
        model.conv1.parameters()
    ):

        parameter.requires_grad = (
            trainable
        )

    # ResNet residual stages.
    for layer in (
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ):

        for parameter in (
            layer.parameters()
        ):

            parameter.requires_grad = (
                trainable
            )

    # Classification head remains trainable.
    for parameter in (
        model.fc.parameters()
    ):

        parameter.requires_grad = True


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    amp_enabled,
):
    """
    Train the classification model for one epoch.
    """

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for (
        images,
        targets,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if amp_enabled:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    targets,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(
                images
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        all_targets.extend(
            targets
            .detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    return {
        "loss": float(
            epoch_loss
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
    }


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate the classification model on a dataset.
    """

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for (
        images,
        targets,
    ) in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            targets,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = (
            probabilities
            .argmax(
                dim=1
            )
        )

        all_targets.extend(
            targets
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .cpu()
            .tolist()
        )

        # Probability of the PNEUMONIA class.
        all_probabilities.extend(
            probabilities[
                :,
                1,
            ]
            .cpu()
            .tolist()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[
            0,
            1,
        ],
    )

    tn = int(
        cm[0, 0]
    )

    fp = int(
        cm[0, 1]
    )

    fn = int(
        cm[1, 0]
    )

    tp = int(
        cm[1, 1]
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    try:

        auc = roc_auc_score(
            all_targets,
            all_probabilities,
        )

    except ValueError:

        auc = 0.0

    return {
        "loss": float(
            loss
        ),
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "specificity": float(
            specificity
        ),
        "f1": float(
            f1
        ),
        "auc": float(
            auc
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    history,
):
    """
    Save the current pretraining state.
    """

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        "best_metric": best_metric,

        "history": history,
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# Pretraining
# ============================================================

def run_pretraining(
    args,
):
    """
    Run the complete Chest-Xray ResNet pretraining workflow.

    This function can be called from the project main.py,
    while the standalone main() below keeps pretrain.py
    executable on its own.
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    set_seed(
        args.seed
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    best_path = os.path.join(
        args.output_dir,
        "best.pt",
    )

    last_path = os.path.join(
        args.output_dir,
        "last.pt",
    )

    history_path = os.path.join(
        args.output_dir,
        "history.json",
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        args.amp
        and device.type == "cuda"
    )

    # ---------------------------------------------------------
    # Configuration summary
    # ---------------------------------------------------------

    print()
    print(
        "=" * 75
    )

    print(
        "CHEST-XRAY RESNET PRETRAINING"
    )

    print(
        "=" * 75
    )

    print(
        f"Architecture:    "
        f"ResNet-{args.architecture}"
    )

    print(
        f"Data:            "
        f"{args.data_dir}"
    )

    print(
        f"Image size:      "
        f"{args.image_size}"
    )

    print(
        f"Batch size:      "
        f"{args.batch_size}"
    )

    print(
        f"Epochs:          "
        f"{args.epochs}"
    )

    print(
        f"Learning rate:   "
        f"{args.lr:.2e}"
    )

    print(
        f"Weight decay:    "
        f"{args.weight_decay:.2e}"
    )

    print(
        f"Workers:         "
        f"{args.num_workers}"
    )

    print(
        f"Device:          "
        f"{device}"
    )

    print(
        f"AMP:             "
        f"{amp_enabled}"
    )

    print(
        f"Freeze epochs:   "
        f"{args.freeze_backbone_epochs}"
    )

    if args.weights is not None:

        print(
            f"Initial weights: "
            f"{args.weights}"
        )

    else:

        print(
            "Initial weights: NONE"
        )

    print(
        f"Output:          "
        f"{args.output_dir}"
    )

    print(
        "=" * 75
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print()
    print(
        "[LOG] Building datasets..."
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = build_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
    )

    print(
        f"[LOG] Train images: "
        f"{len(train_dataset)}"
    )

    print(
        f"[LOG] Validation images: "
        f"{len(val_dataset)}"
    )

    print(
        f"[LOG] Test images: "
        f"{len(test_dataset)}"
    )

    print(
        f"[LOG] Classes: "
        f"{train_dataset.classes}"
    )

    print(
        f"[LOG] Class mapping: "
        f"{train_dataset.class_to_idx}"
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print()
    print(
        f"[LOG] Creating ResNet-{args.architecture}..."
    )

    model = build_model(
        device=device,
        architecture=args.architecture,
        weights_path=args.weights,
    )

    # ---------------------------------------------------------
    # Freeze configuration
    # ---------------------------------------------------------

    if (
        args.freeze_backbone_epochs
        > 0
    ):

        set_backbone_trainable(
            model,
            trainable=False,
        )

        print(
            "[LOG] Backbone initially frozen "
            f"for {args.freeze_backbone_epochs} epochs."
        )

    else:

        set_backbone_trainable(
            model,
            trainable=True,
        )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )
    )

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # ---------------------------------------------------------
    # Training state
    # ---------------------------------------------------------

    history = []

    best_metric = -float(
        "inf"
    )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # -----------------------------------------------------
        # Unfreeze backbone
        # -----------------------------------------------------

        if (
            args.freeze_backbone_epochs
            > 0
            and
            epoch
            == args.freeze_backbone_epochs + 1
        ):

            print()
            print(
                "[LOG] Unfreezing ResNet backbone."
            )

            set_backbone_trainable(
                model,
                trainable=True,
            )

            # Rebuild optimizer so all parameters
            # are now updated.
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr * 0.5,
                weight_decay=args.weight_decay,
            )

            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(
                        args.epochs
                        - args.freeze_backbone_epochs
                    ),
                    eta_min=args.lr * 0.005,
                )
            )

        # -----------------------------------------------------
        # Current learning rate
        # -----------------------------------------------------

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
        )

        print()
        print(
            "=" * 75
        )

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.3e}"
        )

        print(
            "=" * 75
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        val_metrics = (
            evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        # -----------------------------------------------------
        # Scheduler
        # -----------------------------------------------------

        scheduler.step()

        # -----------------------------------------------------
        # Print training metrics
        # -----------------------------------------------------

        print(
            "[TRAIN]"
        )

        print(
            f"  loss:      "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"  accuracy:  "
            f"{train_metrics['accuracy']:.6f}"
        )

        print(
            f"  precision: "
            f"{train_metrics['precision']:.6f}"
        )

        print(
            f"  recall:    "
            f"{train_metrics['recall']:.6f}"
        )

        print(
            f"  F1:        "
            f"{train_metrics['f1']:.6f}"
        )

        # -----------------------------------------------------
        # Print validation metrics
        # -----------------------------------------------------

        print(
            "[VAL]"
        )

        print(
            f"  loss:       "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"  accuracy:   "
            f"{val_metrics['accuracy']:.6f}"
        )

        print(
            f"  precision:  "
            f"{val_metrics['precision']:.6f}"
        )

        print(
            f"  recall:     "
            f"{val_metrics['recall']:.6f}"
        )

        print(
            f"  specificity: "
            f"{val_metrics['specificity']:.6f}"
        )

        print(
            f"  F1:          "
            f"{val_metrics['f1']:.6f}"
        )

        print(
            f"  AUC:         "
            f"{val_metrics['auc']:.6f}"
        )

        print(
            f"  TP={val_metrics['tp']} "
            f"TN={val_metrics['tn']} "
            f"FP={val_metrics['fp']} "
            f"FN={val_metrics['fn']}"
        )

        # -----------------------------------------------------
        # History
        # -----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(
            epoch_record
        )

        # -----------------------------------------------------
        # Last checkpoint
        # -----------------------------------------------------

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            history=history,
        )

        # -----------------------------------------------------
        # Best checkpoint
        # -----------------------------------------------------

        selection_metric = (
            val_metrics["f1"]
        )

        if (
            selection_metric
            > best_metric
        ):

            best_metric = (
                selection_metric
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                history=history,
            )

            print()
            print(
                "[CHECKPOINT] NEW BEST"
            )

            print(
                f"[CHECKPOINT] "
                f"Validation F1 = "
                f"{best_metric:.6f}"
            )

            print(
                f"[CHECKPOINT] "
                f"Saved: {best_path}"
            )

    # ========================================================
    # Save history
    # ========================================================

    with open(
        history_path,
        "w",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    # ========================================================
    # Final test evaluation
    # ========================================================

    print()
    print(
        "=" * 75
    )

    print(
        "FINAL TEST EVALUATION"
    )

    print(
        "=" * 75
    )

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    del best_checkpoint

    test_metrics = (
        evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
    )

    print(
        f"Test loss:        "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test accuracy:    "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Test precision:   "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Test recall:      "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"Test specificity: "
        f"{test_metrics['specificity']:.6f}"
    )

    print(
        f"Test F1:          "
        f"{test_metrics['f1']:.6f}"
    )

    print(
        f"Test AUC:         "
        f"{test_metrics['auc']:.6f}"
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "architecture": (
            args.architecture
        ),

        "image_size": (
            args.image_size
        ),

        "data_dir": (
            args.data_dir
        ),

        "weights_initialization": (
            args.weights
        ),

        "epochs": (
            args.epochs
        ),

        "batch_size": (
            args.batch_size
        ),

        "learning_rate": (
            args.lr
        ),

        "weight_decay": (
            args.weight_decay
        ),

        "num_workers": (
            args.num_workers
        ),

        "seed": (
            args.seed
        ),

        "freeze_backbone_epochs": (
            args.freeze_backbone_epochs
        ),

        "best_validation_f1": (
            best_metric
        ),

        "test_metrics": (
            test_metrics
        ),

        "class_to_idx": (
            train_dataset.class_to_idx
        ),

        "best_checkpoint": (
            best_path
        ),

        "last_checkpoint": (
            last_path
        ),
    }

    metadata_path = os.path.join(
        args.output_dir,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # Completion
    # ========================================================

    print()
    print(
        "=" * 75
    )

    print(
        "PRETRAINING COMPLETED"
    )

    print(
        "=" * 75
    )

    print(
        f"Architecture:    "
        f"ResNet-{args.architecture}"
    )

    print(
        f"Best checkpoint: "
        f"{best_path}"
    )

    print(
        f"Last checkpoint: "
        f"{last_path}"
    )

    print(
        f"History:         "
        f"{history_path}"
    )

    print(
        f"Metadata:        "
        f"{metadata_path}"
    )

    print(
        "=" * 75
    )


# ============================================================
# Standalone entry point
# ============================================================

def main():

    args = parse_args()

    run_pretraining(
        args
    )


if __name__ == "__main__":
    main()