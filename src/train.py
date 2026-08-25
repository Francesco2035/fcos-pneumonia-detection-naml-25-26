import os

import torch
from src.config import (
    LEVELS,
    STRIDES,
    LAST_CHECKPOINT_NAME,
    BEST_CHECKPOINT_NAME,
    SAVE_EVERY_N_EPOCHS,
)

from torch.utils.data import DataLoader, Subset
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)
from src.datasets.split import create_train_val_split

from src.models.target_generator import TargetGenerator
from src.detection_loss import DetectionLoss
from src.inference import DetectionPostProcessor
from src.evaluate import DetectionEvaluator


class Trainer:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        resume: bool,
        model: nn.Module,
        criterion: nn.Module,
        target_generator: TargetGenerator,
        postprocessor: DetectionPostProcessor,
        evaluator: DetectionEvaluator,
        optimizer: torch.optim.Optimizer,
        resume_checkpoint: str | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device | None = None,
        batch_size: int = 1,
        val_ratio: float = 0.2,
        seed: int = 42,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
        log_dir: str = "runs/experiment",
        checkpoint_dir: str = "checkpoints",

        # Logging configuration
        log_scalars: bool = True,
        log_histograms: bool = False,
        log_gradients: bool = False,
        log_hparams: bool = False,
        histogram_every_n_epochs: int = 5,
        gradient_every_n_steps: int = 100,

        # NEW:
        # Number of initial epochs during which the
        # ResNet backbone remains frozen.
        freeze_resnet_epochs: int = 0,
    ):
        # -----------------------------------------------------
        # Core components
        # -----------------------------------------------------

        self.model = model
        self.criterion = criterion
        self.target_generator = target_generator
        self.postprocessor = postprocessor
        self.evaluator = evaluator

        self.optimizer = optimizer
        self.scheduler = scheduler

        # -----------------------------------------------------
        # Device
        # -----------------------------------------------------

        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.model.to(self.device)

        # -----------------------------------------------------
        # Training configuration
        # -----------------------------------------------------

        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers

        # -----------------------------------------------------
        # Data
        # -----------------------------------------------------

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader, self.val_loader = (
            self._create_data_loaders()
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        self.writer = SummaryWriter(log_dir)

        self.log_scalars = log_scalars
        self.log_histograms = log_histograms
        self.log_gradients = log_gradients
        self.log_hparams = log_hparams

        self.histogram_every_n_epochs = (
            histogram_every_n_epochs
        )

        self.gradient_every_n_steps = (
            gradient_every_n_steps
        )

        # -----------------------------------------------------
        # Checkpoints
        # -----------------------------------------------------

        self.checkpoint_dir = checkpoint_dir

        # -----------------------------------------------------
        # Training state
        # -----------------------------------------------------

        self.current_epoch = 0
        self.global_step = 0

        self.best_metric = -float("inf")

        self.resume = resume
        self.resume_checkpoint = resume_checkpoint

        # -----------------------------------------------------
        # ResNet backbone freezing
        # -----------------------------------------------------

        if freeze_resnet_epochs < 0:
            raise ValueError(
                "freeze_resnet_epochs must be >= 0."
            )

        self.freeze_resnet_epochs = (
            freeze_resnet_epochs
        )

        # In the current detector implementation the
        # ResNet backbone is model.fpn.backbone.
        self.backbone = self.model.fpn.backbone

        if self.freeze_resnet_epochs > 0:
            self._set_backbone_trainable(
                False
            )

            self._set_backbone_eval()

            print()
            print(
                "[LOG] ResNet backbone frozen "
                f"for the first "
                f"{self.freeze_resnet_epochs} epochs."
            )

    # =========================================================
    # Data preparation
    # =========================================================

    def _create_data_loaders(self):

        # Create a reproducible train/validation split.
        train_indices, val_indices = (
            create_train_val_split(
                self.train_dataset,
                val_ratio=self.val_ratio,
                seed=self.seed,
            )
        )

        # Train loader:
        # - uses only train_indices
        # - shuffle=True
        # - training transforms come from train_dataset
        train_loader = (
            self.train_dataset.get_dataloader(
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.train_num_workers,
                indices=train_indices,
            )
        )

        # Validation loader:
        # - uses only val_indices
        # - shuffle=False
        # - validation transforms come from val_dataset
        val_loader = (
            self.val_dataset.get_dataloader(
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.val_num_workers,
                indices=val_indices,
            )
        )

        print(
            f"[LOG] Train samples: "
            f"{len(train_indices)}"
        )

        print(
            f"[LOG] Validation samples: "
            f"{len(val_indices)}"
        )

        print(
            "[LOG] Dataloaders created"
        )

        return train_loader, val_loader

    # =========================================================
    # ResNet freeze / unfreeze
    # =========================================================

    def _set_backbone_trainable(
        self,
        trainable: bool,
    ):
        """
        Freeze/unfreeze the ResNet backbone.

        FPN and detection heads are not affected.
        """

        for parameter in (
            self.backbone.parameters()
        ):
            parameter.requires_grad = (
                trainable
            )

    def _set_backbone_eval(
        self,
    ):
        """
        Keep BatchNorm/dropout-like stateful layers inside
        the frozen ResNet in evaluation mode.

        This prevents BatchNorm running statistics from
        changing while the backbone is frozen.
        """

        self.backbone.eval()

    def _set_backbone_train_mode(
        self,
    ):
        """
        Restore training mode for the ResNet backbone
        after unfreezing.
        """

        self.backbone.train()

    def _update_backbone_freeze(
        self,
        epoch: int,
    ):
        """
        Example with freeze_resnet_epochs=3:

            epoch 1 -> frozen
            epoch 2 -> frozen
            epoch 3 -> frozen
            epoch 4 -> unfrozen
        """

        if self.freeze_resnet_epochs <= 0:
            return

        if (
            epoch
            == self.freeze_resnet_epochs + 1
        ):

            self._set_backbone_trainable(
                True
            )

            self._set_backbone_train_mode()

            print()
            print(
                "[LOG] ResNet backbone "
                "UNFROZEN."
            )

            print(
                "[LOG] ResNet parameters are "
                "now trainable from "
                f"epoch {epoch}."
            )

    # =========================================================
    # Target generation
    # =========================================================

    def _build_targets(
        self,
        predictions,
        dataset_targets,
    ):
        """
        Build P3-P7 targets from the dataset ground-truth boxes.

        predictions is used just to know the feature map shape.
        """

        batch_targets = []

        batch_size = (
            predictions[
                LEVELS[0]
            ][
                "classification"
            ].shape[0]
        )

        for b in range(batch_size):

            image_targets = {}

            boxes = (
                dataset_targets[b]["boxes"]
                .to(self.device)
            )

            for level in LEVELS:

                stride = STRIDES[level]

                _, _, height, width = (
                    predictions[level][
                        "classification"
                    ].shape
                )

                target = (
                    self.target_generator.generate_targets(
                        label_boxes=boxes,
                        feature_shape=(
                            height,
                            width,
                        ),
                        stride=stride,
                        device=self.device,
                    )
                )

                image_targets[level] = (
                    target
                )

            batch_targets.append(
                image_targets
            )

        return batch_targets

    # =========================================================
    # One training epoch
    # =========================================================

    def train_one_epoch(self):
        """
        Run one complete training epoch.

        Returns:
            Dictionary containing epoch-level training statistics.
        """

        import time

        # -----------------------------------------------------
        # Model mode
        # -----------------------------------------------------

        self.model.train()

        # model.train() recursively sets the entire backbone
        # to train mode. If it is currently frozen, restore eval
        # mode for the backbone so BN running statistics do not
        # change.
        if (
            self.freeze_resnet_epochs > 0
            and
            self.current_epoch
            <= self.freeze_resnet_epochs
        ):
            self._set_backbone_eval()

        # -----------------------------------------------------
        # Epoch accumulators
        # -----------------------------------------------------

        total_loss = 0.0
        center_loss = 0.0
        regression_loss = 0.0
        centerness_loss = 0.0

        # Per-level loss
        level_loss = {
            level: 0.0
            for level in LEVELS
        }

        # Per-level positive locations
        level_positive = {
            level: 0.0
            for level in LEVELS
        }

        # -----------------------------------------------------
        # Timing
        # -----------------------------------------------------

        epoch_start = time.perf_counter()

        data_time = 0.0
        forward_time = 0.0
        target_time = 0.0
        backward_time = 0.0

        batch_start = time.perf_counter()

        window_batches = 0

        # -----------------------------------------------------
        # Training loop
        # -----------------------------------------------------

        for batch_idx, (
            images,
            dataset_targets,
        ) in enumerate(
            self.train_loader,
            start=1,
        ):

            # -------------------------------------------------
            # Data loading time
            # -------------------------------------------------

            current_time = (
                time.perf_counter()
            )

            data_time += (
                current_time
                - batch_start
            )

            # -------------------------------------------------
            # Move images to device
            # -------------------------------------------------

            images = images.to(
                self.device
            )

            # -------------------------------------------------
            # Reset gradients
            # -------------------------------------------------

            self.optimizer.zero_grad()

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------

            forward_start = (
                time.perf_counter()
            )

            predictions = self.model(
                images
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            forward_time += (
                time.perf_counter()
                - forward_start
            )

            # -------------------------------------------------
            # Target generation
            # -------------------------------------------------

            target_start = (
                time.perf_counter()
            )

            targets = self._build_targets(
                predictions,
                dataset_targets,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            target_time += (
                time.perf_counter()
                - target_start
            )

            # -------------------------------------------------
            # Compute detection loss
            # -------------------------------------------------

            loss_start = (
                time.perf_counter()
            )

            losses = self.criterion(
                predictions,
                targets,
            )

            # Tensor required for backward().
            loss = losses["total"]

            # -------------------------------------------------
            # Accumulate global losses
            # -------------------------------------------------

            total_loss += (
                losses["total"].item()
            )

            center_loss += (
                losses["center"].item()
            )

            regression_loss += (
                losses["regression"].item()
            )

            centerness_loss += (
                losses["centerness"].item()
            )

            # -------------------------------------------------
            # Accumulate per-level statistics
            # -------------------------------------------------

            for level in LEVELS:

                level_results = (
                    losses["levels"][level]
                )

                for result in (
                    level_results
                ):

                    level_loss[level] += (
                        result[
                            "total"
                        ].item()
                    )

                    level_positive[level] += (
                        result[
                            "num_positive"
                        ].item()
                    )

            # -------------------------------------------------
            # Backward + optimizer
            # -------------------------------------------------

            loss.backward()

            # -------------------------------------------------
            # Gradient clipping
            # -------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            backward_time += (
                time.perf_counter()
                - loss_start
            )

            # -------------------------------------------------
            # Global training step
            # -------------------------------------------------

            self.global_step += 1

            # -------------------------------------------------
            # Optional gradient logging
            # -------------------------------------------------

            self._log_gradient_histograms()

            # -------------------------------------------------
            # Update timing window
            # -------------------------------------------------

            window_batches += 1

            batch_start = (
                time.perf_counter()
            )

            # -------------------------------------------------
            # Progress logging
            # -------------------------------------------------

            if (
                batch_idx % 100 == 0
                or
                batch_idx
                == len(self.train_loader)
            ):

                window_time = (
                    data_time
                    + forward_time
                    + target_time
                    + backward_time
                )

                progress = (
                    100.0
                    * batch_idx
                    / len(self.train_loader)
                )

                processed_images = min(
                    batch_idx
                    * self.batch_size,
                    len(
                        self.train_loader
                        .dataset
                    ),
                )

                elapsed_epoch = (
                    time.perf_counter()
                    - epoch_start
                )

                images_per_second = (
                    processed_images
                    / elapsed_epoch
                    if elapsed_epoch > 0
                    else 0.0
                )

                if window_batches > 0:

                    avg_data = (
                        data_time
                        / window_batches
                    )

                    avg_forward = (
                        forward_time
                        / window_batches
                    )

                    avg_target = (
                        target_time
                        / window_batches
                    )

                    avg_backward = (
                        backward_time
                        / window_batches
                    )

                else:

                    avg_data = 0.0
                    avg_forward = 0.0
                    avg_target = 0.0
                    avg_backward = 0.0

                print(
                    f"\n[TRAIN] "
                    f"epoch={self.current_epoch} "
                    f"batch={batch_idx}/"
                    f"{len(self.train_loader)} "
                    f"images={processed_images}/"
                    f"{len(self.train_loader.dataset)} "
                    f"progress={progress:.1f}%"
                )

                print(
                    f"        "
                    f"loss={loss.item():.6f} "
                    f"speed={images_per_second:.2f} img/s"
                )

                print(
                    f"        "
                    f"avg batch time: "
                    f"{window_time / window_batches:.4f}s"
                )

                print(
                    f"        "
                    f"data loading:   "
                    f"{avg_data:.4f}s"
                )

                print(
                    f"        "
                    f"forward:        "
                    f"{avg_forward:.4f}s"
                )

                print(
                    f"        "
                    f"target build:   "
                    f"{avg_target:.4f}s"
                )

                print(
                    f"        "
                    f"backward+step:  "
                    f"{avg_backward:.4f}s"
                )

                # Reset timing window.
                data_time = 0.0
                forward_time = 0.0
                target_time = 0.0
                backward_time = 0.0
                window_batches = 0

        # -----------------------------------------------------
        # Convert accumulated sums to epoch averages
        # -----------------------------------------------------

        num_batches = len(
            self.train_loader
        )

        total_loss /= num_batches
        center_loss /= num_batches
        regression_loss /= num_batches
        centerness_loss /= num_batches

        for level in LEVELS:

            level_loss[level] /= (
                num_batches
            )

            level_positive[level] /= (
                num_batches
            )

        # -----------------------------------------------------
        # Epoch statistics
        # -----------------------------------------------------

        stats = {
            "total_loss": total_loss,
            "center_loss": center_loss,
            "regression_loss": regression_loss,
            "centerness_loss": centerness_loss,
            "level_loss": level_loss,
            "level_positive": level_positive,
        }

        # -----------------------------------------------------
        # Epoch logging
        # -----------------------------------------------------

        self._log_training_losses(
            stats
        )

        self._log_parameter_histograms()

        # -----------------------------------------------------
        # Epoch summary
        # -----------------------------------------------------

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"\n[TRAIN] Epoch "
            f"{self.current_epoch} completed "
            f"in {epoch_time / 60.0:.2f} min"
        )

        return stats

    # =========================================================
    # Logging
    # =========================================================

    def _log_training_losses(
        self,
        stats,
    ):
        """
        Log epoch-level training statistics.
        """

        if not self.log_scalars:
            return

        epoch = self.current_epoch

        # Global losses
        self.writer.add_scalar(
            "train/total_loss",
            stats["total_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "train/center_loss",
            stats["center_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "train/regression_loss",
            stats["regression_loss"],
            epoch,
        )

        self.writer.add_scalar(
            "train/centerness_loss",
            stats["centerness_loss"],
            epoch,
        )

        # Per-level statistics
        for level in LEVELS:

            self.writer.add_scalar(
                f"train/{level}/loss",
                stats["level_loss"][level],
                epoch,
            )

            self.writer.add_scalar(
                f"train/{level}/positive",
                stats["level_positive"][level],
                epoch,
            )

        # Learning rate
        self.writer.add_scalar(
            "train/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            epoch,
        )

    def _log_validation_metrics(
        self,
        metrics,
    ):
        """
        Log epoch-level validation metrics.
        """

        if not self.log_scalars:
            return

        epoch = self.current_epoch

        self.writer.add_scalar(
            "val/AP",
            metrics["AP"],
            epoch,
        )

        self.writer.add_scalar(
            "val/AP_M",
            metrics["AP_M"],
            epoch,
        )

        self.writer.add_scalar(
            "val/AP_L",
            metrics["AP_L"],
            epoch,
        )

        self.writer.add_scalar(
            "val/AR@10",
            metrics["AR@10"],
            epoch,
        )

        self.writer.add_scalar(
            "val/AR_M",
            metrics["AR_M"],
            epoch,
        )

        self.writer.add_scalar(
            "val/AR_L",
            metrics["AR_L"],
            epoch,
        )

        # Extra logging for the freezing schedule.
        self.writer.add_scalar(
            "train/resnet_frozen",
            float(
                self._is_backbone_frozen()
            ),
            epoch,
        )

    def _is_backbone_frozen(
        self,
    ):
        return not any(
            parameter.requires_grad
            for parameter
            in self.backbone.parameters()
        )

    def _log_parameter_histograms(self):
        """
        Log parameter distributions for selected model components.
        """

        if not self.log_histograms:
            return

        if (
            self.current_epoch
            % self.histogram_every_n_epochs
            != 0
        ):
            return

        epoch = self.current_epoch

        for name, parameter in (
            self.model.named_parameters()
        ):

            if not parameter.requires_grad:
                continue

            name_lower = name.lower()

            # Log selected backbone parameters.
            if "backbone" in name_lower:

                self.writer.add_histogram(
                    f"weights/{name}",
                    parameter.detach().cpu(),
                    epoch,
                )

            # Log FPN parameters.
            elif "fpn" in name_lower:

                self.writer.add_histogram(
                    f"weights/{name}",
                    parameter.detach().cpu(),
                    epoch,
                )

            # Log detection heads.
            elif "head" in name_lower:

                self.writer.add_histogram(
                    f"weights/{name}",
                    parameter.detach().cpu(),
                    epoch,
                )

    def _log_gradient_histograms(self):
        """
        Log gradient distributions for model parameters.
        """

        if not self.log_gradients:
            return

        if (
            self.global_step
            % self.gradient_every_n_steps
            != 0
        ):
            return

        step = self.global_step

        for name, parameter in (
            self.model.named_parameters()
        ):

            if parameter.grad is None:
                continue

            name_lower = name.lower()

            if (
                "backbone" in name_lower
                or
                "fpn" in name_lower
                or
                "head" in name_lower
            ):

                self.writer.add_histogram(
                    f"gradients/{name}",
                    parameter.grad.detach().cpu(),
                    step,
                )

    # =========================================================
    # Validation
    # =========================================================

    def validate(self):
        """
        Run validation and compute detection metrics.
        """

        self.model.eval()

        with torch.no_grad():

            metrics = (
                self.evaluator.evaluate(
                    self.val_loader
                )
            )

        self._log_validation_metrics(
            metrics
        )

        return metrics

    # =========================================================
    # Checkpointing
    # =========================================================

    def save_checkpoint(
        self,
        path,
    ):
        """
        Save the complete training state.
        """

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "best_metric": self.best_metric,
            "freeze_resnet_epochs": (
                self.freeze_resnet_epochs
            ),
        }

        if self.scheduler is not None:
            checkpoint[
                "scheduler_state_dict"
            ] = (
                self.scheduler.state_dict()
            )

        torch.save(
            checkpoint,
            path,
        )

    def load_checkpoint(
        self,
        path,
    ):
        """
        Restore the complete training state.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            self.scheduler is not None
            and
            "scheduler_state_dict"
            in checkpoint
        ):

            self.scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        self.current_epoch = (
            checkpoint["epoch"]
        )

        self.global_step = (
            checkpoint["global_step"]
        )

        self.best_metric = (
            checkpoint["best_metric"]
        )

        print(
            f"[LOG] Checkpoint loaded: "
            f"{path}"
        )

        print(
            f"[LOG] Resuming from epoch: "
            f"{self.current_epoch}"
        )

        # Restore the correct backbone state after loading.
        if (
            self.freeze_resnet_epochs > 0
            and
            self.current_epoch
            <= self.freeze_resnet_epochs
        ):

            self._set_backbone_trainable(
                False
            )

            self._set_backbone_eval()

            print(
                "[LOG] Resumed state: "
                "ResNet remains frozen."
            )

        else:

            self._set_backbone_trainable(
                True
            )

            self._set_backbone_train_mode()

            print(
                "[LOG] Resumed state: "
                "ResNet is trainable."
            )

    # =========================================================
    # Training loop
    # =========================================================

    def train(
        self,
        num_epochs,
    ):
        """
        Run the complete training process.

        One epoch consists of:
            1. training
            2. validation
            3. scheduler update
            4. checkpointing
        """

        if num_epochs < 1:
            raise ValueError(
                "num_epochs must be >= 1."
            )

        # -----------------------------------------------------
        # Prepare checkpoint directory
        # -----------------------------------------------------

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Resume previous training
        # -----------------------------------------------------

        if self.resume:

            if self.resume_checkpoint is None:
                raise ValueError(
                    "resume=True but no resume "
                    "checkpoint was provided."
                )

            if not os.path.isfile(
                self.resume_checkpoint
            ):
                raise FileNotFoundError(
                    "Resume checkpoint not found: "
                    f"{self.resume_checkpoint}"
                )

            self.load_checkpoint(
                self.resume_checkpoint
            )

            if (
                self.current_epoch
                >= num_epochs
            ):

                print(
                    "[LOG] Training already "
                    "completed up to epoch "
                    f"{self.current_epoch}."
                )

                self.writer.flush()

                return

        # -----------------------------------------------------
        # Epoch loop
        # -----------------------------------------------------

        for epoch in range(
            self.current_epoch + 1,
            num_epochs + 1,
        ):

            self.current_epoch = epoch

            # -------------------------------------------------
            # Backbone freeze / unfreeze schedule
            # -------------------------------------------------

            self._update_backbone_freeze(
                epoch
            )

            print()
            print(
                "=" * 60
            )

            print(
                f"Epoch {epoch}/{num_epochs}"
            )

            print(
                "=" * 60
            )

            if self.freeze_resnet_epochs > 0:

                if (
                    epoch
                    <= self.freeze_resnet_epochs
                ):
                    print(
                        "[TRAIN] "
                        "ResNet backbone: FROZEN"
                    )

                else:
                    print(
                        "[TRAIN] "
                        "ResNet backbone: TRAINABLE"
                    )

            # -------------------------------------------------
            # Training
            # -------------------------------------------------

            train_stats = (
                self.train_one_epoch()
            )

            print(
                f"[TRAIN] "
                f"loss="
                f"{train_stats['total_loss']:.6f}"
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            val_metrics = (
                self.validate()
            )

            print(
                f"[VAL] "
                f"AP={val_metrics['AP']:.6f} "
                f"AR@10={val_metrics['AR@10']:.6f}"
            )

            # -------------------------------------------------
            # Scheduler
            #
            # StepLR is epoch-based.
            # Current epoch uses current LR.
            # New LR is used by next epoch.
            # -------------------------------------------------

            if self.scheduler is not None:
                self.scheduler.step()

            # -------------------------------------------------
            # Update best metric
            #
            # AP is the primary validation metric.
            # -------------------------------------------------

            current_metric = float(
                val_metrics["AP"]
            )

            is_best = (
                current_metric
                > self.best_metric
            )

            if is_best:

                self.best_metric = (
                    current_metric
                )

                best_path = os.path.join(
                    self.checkpoint_dir,
                    BEST_CHECKPOINT_NAME,
                )

                self.save_checkpoint(
                    best_path
                )

                print(
                    "[CHECKPOINT] "
                    f"New best AP: "
                    f"{self.best_metric:.6f}"
                )

            # -------------------------------------------------
            # Save last checkpoint
            #
            # Saved AFTER best_metric update so last.pt
            # represents the complete state of the epoch.
            # -------------------------------------------------

            if (
                epoch
                % SAVE_EVERY_N_EPOCHS
                == 0
            ):

                last_path = os.path.join(
                    self.checkpoint_dir,
                    LAST_CHECKPOINT_NAME,
                )

                self.save_checkpoint(
                    last_path
                )

                print(
                    "[CHECKPOINT] "
                    f"Saved: {last_path}"
                )

            # -------------------------------------------------
            # Flush TensorBoard
            # -------------------------------------------------

            self.writer.flush()

        # -----------------------------------------------------
        # Training finished
        # -----------------------------------------------------

        self.writer.flush()

        print()
        print(
            "=" * 60
        )

        print(
            "Training completed."
        )

        print(
            f"Best validation AP: "
            f"{self.best_metric:.6f}"
        )

        print(
            "=" * 60
        )