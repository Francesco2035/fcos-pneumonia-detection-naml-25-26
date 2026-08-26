import os
import time

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR

from src.train import Trainer


class ModelEMA:
    """
    Exponential Moving Average of trainable model parameters.

    The implementation mirrors the advanced training recipe used in the
    reference engine: EMA is updated after every optimizer step and the
    shadow weights are used during validation/checkpoint selection.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialize(model)

    def _initialize(self, model):
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(),
                    alpha=1.0 - self.decay,
                )

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.shadow.items()
        }

    def load_state_dict(self, state_dict):
        self.shadow = {
            name: value.clone()
            for name, value in state_dict.items()
        }
        self.backup = {}


class TrainerV2(Trainer):
    """
    Advanced FCOS trainer.

    This class intentionally inherits the existing Trainer so the dataset
    handling, target generation, logging format, TensorBoard logging and
    checkpoint directory conventions stay compatible with the normal trainer.

    V2 changes only the optimization recipe:

        - initial ResNet freeze is preserved
        - 2-epoch LR warmup
        - cosine annealing
        - after unfreeze: backbone LR = 0.1 * head/base LR
        - EMA of trainable weights
        - gradient clipping remains identical

    The standard Trainer is left untouched.
    """

    def __init__(
        self,
        *args,
        base_lr=1e-4,
        backbone_lr_factor=0.1,
        warmup_epochs=2,
        ema_decay=0.999,
        use_ema=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if base_lr <= 0:
            raise ValueError("base_lr must be > 0.")
        if backbone_lr_factor <= 0:
            raise ValueError("backbone_lr_factor must be > 0.")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0.")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1).")

        self.base_lr = float(base_lr)
        self.backbone_lr_factor = float(backbone_lr_factor)
        self.warmup_epochs = int(warmup_epochs)
        self.ema_decay = float(ema_decay)
        self.use_ema = bool(use_ema)

        self._optimizer_after_unfreeze = False
        self._build_initial_optimizer()

        if self.use_ema:
            self.ema = ModelEMA(
                self.model,
                decay=self.ema_decay,
            )
        else:
            self.ema = None

    # =========================================================
    # Parameter groups
    # =========================================================

    def _backbone_parameters(self):
        return [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        ]

    def _head_parameters(self):
        backbone_parameter_ids = {
            id(parameter)
            for parameter in self.backbone.parameters()
        }

        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in backbone_parameter_ids
        ]

    def _build_initial_optimizer(self, total_epochs=None):
        """
        Frozen-backbone optimizer: only trainable FPN/head parameters.

        Warmup is applied to the base LR. After warmup, cosine annealing is
        used until the backbone is unfrozen.
        """

        head_parameters = self._head_parameters()

        if not head_parameters:
            raise RuntimeError(
                "No trainable parameters found for TrainerV2 "
                "while the backbone is frozen."
            )

        self.optimizer = torch.optim.Adam(
            head_parameters,
            lr=self.base_lr,
            weight_decay=self._weight_decay(),
        )

        if total_epochs is None:
            total_epochs_hint = max(
                self.warmup_epochs + 1,
                self.freeze_resnet_epochs + 1,
            )
        else:
            total_epochs_hint = max(
                int(total_epochs),
                self.warmup_epochs + 1,
            )

        if self.warmup_epochs > 0:
            self.optimizer = torch.optim.Adam(
                head_parameters,
                lr=self.base_lr,
                weight_decay=self._weight_decay(),
            )

            warmup_iters = self.warmup_epochs

            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_iters,
            )

            cosine_span = max(
                1,
                total_epochs_hint - self.warmup_epochs,
            )

            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_span,
                eta_min=1e-6,
            )

            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs_hint,
                eta_min=1e-6,
            )

        print()
        print(
            "[V2] Optimizer initialized: "
            f"heads LR={self.base_lr:.6e}"
        )

        if self.warmup_epochs > 0:
            print(
                "[V2] LR schedule: "
                f"warmup({self.warmup_epochs} epochs) + cosine"
            )
        else:
            print(
                "[V2] LR schedule: cosine"
            )

    def _weight_decay(self):
        # The existing main passes Adam with this same value into Trainer.
        # Preserve it when possible by reading the current optimizer groups.
        if getattr(self, "optimizer", None) is not None:
            groups = getattr(self.optimizer, "param_groups", [])
            if groups:
                return groups[0].get("weight_decay", 0.0)
        return 0.0

    def _rebuild_optimizer_after_unfreeze(self, remaining_epochs):
        backbone_parameters = self._backbone_parameters()
        head_parameters = self._head_parameters()

        if not backbone_parameters:
            raise RuntimeError(
                "Backbone was unfrozen but no backbone parameters "
                "are marked trainable."
            )

        if not head_parameters:
            raise RuntimeError(
                "No trainable detector/FPN/head parameters found."
            )

        backbone_lr = (
            self.base_lr * self.backbone_lr_factor
        )

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": backbone_parameters,
                    "lr": backbone_lr,
                },
                {
                    "params": head_parameters,
                    "lr": self.base_lr,
                },
            ],
            weight_decay=self._weight_decay(),
        )

        cosine_epochs = max(
            1,
            int(remaining_epochs),
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )

        self._optimizer_after_unfreeze = True

        print()
        print(
            "[V2] Optimizer rebuilt after ResNet unfreeze."
        )

        print(
            f"[V2] Backbone LR: {backbone_lr:.6e} "
            f"({self.backbone_lr_factor:.2f} x base LR)"
        )

        print(
            f"[V2] FPN/head LR: {self.base_lr:.6e}"
        )

        print(
            f"[V2] Cosine schedule remaining epochs: "
            f"{cosine_epochs}"
        )

        if self.use_ema:
            self.ema = ModelEMA(
                self.model,
                decay=self.ema_decay,
            )
            print(
                f"[V2] EMA reinitialized after unfreeze "
                f"(decay={self.ema_decay})"
            )

    # =========================================================
    # Freeze/unfreeze
    # =========================================================

    def _update_backbone_freeze_v2(self, epoch, num_epochs):
        if self.freeze_resnet_epochs <= 0:
            return

        if epoch != self.freeze_resnet_epochs + 1:
            return

        self._set_backbone_trainable(True)
        self._set_backbone_train_mode()

        print()
        print(
            "[LOG] ResNet backbone UNFROZEN."
        )
        print(
            "[LOG] ResNet parameters are now trainable "
            f"from epoch {epoch}."
        )

        remaining_epochs = (
            num_epochs - epoch + 1
        )

        self._rebuild_optimizer_after_unfreeze(
            remaining_epochs=remaining_epochs,
        )

    # =========================================================
    # Training epoch
    # =========================================================

    def train_one_epoch(self):
        """
        Same logging/timing behavior as the original Trainer, but using
        the V2 optimizer and EMA updates.
        """

        self.model.train()

        if (
            self.freeze_resnet_epochs > 0
            and self.current_epoch <= self.freeze_resnet_epochs
        ):
            self._set_backbone_eval()

        total_loss = 0.0
        center_loss = 0.0
        regression_loss = 0.0
        centerness_loss = 0.0

        level_loss = {
            level: 0.0
            for level in self._levels()
        }

        level_positive = {
            level: 0.0
            for level in self._levels()
        }

        epoch_start = time.perf_counter()

        data_time = 0.0
        forward_time = 0.0
        target_time = 0.0
        backward_time = 0.0

        batch_start = time.perf_counter()
        window_batches = 0

        for batch_idx, (
            images,
            dataset_targets,
        ) in enumerate(
            self.train_loader,
            start=1,
        ):

            current_time = time.perf_counter()
            data_time += current_time - batch_start

            images = images.to(self.device)

            self.optimizer.zero_grad()

            forward_start = time.perf_counter()

            predictions = self.model(
                images
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            forward_time += (
                time.perf_counter()
                - forward_start
            )

            target_start = time.perf_counter()

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

            loss_start = time.perf_counter()

            losses = self.criterion(
                predictions,
                targets,
            )

            loss = losses["total"]

            total_loss += losses["total"].item()
            center_loss += losses["center"].item()
            regression_loss += losses["regression"].item()
            centerness_loss += losses["centerness"].item()

            for level in self._levels():
                level_results = losses["levels"][level]

                for result in level_results:
                    level_loss[level] += (
                        result["total"].item()
                    )
                    level_positive[level] += (
                        result["num_positive"].item()
                    )

            loss.backward()

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

            if self.ema is not None:
                self.ema.update(
                    self.model
                )

            self.global_step += 1
            self._log_gradient_histograms()

            window_batches += 1
            batch_start = time.perf_counter()

            if (
                batch_idx % 100 == 0
                or batch_idx == len(self.train_loader)
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
                    batch_idx * self.batch_size,
                    len(
                        self.train_loader.dataset
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
                    f"        loss={loss.item():.6f} "
                    f"speed={images_per_second:.2f} img/s"
                )

                print(
                    f"        avg batch time: "
                    f"{window_time / window_batches:.4f}s"
                )

                print(
                    f"        data loading:   "
                    f"{avg_data:.4f}s"
                )

                print(
                    f"        forward:        "
                    f"{avg_forward:.4f}s"
                )

                print(
                    f"        target build:   "
                    f"{avg_target:.4f}s"
                )

                print(
                    f"        backward+step:  "
                    f"{avg_backward:.4f}s"
                )

                data_time = 0.0
                forward_time = 0.0
                target_time = 0.0
                backward_time = 0.0
                window_batches = 0

        num_batches = len(
            self.train_loader
        )

        if num_batches <= 0:
            raise RuntimeError(
                "Training DataLoader contains no batches."
            )

        total_loss /= num_batches
        center_loss /= num_batches
        regression_loss /= num_batches
        centerness_loss /= num_batches

        for level in self._levels():
            level_loss[level] /= num_batches
            level_positive[level] /= num_batches

        stats = {
            "total_loss": total_loss,
            "center_loss": center_loss,
            "regression_loss": regression_loss,
            "centerness_loss": centerness_loss,
            "level_loss": level_loss,
            "level_positive": level_positive,
        }

        self._log_training_losses(
            stats
        )

        self._log_parameter_histograms()

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

    def _levels(self):
        # Imported indirectly through the base Trainer's target generator
        # constants; avoid introducing a new config dependency here.
        from src.config import LEVELS
        return LEVELS

    # =========================================================
    # Validation
    # =========================================================

    def validate(self):
        self.model.eval()

        if self.ema is not None:
            self.ema.apply_shadow(
                self.model
            )

        try:
            with torch.no_grad():
                metrics = self.evaluator.evaluate(
                    self.val_loader
                )
        finally:
            if self.ema is not None:
                self.ema.restore(
                    self.model
                )

        self._log_validation_metrics(
            metrics
        )

        return metrics

    # =========================================================
    # Checkpointing
    # =========================================================

    def save_checkpoint(self, path):
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "freeze_resnet_epochs": self.freeze_resnet_epochs,
            "base_lr": self.base_lr,
            "backbone_lr_factor": self.backbone_lr_factor,
            "warmup_epochs": self.warmup_epochs,
            "ema_decay": self.ema_decay,
            "use_ema": self.use_ema,
        }

        if self.scheduler is not None:
            checkpoint[
                "scheduler_state_dict"
            ] = self.scheduler.state_dict()

        if self.ema is not None:
            checkpoint[
                "ema_state_dict"
            ] = self.ema.state_dict()

        torch.save(
            checkpoint,
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.current_epoch = checkpoint[
            "epoch"
        ]
        self.global_step = checkpoint[
            "global_step"
        ]
        self.best_metric = checkpoint[
            "best_metric"
        ]

        # Rebuild the optimizer to exactly match the resumed backbone state.
        if (
            self.freeze_resnet_epochs > 0
            and self.current_epoch <= self.freeze_resnet_epochs
        ):
            self._set_backbone_trainable(False)
            self._set_backbone_eval()
            self._build_initial_optimizer()
        else:
            self._set_backbone_trainable(True)
            self._set_backbone_train_mode()
            remaining_epochs = max(
                1,
                self.current_epoch,
            )
            self._rebuild_optimizer_after_unfreeze(
                remaining_epochs=remaining_epochs,
            )

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer_state_dict"
                    ]
                )
            except ValueError as exc:
                print(
                    "[V2] Warning: optimizer state could not be "
                    "restored exactly; continuing with rebuilt optimizer."
                )
                print(
                    f"[V2] Optimizer restore detail: {exc}"
                )

        if (
            self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            try:
                self.scheduler.load_state_dict(
                    checkpoint[
                        "scheduler_state_dict"
                    ]
                )
            except (ValueError, KeyError) as exc:
                print(
                    "[V2] Warning: scheduler state could not be "
                    "restored exactly; continuing with rebuilt scheduler."
                )
                print(
                    f"[V2] Scheduler restore detail: {exc}"
                )

        if self.use_ema:
            saved_ema = checkpoint.get(
                "ema_state_dict"
            )
            if saved_ema is not None:
                self.ema = ModelEMA(
                    self.model,
                    decay=self.ema_decay,
                )
                self.ema.load_state_dict(
                    saved_ema
                )

        print(
            f"[LOG] Checkpoint loaded: {path}"
        )
        print(
            f"[LOG] Resuming from epoch: "
            f"{self.current_epoch}"
        )

    # =========================================================
    # Full training loop
    # =========================================================

    def train(self, num_epochs):
        if num_epochs < 1:
            raise ValueError(
                "num_epochs must be >= 1."
            )

        # Rebuild the initial warmup+cosine schedule with the actual
        # requested training horizon. This keeps the warmup identical
        # across experiments regardless of the freeze duration.
        if (
            self.current_epoch == 0
            and not self._optimizer_after_unfreeze
        ):
            self._build_initial_optimizer(
                total_epochs=num_epochs
            )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        if self.resume:
            if self.resume_checkpoint is None:
                raise ValueError(
                    "resume=True but no resume checkpoint was provided."
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

            if self.current_epoch >= num_epochs:
                print(
                    "[LOG] Training already completed up to epoch "
                    f"{self.current_epoch}."
                )
                self.writer.flush()
                return

        for epoch in range(
            self.current_epoch + 1,
            num_epochs + 1,
        ):

            self.current_epoch = epoch

            self._update_backbone_freeze_v2(
                epoch,
                num_epochs,
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
                if epoch <= self.freeze_resnet_epochs:
                    print(
                        "[TRAIN] ResNet backbone: FROZEN"
                    )
                else:
                    print(
                        "[TRAIN] ResNet backbone: TRAINABLE"
                    )

            # Explicitly expose the effective LR configuration.
            if len(self.optimizer.param_groups) == 1:
                print(
                    f"[V2] Learning rate: "
                    f"{self.optimizer.param_groups[0]['lr']:.6e}"
                )
            else:
                print(
                    f"[V2] Backbone LR: "
                    f"{self.optimizer.param_groups[0]['lr']:.6e}"
                )
                print(
                    f"[V2] FPN/head LR: "
                    f"{self.optimizer.param_groups[1]['lr']:.6e}"
                )

            train_stats = self.train_one_epoch()

            print(
                f"[TRAIN] loss="
                f"{train_stats['total_loss']:.6f}"
            )

            val_metrics = self.validate()

            print(
                f"[VAL] "
                f"AP={val_metrics['AP']:.6f} "
                f"AR@10={val_metrics['AR@10']:.6f}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

            current_metric = float(
                val_metrics["AP"]
            )

            is_best = (
                current_metric
                > self.best_metric
            )

            if is_best:
                self.best_metric = current_metric

                best_path = os.path.join(
                    self.checkpoint_dir,
                    "best.pt",
                )

                # Match the reference engine: when EMA is enabled,
                # the best checkpoint stores the EMA weights.
                if self.ema is not None:
                    self.ema.apply_shadow(self.model)

                try:
                    self.save_checkpoint(
                        best_path
                    )
                finally:
                    if self.ema is not None:
                        self.ema.restore(self.model)

                print(
                    "[CHECKPOINT] New best AP: "
                    f"{self.best_metric:.6f}"
                )

            last_path = os.path.join(
                self.checkpoint_dir,
                "last.pt",
            )

            self.save_checkpoint(
                last_path
            )

            print(
                "[CHECKPOINT] Saved: "
                f"{last_path}"
            )

            self.writer.flush()

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
