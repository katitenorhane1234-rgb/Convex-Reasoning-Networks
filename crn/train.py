
"""
train.py
========
Training loop for Convex Reasoning Networks.

Implements a complete, self-contained training pipeline:

* Optimizer and LR scheduler construction
* Per-epoch train / validation loops
* Gradient clipping
* Checkpoint saving (best model + periodic)
* Early stopping
* Metric logging (loss, gradient norm, spectral diagnostics)
* Deterministic seeding

The primary entry point is :func:`train`, which accepts a
:class:`~config.CRNConfig` and returns a :class:`TrainingResult`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader

from config import CRNConfig, CHECKPOINTS_DIR
from crn import CRN, build_crn
from dataset import build_dataloaders
from metric import BaseMetric
from utils import set_seed, count_parameters, get_device


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    """Metrics recorded at the end of a single epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    lr: float
    grad_norm: float
    spectral_norm_A: float
    condition_number_M: float
    epoch_time_s: float


@dataclass
class TrainingResult:
    """
    Complete record of a training run.

    Attributes
    ----------
    best_val_loss:
        Lowest validation loss achieved during training.
    best_epoch:
        Epoch at which the best validation loss was achieved.
    epoch_metrics:
        List of per-epoch metrics (one entry per epoch).
    total_time_s:
        Wall-clock training time in seconds.
    checkpoint_path:
        Path to the saved best-model checkpoint.
    stopped_early:
        True if early stopping triggered before ``max_epochs``.
    """

    best_val_loss: float
    best_epoch: int
    epoch_metrics: list[EpochMetrics] = field(default_factory=list)
    total_time_s: float = 0.0
    checkpoint_path: Optional[Path] = None
    stopped_early: bool = False


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def trajectory_loss(
    predicted: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Mean squared error over a predicted state trajectory.

    Computes::

        L = (1/T) Σ_{t=1}^{T} ‖x̂_t - x_t‖²

    where the sum is over time steps (t=0, the initial state, is excluded
    since it is given as input and not predicted).

    Parameters
    ----------
    predicted:
        Model output of shape ``(batch, T+1, state_dim)``.
    target:
        Ground-truth trajectory of shape ``(batch, T+1, state_dim)``.
    reduction:
        ``'mean'`` averages over batch and time; ``'none'`` returns per-sample
        per-step losses.

    Returns
    -------
    Tensor
        Scalar loss (``reduction='mean'``) or shape ``(batch, T)`` otherwise.
    """
    # Exclude t=0 (initial state is given, not predicted)
    pred = predicted[:, 1:, :]          # (batch, T, state_dim)
    tgt = target[:, 1:, :]              # (batch, T, state_dim)

    # Per-step squared error: (batch, T)
    sq_err = ((pred - tgt) ** 2).sum(dim=-1)  # sum over state_dim

    if reduction == "none":
        return sq_err                    # (batch, T)
    elif reduction == "mean":
        return sq_err.mean()             # scalar
    elif reduction == "sum":
        return sq_err.sum()
    else:
        raise ValueError(f"Unknown reduction '{reduction}'. Expected 'mean', 'sum', or 'none'.")


# ---------------------------------------------------------------------------
# Optimizer & scheduler construction
# ---------------------------------------------------------------------------


def build_optimizer(model: CRN, cfg: CRNConfig) -> Optimizer:
    """
    Construct the AdamW optimizer.

    Parameters
    ----------
    model:
        The CRN model.
    cfg:
        Full experiment configuration.

    Returns
    -------
    Optimizer
    """
    return AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    cfg: CRNConfig,
) -> Optional[LRScheduler]:
    """
    Construct the learning-rate scheduler.

    Parameters
    ----------
    optimizer:
        The optimizer to schedule.
    cfg:
        Full experiment configuration.

    Returns
    -------
    Optional[LRScheduler]
        The scheduler, or ``None`` if ``cfg.train.lr_scheduler == 'none'``.
    """
    sched = cfg.train.lr_scheduler
    if sched == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-7)
    elif sched == "step":
        return StepLR(
            optimizer,
            step_size=cfg.train.lr_step_size,
            gamma=cfg.train.lr_gamma,
        )
    elif sched == "none":
        return None
    else:
        raise ValueError(f"Unknown lr_scheduler '{sched}'. Expected 'cosine', 'step', or 'none'.")


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class CheckpointManager:
    """
    Handles saving and loading model checkpoints.

    Saves the following state into each checkpoint file:

    * model state dict
    * optimizer state dict
    * scheduler state dict (if present)
    * epoch number
    * validation loss
    * full experiment configuration

    Parameters
    ----------
    cfg:
        Experiment configuration (used to construct save paths).
    """

    def __init__(self, cfg: CRNConfig) -> None:
        self.cfg = cfg
        self.save_dir = CHECKPOINTS_DIR / cfg.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: CRN,
        optimizer: Optimizer,
        epoch: int,
        val_loss: float,
        scheduler: Optional[LRScheduler] = None,
        tag: str = "checkpoint",
    ) -> Path:
        """
        Save a checkpoint to disk.

        Parameters
        ----------
        model:
            CRN model.
        optimizer:
            Optimizer.
        epoch:
            Current epoch number.
        val_loss:
            Validation loss at this epoch.
        scheduler:
            Optional LR scheduler.
        tag:
            Filename stem (e.g. ``'best'``, ``'epoch_050'``).

        Returns
        -------
        Path
            Path of the saved checkpoint file.
        """
        path = self.save_dir / f"{tag}.pt"
        payload: dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": self.cfg.to_dict(),
        }
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        return path

    def load(
        self,
        path: Path,
        model: CRN,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, float]:
        """
        Load a checkpoint from disk.

        Parameters
        ----------
        path:
            Checkpoint file path.
        model:
            CRN model to load state into.
        optimizer:
            Optional optimizer to restore.
        scheduler:
            Optional scheduler to restore.
        device:
            Target device.

        Returns
        -------
        tuple
            ``(epoch, val_loss)`` at the time of the checkpoint.
        """
        map_location = device if device is not None else "cpu"
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        return payload["epoch"], payload["val_loss"]

    def best_checkpoint_path(self) -> Path:
        """Return the path to the best-model checkpoint."""
        return self.save_dir / "best.pt"

    def periodic_checkpoint_path(self, epoch: int) -> Path:
        """Return the path for an epoch-periodic checkpoint."""
        return self.save_dir / f"epoch_{epoch:04d}.pt"


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """
    Tracks validation loss and signals when training should stop.

    Training stops when the validation loss does not improve by more than
    ``min_delta`` for ``patience`` consecutive epochs.

    Parameters
    ----------
    patience:
        Maximum number of epochs without improvement.
    min_delta:
        Minimum absolute improvement required to count as an improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False
        self._improved: bool = False

    def step(self, val_loss: float) -> bool:
        """
        Update state based on the current validation loss.

        Parameters
        ----------
        val_loss:
            Current epoch's validation loss.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._improved = True
        else:
            self.counter += 1
            self._improved = False

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    @property
    def improved(self) -> bool:
        """True if the last :meth:`step` call found an improvement."""
        return self._improved


# ---------------------------------------------------------------------------
# Single-epoch training and validation loops
# ---------------------------------------------------------------------------


def train_epoch(
    model: CRN,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    cfg: CRNConfig,
) -> Tuple[float, float]:
    """
    Run one training epoch.

    Parameters
    ----------
    model:
        CRN model (set to training mode internally).
    loader:
        Training DataLoader.
    optimizer:
        Optimizer.
    device:
        Compute device.
    cfg:
        Experiment configuration.

    Returns
    -------
    tuple
        ``(mean_loss, mean_grad_norm)`` over all batches.
    """
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    for batch in loader:
        states, inputs, contexts = batch
        states = states.to(device)       # (batch, T+1, state_dim)
        inputs = inputs.to(device)       # (batch, T, input_dim)

        optimizer.zero_grad()

        x0 = states[:, 0, :]             # (batch, state_dim)
        predicted_states, _ = model(x0, inputs)

        loss = trajectory_loss(predicted_states, states)
        loss.backward()

        # Gradient clipping
        grad_norm = 0.0
        if cfg.train.gradient_clip_norm > 0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.train.gradient_clip_norm,
                ).item()
            )
        else:
            # Compute norm without clipping for logging
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = total_norm ** 0.5

        optimizer.step()

        total_loss += loss.item()
        total_grad_norm += grad_norm
        n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_grad_norm / n_batches


@torch.no_grad()
def validate_epoch(
    model: CRN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate the model on a validation or test DataLoader.

    Parameters
    ----------
    model:
        CRN model (set to eval mode internally).
    loader:
        DataLoader.
    device:
        Compute device.

    Returns
    -------
    float
        Mean trajectory loss over the entire loader.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, inputs, contexts = batch
        states = states.to(device)
        inputs = inputs.to(device)

        x0 = states[:, 0, :]
        predicted_states, _ = model(x0, inputs)
        loss = trajectory_loss(predicted_states, states)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Primary training entry point
# ---------------------------------------------------------------------------


def train(cfg: CRNConfig) -> Tuple[CRN, TrainingResult]:
    """
    Run the full training pipeline.

    Steps:

    1. Seed everything for reproducibility.
    2. Build dataloaders, model, optimizer, and scheduler.
    3. Run the train / validate loop for up to ``cfg.train.epochs`` epochs.
    4. Save checkpoints (best model + periodic).
    5. Apply early stopping if validation loss stagnates.
    6. Return the trained model and a complete :class:`TrainingResult`.

    Parameters
    ----------
    cfg:
        Full experiment configuration.

    Returns
    -------
    tuple
        ``(trained_model, training_result)``
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # Model
    model = build_crn(cfg).to(device)

    # Optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Utilities
    ckpt_mgr = CheckpointManager(cfg)
    early_stop = EarlyStopping(patience=cfg.train.early_stopping_patience)

    best_val_loss = float("inf")
    best_epoch = 0
    epoch_metrics_list: list[EpochMetrics] = []
    best_ckpt_path: Optional[Path] = None

    t_start = time.monotonic()

    for epoch in range(1, cfg.train.epochs + 1):
        t_epoch = time.monotonic()

        train_loss, grad_norm = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss = validate_epoch(model, val_loader, device)

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Spectral diagnostics
        with torch.no_grad():
            sigma_A = float(model.spectral_norm_A().item())
            kappa_M = float(model.condition_number_M().item())

        epoch_time = time.monotonic() - t_epoch

        em = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            grad_norm=grad_norm,
            spectral_norm_A=sigma_A,
            condition_number_M=kappa_M,
            epoch_time_s=epoch_time,
        )
        epoch_metrics_list.append(em)

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt_path = ckpt_mgr.save(
                model, optimizer, epoch, val_loss, scheduler, tag="best"
            )

        # Periodic checkpoint
        if epoch % cfg.train.checkpoint_every == 0:
            ckpt_mgr.save(
                model, optimizer, epoch, val_loss, scheduler,
                tag=f"epoch_{epoch:04d}"
            )

        # Early stopping
        if early_stop.step(val_loss):
            result = TrainingResult(
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epoch_metrics=epoch_metrics_list,
                total_time_s=time.monotonic() - t_start,
                checkpoint_path=best_ckpt_path,
                stopped_early=True,
            )
            # Reload best model weights
            if best_ckpt_path is not None and best_ckpt_path.exists():
                ckpt_mgr.load(best_ckpt_path, model, device=device)
            return model, result

    total_time = time.monotonic() - t_start

    # Reload best model weights
    if best_ckpt_path is not None and best_ckpt_path.exists():
        ckpt_mgr.load(best_ckpt_path, model, device=device)

    result = TrainingResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        epoch_metrics=epoch_metrics_list,
        total_time_s=total_time,
        checkpoint_path=best_ckpt_path,
        stopped_early=False,
    )
    return model, result


# ---------------------------------------------------------------------------
# Resume training from a checkpoint
# ---------------------------------------------------------------------------


def resume_training(
    cfg: CRNConfig,
    checkpoint_path: Path,
) -> Tuple[CRN, TrainingResult]:
    """
    Resume a training run from a saved checkpoint.

    Parameters
    ----------
    cfg:
        Experiment configuration (must match the original run).
    checkpoint_path:
        Path to the checkpoint file to resume from.

    Returns
    -------
    tuple
        ``(trained_model, training_result)``
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    train_loader, val_loader, _ = build_dataloaders(cfg)

    model = build_crn(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    ckpt_mgr = CheckpointManager(cfg)
    start_epoch, best_val_loss = ckpt_mgr.load(
        checkpoint_path, model, optimizer, scheduler, device=device
    )

    early_stop = EarlyStopping(patience=cfg.train.early_stopping_patience)
    early_stop.best_loss = best_val_loss

    best_epoch = start_epoch
    epoch_metrics_list: list[EpochMetrics] = []
    best_ckpt_path: Optional[Path] = Path(checkpoint_path)

    t_start = time.monotonic()

    for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
        t_epoch = time.monotonic()

        train_loss, grad_norm = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss = validate_epoch(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        with torch.no_grad():
            sigma_A = float(model.spectral_norm_A().item())
            kappa_M = float(model.condition_number_M().item())

        em = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=current_lr,
            grad_norm=grad_norm,
            spectral_norm_A=sigma_A,
            condition_number_M=kappa_M,
            epoch_time_s=time.monotonic() - t_epoch,
        )
        epoch_metrics_list.append(em)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt_path = ckpt_mgr.save(
                model, optimizer, epoch, val_loss, scheduler, tag="best"
            )

        if epoch % cfg.train.checkpoint_every == 0:
            ckpt_mgr.save(
                model, optimizer, epoch, val_loss, scheduler,
                tag=f"epoch_{epoch:04d}"
            )

        if early_stop.step(val_loss):
            if best_ckpt_path is not None and best_ckpt_path.exists():
                ckpt_mgr.load(best_ckpt_path, model, device=device)
            return model, TrainingResult(
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epoch_metrics=epoch_metrics_list,
                total_time_s=time.monotonic() - t_start,
                checkpoint_path=best_ckpt_path,
                stopped_early=True,
            )

    if best_ckpt_path is not None and best_ckpt_path.exists():
        ckpt_mgr.load(best_ckpt_path, model, device=device)

    return model, TrainingResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        epoch_metrics=epoch_metrics_list,
        total_time_s=time.monotonic() - t_start,
        checkpoint_path=best_ckpt_path,
        stopped_early=False,
    )
