"""
trainer.py
----------
Training loop for the convolutional autoencoder.

Epoch structure
---------------
1. Train on the shuffled training loader with clipped gradients.
2. Measure the reconstruction loss on the validation loader.
3. Step the plateau LR scheduler and the early-stopping monitor.
4. Checkpoint the model whenever validation improves.
5. Rewrite the training history so a killed run still leaves evidence.

After the last epoch the best checkpoint is restored, so the caller holds the
weights with the lowest validation loss rather than the last ones trained.
"""

import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _build_criterion(name: str) -> nn.Module:
    """Map ModelConfig.loss to its reconstruction loss."""
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name!r}")


def _write_history(history: dict, path: Path) -> None:
    """Replace the history file in one step, so a kill never leaves it half-written."""
    tmp = path.with_suffix(".json.tmp")   # same directory, so os.replace stays atomic
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Tracks the best validation loss and flags when training should end.

    Parameters
    ----------
    patience  : consecutive non-improving epochs before should_stop is set
    min_delta : smallest drop below best_loss that counts as an improvement
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.best_epoch  = 0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Record the validation loss of one epoch.

        Parameters
        ----------
        val_loss : validation loss of this epoch
        epoch    : 1-based epoch number

        Returns
        -------
        bool
            True if this epoch improved on best_loss by more than min_delta.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Trains an autoencoder to reconstruct its input spectrograms.

    Parameters
    ----------
    model        : ConvAutoencoder (any nn.Module with save() and load() will do)
    train_loader : DataLoader of (B, 1, H, W) batches, shuffled
    val_loader   : DataLoader of (B, 1, H, W) batches
    config       : Config
    device       : defaults to CUDA when available, else CPU
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config = cfg,
        device: torch.device | None = None,
    ):
        tc = config.train

        self.config       = config
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.criterion = _build_criterion(config.model.loss)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=tc.scheduler_factor,
            patience=tc.scheduler_patience,
        )
        self.early_stopping = EarlyStopping(tc.patience, tc.min_delta)

        # Pre-clip gradient norm of every step of the latest training pass
        self.grad_norms: list[float] = []

    # ── Single pass ───────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        """
        Run one pass over a loader.

        Parameters
        ----------
        loader   : DataLoader of (B, 1, H, W) batches
        training : True  — train mode with gradient updates; fills self.grad_norms
                   False — eval mode under torch.no_grad()

        Returns
        -------
        float
            Mean loss per sample. Each batch is weighted by its size because
            the last batch can be smaller than the others.
        """
        if training:
            self.model.train()
            self.grad_norms = []
        else:
            self.model.eval()

        total_loss = 0.0
        n_samples  = 0

        with torch.enable_grad() if training else torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                loss  = self.criterion(self.model(batch), batch)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # clip_grad_norm_ returns the norm measured before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.train.grad_clip_norm,
                    )
                    self.grad_norms.append(grad_norm.item())
                    self.optimizer.step()

                total_loss += loss.item() * len(batch)
                n_samples  += len(batch)

        if n_samples == 0:
            raise ValueError(f"Empty {'training' if training else 'validation'} loader: "
                             "is the capture too short for val_split?")
        return total_loss / n_samples

    # ── Training loop ─────────────────────────────────────────────────────

    def fit(self) -> dict:
        """
        Train up to train.epochs, then restore the best weights.

        Returns
        -------
        dict
            Lists with one entry per epoch: train_loss, val_loss, lr,
            epoch_time (seconds), grad_norm (mean pre-clip norm),
            clip_fraction (fraction of steps whose pre-clip norm exceeded the limit).
            Scalars: best_epoch (1-based), best_val_loss, stopped_early.

        Raises
        ------
        RuntimeError
            If no epoch produced a finite validation loss, so there is no
            checkpoint from this run to restore.
        """
        tc = self.config.train
        torch.manual_seed(tc.seed)   # reproducible batch shuffling

        checkpoint_dir = Path(self.config.checkpoint_dir)
        results_dir    = Path(self.config.results_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        best_path    = checkpoint_dir / "best_model.pt"
        history_path = results_dir / "training_history.json"

        history = {
            "train_loss":    [],
            "val_loss":      [],
            "lr":            [],
            "epoch_time":    [],
            "grad_norm":     [],
            "clip_fraction": [],
            "best_epoch":    0,
            "best_val_loss": float("inf"),
            "stopped_early": False,
        }

        for epoch in range(1, tc.epochs + 1):
            # Read before scheduler.step: this is the LR the epoch trains with
            lr    = self.optimizer.param_groups[0]["lr"]
            start = time.perf_counter()

            train_loss = self._run_epoch(self.train_loader, training=True)
            val_loss   = self._run_epoch(self.val_loader, training=False)
            epoch_time = time.perf_counter() - start

            print(f"[trainer] epoch {epoch:3d}/{tc.epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"lr={lr:.2e}  time={epoch_time:.1f}s", flush=True)

            self.scheduler.step(val_loss)
            if self.early_stopping.step(val_loss, epoch):
                self.model.save(best_path)

            n_steps = max(len(self.grad_norms), 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr)
            history["epoch_time"].append(epoch_time)
            history["grad_norm"].append(sum(self.grad_norms) / n_steps)
            history["clip_fraction"].append(
                sum(norm > tc.grad_clip_norm for norm in self.grad_norms) / n_steps
            )
            history["best_epoch"]    = self.early_stopping.best_epoch
            history["best_val_loss"] = self.early_stopping.best_loss
            # Hitting the patience limit on the final epoch cuts nothing short
            history["stopped_early"] = self.early_stopping.should_stop and epoch < tc.epochs

            _write_history(history, history_path)

            if history["stopped_early"]:
                break

        # A best_model.pt left by an earlier run must not pass as this run's result
        if history["best_epoch"] == 0:
            raise RuntimeError("No finite validation loss in this run, so it saved no "
                               f"checkpoint to restore (see {history_path}).")

        self.model.load(best_path, self.device)
        self.model.eval()

        print(f"[trainer] done  best_epoch={history['best_epoch']}  "
              f"best_val_loss={history['best_val_loss']:.6f}  "
              f"stopped_early={history['stopped_early']}", flush=True)
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config = cfg,
    device: torch.device | None = None,
) -> dict:
    """Build a Trainer and run it. Returns the history dict from Trainer.fit()."""
    return Trainer(model, train_loader, val_loader, config, device).fit()
