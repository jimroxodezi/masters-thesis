"""
detector.py
-----------
Anomaly scoring, threshold fitting and evaluation on top of the autoencoder.

Pipeline
--------
1. Score each spectrogram by its per-sample MSE reconstruction error.
2. Fit the threshold as a percentile of the training-set scores.
3. Flag a segment as anomalous when its score is strictly above the threshold.
4. Evaluate the flags and the score ranking against ground-truth labels.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from collections.abc import Iterable
from pathlib import Path
import pickle

from config import Config, cfg
from model import ConvAutoencoder


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detector
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Flags spectrograms whose reconstruction error exceeds a threshold fitted
    on normal training data.

    Parameters
    ----------
    model  : ConvAutoencoder
        Trained autoencoder. Scoring runs on the device of its parameters.
    config : Config
        train.batch_size : int
            Batch size used inside score_array().
        anomaly.threshold_percentile : float
            Percentile of the training scores used by fit_threshold().
    """

    def __init__(self, model: ConvAutoencoder, config: Config = cfg):
        self.model  = model
        self.config = config
        self.threshold: float | None = None
        # Percentile behind the threshold. A loaded detector takes it from the file,
        # not the config, so saving it again cannot mislabel the threshold.
        self.percentile: float = config.anomaly.threshold_percentile

    # ── Scoring ───────────────────────────────────────────────────────────

    def score(self, loader: Iterable[torch.Tensor]) -> np.ndarray:
        """
        Compute the per-sample MSE for every batch, in loader order.

        Parameters
        ----------
        loader : DataLoader, or any iterable, of (B, 1, H, W) tensors

        Returns
        -------
        np.ndarray, float32, shape (N,)
        """
        # reconstruction_error() does not set the mode, and in train mode BatchNorm
        # would use and update batch statistics, so a score would depend on its batch.
        self.model.eval()
        device = next(self.model.parameters()).device
        errors = [
            self.model.reconstruction_error(batch.to(device)).cpu().numpy()
            for batch in loader
        ]
        return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)

    def score_array(self, spectrograms: np.ndarray) -> np.ndarray:
        """
        Compute the per-sample MSE for a stack of spectrograms, batch by batch.

        Parameters
        ----------
        spectrograms : np.ndarray, float32, shape (N, H, W)
            Values should already be normalised to [0, 1].

        Returns
        -------
        np.ndarray, float32, shape (N,)
        """
        bs = self.config.train.batch_size
        # (N, H, W) → (B, 1, H, W); each batch is a view, so no full-size tensor is built
        batches = (
            torch.from_numpy(spectrograms[i:i + bs]).unsqueeze(1)
            for i in range(0, len(spectrograms), bs)
        )
        return self.score(batches)   # score() puts the model in eval mode first

    # ── Threshold and decision ────────────────────────────────────────────

    def fit_threshold(self, train_loader: DataLoader) -> float:
        """
        Set the threshold to a percentile of the training-set scores.

        Parameters
        ----------
        train_loader : DataLoader of (B, 1, H, W) tensors
            May shuffle: a percentile does not depend on order.

        Returns
        -------
        float
            The threshold, also stored in self.threshold.
        """
        p      = self.config.anomaly.threshold_percentile
        errors = self.score(train_loader)
        self.threshold  = float(np.percentile(errors, p))
        self.percentile = p
        print(f"[detector] threshold={self.threshold:.6f}  "
              f"(percentile {p:g} of {len(errors)} training scores)")
        return self.threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Flag each score as anomalous (1) or normal (0).

        A score equal to the threshold is normal: the comparison is strict.

        Parameters
        ----------
        scores : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, int8, shape (N,)
        """
        if self.threshold is None:
            raise RuntimeError("Call fit_threshold() before predict().")
        return (np.asarray(scores) > self.threshold).astype(np.int8)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        """
        Measure the flags and the score ranking against ground-truth labels.

        A ratio with a zero denominator is 0.0, with no warning. auc_pr is average
        precision: the step-wise area under the precision-recall curve, without
        trapezoidal interpolation.

        Parameters
        ----------
        scores : np.ndarray, shape (N,)
        labels : np.ndarray, any integer dtype, shape (N,)
            1 = anomalous, 0 = normal.

        Returns
        -------
        dict[str, float]
            accuracy, precision, recall, f1, auc_roc, auc_pr, fpr, and the
            confusion counts tp, fp, tn, fn. auc_roc and auc_pr are NaN when
            the labels contain only one class.
        """
        preds = self.predict(scores)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        # Undefined with one class: sklearn would warn or return a misleading 0.0 or 1.0
        auc_roc = auc_pr = float("nan")
        if len(np.unique(labels)) == 2:
            auc_roc = float(roc_auc_score(labels, scores))
            auc_pr  = float(average_precision_score(labels, scores))

        metrics = {
            "accuracy":  float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0.0)),
            "recall":    float(recall_score(labels, preds, zero_division=0.0)),
            "f1":        float(f1_score(labels, preds, zero_division=0.0)),
            "auc_roc":   auc_roc,
            "auc_pr":    auc_pr,
            "fpr":       float(fp / (fp + tn)) if fp + tn > 0 else 0.0,
            "tp":        float(tp),
            "fp":        float(fp),
            "tn":        float(tn),
            "fn":        float(fn),
        }

        print(f"[detector] accuracy={metrics['accuracy']:.4f}  "
              f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc_roc={auc_roc:.4f}  auc_pr={auc_pr:.4f}  "
              f"fpr={metrics['fpr']:.4f}  tp={tp}  fp={fp}  tn={tn}  fn={fn}")

        return metrics

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"threshold": self.threshold, "percentile": self.percentile}, f)
        print(f"[detector] saved → {path}")

    @classmethod
    def load(
        cls, path: Path, model: ConvAutoencoder, config: Config = cfg
    ) -> "AnomalyDetector":
        with open(path, "rb") as f:
            d = pickle.load(f)
        det = cls(model, config)
        det.threshold  = d["threshold"]
        det.percentile = d["percentile"]
        print(f"[detector] loaded ← {path}")
        return det
