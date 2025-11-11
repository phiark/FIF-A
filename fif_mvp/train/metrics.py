"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, num_bins: int = 15
) -> float:
    """Standard ECE with equal-width bins."""

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if not mask.any():
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_labels: int
) -> np.ndarray:
    """Return counts matrix."""

    matrix = np.zeros((num_labels, num_labels), dtype=int)
    for y_true, y_pred in zip(labels, preds):
        matrix[y_true, y_pred] += 1
    return matrix
