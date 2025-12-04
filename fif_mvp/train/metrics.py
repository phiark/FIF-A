"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


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

    flat = preds.astype(np.int64) + num_labels * labels.astype(np.int64)
    counts = np.bincount(flat, minlength=num_labels * num_labels)
    return counts.reshape(num_labels, num_labels)


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return AUROC, guarding against degenerate label/scores."""

    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.0


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return AUPRC, guarding against degenerate label/scores."""

    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.0
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return 0.0


def coverage_risk(
    errors: np.ndarray,
    energies: np.ndarray,
    coverage_points: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
) -> dict:
    """Compute coverage-risk curve (low energy = keep).

    Returns the full risk/coverage arrays plus AURC and risk sampled at preset coverages.
    """

    n = energies.shape[0]
    if n == 0:
        return {"aurc": 0.0, "coverages": [], "risks": [], "risk_at": {}}
    order = np.argsort(energies)  # keep low-energy (confident) points first
    sorted_errors = errors[order]
    cum_errors = np.cumsum(sorted_errors)
    idx = np.arange(1, n + 1)
    coverages = idx / n
    risks = cum_errors / idx
    aurc = float(np.trapz(risks, coverages))
    risk_at = {}
    for cov in coverage_points:
        cov = float(cov)
        target_idx = min(n - 1, max(0, int(np.ceil(cov * n)) - 1))
        risk_at[cov] = float(risks[target_idx])
    return {
        "aurc": aurc,
        "coverages": coverages,
        "risks": risks,
        "risk_at": risk_at,
    }


def split_energy_quantiles(
    errors: np.ndarray,
    energies: np.ndarray,
    percentiles: tuple[int, ...] = (50, 90, 99),
) -> dict:
    """Return energy quantiles for correct vs incorrect subsets."""

    result: dict = {}
    for tag, mask in (("correct", errors == 0), ("incorrect", errors == 1)):
        vals = energies[mask]
        if vals.size == 0:
            result[tag] = {f"p{p}": 0.0 for p in percentiles}
        else:
            qs = np.percentile(vals, percentiles)
            result[tag] = {f"p{p}": float(q) for p, q in zip(percentiles, qs)}
    return result
