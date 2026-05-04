"""Model evaluation metrics including profit-impact analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    """Complete evaluation metrics for a prediction model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    profit_factor: float
    win_rate_if_traded: float
    total_samples: int
    positive_predictions: int
    true_positives: int
    false_positives: int


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    price_returns: np.ndarray | None = None,
) -> EvaluationResult:
    """Evaluate model predictions with classification and profit metrics.

    Args:
        y_true: Ground truth binary labels.
        y_proba: Predicted probabilities (0 to 1).
        threshold: Classification threshold.
        price_returns: Actual price returns for profit-impact analysis.
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    tp = int(((y_pred == 1) & (y_true_int == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_int == 0)).sum())
    tn = int(((y_pred == 0) & (y_true_int == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_int == 1)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    auc_roc = _compute_auc(y_true, y_proba)

    # Profit impact — if we only traded when model predicted UP
    profit_factor = 0.0
    win_rate = 0.0
    if price_returns is not None and (y_pred == 1).any():
        traded_returns = price_returns[y_pred == 1]
        gains = traded_returns[traded_returns > 0].sum()
        losses = abs(traded_returns[traded_returns < 0].sum())
        profit_factor = gains / max(losses, 1e-8)
        win_rate = float((traded_returns > 0).mean())
    elif (y_pred == 1).any():
        # Use true labels as a proxy when actual returns not available
        traded_truth = y_true_int[y_pred == 1]
        win_rate = float(traded_truth.mean())
        profit_factor = win_rate / max(1.0 - win_rate, 1e-8)

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        profit_factor=profit_factor,
        win_rate_if_traded=win_rate,
        total_samples=len(y_true),
        positive_predictions=int(y_pred.sum()),
        true_positives=tp,
        false_positives=fp,
    )


def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn dependency (trapezoidal rule)."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    sorted_indices = np.argsort(-y_scores)
    y_sorted = y_true[sorted_indices].astype(int)

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_prev, fpr_prev = 0.0, 0.0
    tp, fp = 0, 0
    auc = 0.0

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return float(auc)
