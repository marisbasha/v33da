"""Evaluation metrics for vocal attribution and 3D localization."""

import numpy as np
from collections import defaultdict


def attribution_accuracy(pred_idx: np.ndarray, true_idx: np.ndarray) -> float:
    """Simple accuracy: fraction of correct attributions."""
    return float(np.mean(pred_idx == true_idx))


def macro_f1(pred_idx: np.ndarray, true_idx: np.ndarray, n_classes: int | None = None) -> float:
    """Macro-averaged F1 score across all classes."""
    if n_classes is None:
        n_classes = max(pred_idx.max(), true_idx.max()) + 1

    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred_idx == c) & (true_idx == c))
        fp = np.sum((pred_idx == c) & (true_idx != c))
        fn = np.sum((pred_idx != c) & (true_idx == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s))


def per_class_metrics(pred_idx: np.ndarray, true_idx: np.ndarray, n_classes: int | None = None) -> list[dict]:
    """Per-class precision, recall, F1."""
    if n_classes is None:
        n_classes = max(pred_idx.max(), true_idx.max()) + 1

    results = []
    for c in range(n_classes):
        tp = np.sum((pred_idx == c) & (true_idx == c))
        fp = np.sum((pred_idx == c) & (true_idx != c))
        fn = np.sum((pred_idx != c) & (true_idx == c))
        support = np.sum(true_idx == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "class": c, "precision": precision, "recall": recall,
            "f1": f1, "support": int(support),
        })
    return results


def localization_error(pred_pos: np.ndarray, true_pos: np.ndarray) -> dict:
    """
    Euclidean localization error statistics.

    Args:
        pred_pos: (N, 3) predicted 3D positions (mm)
        true_pos: (N, 3) ground truth 3D positions (mm)

    Returns:
        dict with mean, median, p90, std errors in mm and cm
    """
    errors_mm = np.linalg.norm(pred_pos - true_pos, axis=1)
    errors_cm = errors_mm / 10.0

    return {
        "mean_mm": float(np.mean(errors_mm)),
        "median_mm": float(np.median(errors_mm)),
        "p90_mm": float(np.percentile(errors_mm, 90)),
        "std_mm": float(np.std(errors_mm)),
        "mean_cm": float(np.mean(errors_cm)),
        "median_cm": float(np.median(errors_cm)),
        "pct_within_5cm": float(np.mean(errors_cm < 5) * 100),
        "pct_within_10cm": float(np.mean(errors_cm < 10) * 100),
        "pct_within_20cm": float(np.mean(errors_cm < 20) * 100),
    }


def expected_calibration_error(probs: np.ndarray, true_idx: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.

    Args:
        probs: (N, C) predicted probabilities
        true_idx: (N,) ground truth class indices

    Returns:
        ECE value (lower is better)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == true_idx).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def distance_binned_metrics(
    pred_idx: np.ndarray,
    true_idx: np.ndarray,
    distances_mm: np.ndarray,
    bins_mm: list[float] | None = None,
) -> dict:
    """
    Attribution accuracy stratified by minimum inter-bird distance.

    Args:
        pred_idx, true_idx: (N,) arrays
        distances_mm: (N,) minimum inter-bird distances
        bins_mm: bin edges (default: [0, 50, 100, 200, 500, inf])

    Returns:
        dict mapping bin label → {accuracy, count}
    """
    if bins_mm is None:
        bins_mm = [0, 50, 100, 200, 500, float("inf")]

    results = {}
    for i in range(len(bins_mm) - 1):
        lo, hi = bins_mm[i], bins_mm[i + 1]
        mask = (distances_mm >= lo) & (distances_mm < hi)
        count = mask.sum()
        if count > 0:
            acc = float(np.mean(pred_idx[mask] == true_idx[mask]))
        else:
            acc = None
        label = f"{lo:.0f}-{hi:.0f}mm" if hi < float("inf") else f">{lo:.0f}mm"
        results[label] = {"accuracy": acc, "count": int(count)}

    return results


def distance_binned_localization(
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    distances_mm: np.ndarray,
    bins_mm: list[float] | None = None,
) -> dict:
    """Localization error stratified by minimum inter-bird distance."""
    if bins_mm is None:
        bins_mm = [0, 50, 100, 200, 500, float("inf")]

    errors_cm = np.linalg.norm(pred_pos - true_pos, axis=1) / 10.0

    results = {}
    for i in range(len(bins_mm) - 1):
        lo, hi = bins_mm[i], bins_mm[i + 1]
        mask = (distances_mm >= lo) & (distances_mm < hi)
        count = mask.sum()
        if count > 0:
            mean_err = float(np.mean(errors_cm[mask]))
            median_err = float(np.median(errors_cm[mask]))
        else:
            mean_err = median_err = None
        label = f"{lo:.0f}-{hi:.0f}mm" if hi < float("inf") else f">{lo:.0f}mm"
        results[label] = {"mean_error_cm": mean_err, "median_error_cm": median_err, "count": int(count)}

    return results


def compute_all_metrics(
    pred_idx: np.ndarray,
    true_idx: np.ndarray,
    pred_pos: np.ndarray,
    true_pos: np.ndarray,
    probs: np.ndarray | None = None,
    distances_mm: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict:
    """Compute all metrics for both tasks."""
    results = {
        "attribution": {
            "accuracy": attribution_accuracy(pred_idx, true_idx),
            "macro_f1": macro_f1(pred_idx, true_idx, n_classes),
            "per_class": per_class_metrics(pred_idx, true_idx, n_classes),
        },
        "localization": localization_error(pred_pos, true_pos),
    }

    if probs is not None:
        results["attribution"]["ece"] = expected_calibration_error(probs, true_idx)

    if distances_mm is not None:
        results["attribution"]["by_distance"] = distance_binned_metrics(
            pred_idx, true_idx, distances_mm,
        )
        results["localization"]["by_distance"] = distance_binned_localization(
            pred_pos, true_pos, distances_mm,
        )

    return results
