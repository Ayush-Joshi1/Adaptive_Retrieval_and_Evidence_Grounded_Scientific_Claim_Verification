from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> Dict[str, float | list[float]]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have identical shapes")
    if a.size == 0:
        return {"mean_difference": 0.0, "ci95": [0.0, 0.0]}

    differences = a - b
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(differences, size=(n_bootstrap, len(differences)), replace=True)
    boot_means = bootstrap.mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.5 * (1.0 - confidence_level), 0.5 * (1.0 + confidence_level)])
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95": [float(lower), float(upper)],
    }


def paired_system_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = 0,
) -> Dict[str, float | list[float]]:
    summary = paired_bootstrap_ci(scores_a, scores_b, n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed)
    a_mean = float(np.mean(np.asarray(scores_a, dtype=float)))
    b_mean = float(np.mean(np.asarray(scores_b, dtype=float)))
    summary["score_a_mean"] = a_mean
    summary["score_b_mean"] = b_mean
    return summary


def holm_bonferroni_adjust(p_values: Sequence[float]) -> list[float]:
    values = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted: list[float] = [1.0 for _ in values]
    running = 1.0
    m = len(values)
    for rank, (index, p) in enumerate(values, start=1):
        adjusted_p = min(1.0, max(0.0, (m - rank + 1) * p))
        if rank > 1:
            adjusted_p = max(adjusted_p, running)
        running = adjusted_p
        adjusted[index] = adjusted_p
    return adjusted


def paired_t_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have identical shapes")
    if a.size == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    from scipy.stats import ttest_rel
    statistic, p_value = ttest_rel(a, b)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def mcnemar_test(y_true_a: Sequence[bool | int | str], y_true_b: Sequence[bool | int | str], *, labels: Sequence[str] | None = None) -> Dict[str, float | int | str]:
    if len(y_true_a) != len(y_true_b):
        raise ValueError("y_true_a and y_true_b must have the same length")

    def to_binary(values: Sequence[bool | int | str]) -> list[bool]:
        converted: list[bool] = []
        for value in values:
            if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                converted.append(bool(value))
            elif isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "correct", "t", "1", "yes", "y"}:
                    converted.append(True)
                elif normalized in {"false", "incorrect", "f", "0", "no", "n"}:
                    converted.append(False)
                elif labels is not None and normalized in {label.lower() for label in labels}:
                    converted.append(True)
                else:
                    converted.append(False)
            else:
                converted.append(bool(value))
        return converted

    a_correct = to_binary(y_true_a)
    b_correct = to_binary(y_true_b)
    n_01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n_10 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    if n_01 + n_10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "note": "No discordant paired outcomes; McNemar test is not informative."}
    statistic = ((abs(n_01 - n_10) - 1.0) ** 2) / float(n_01 + n_10)
    try:
        from scipy.stats import chi2
        p_value = float(chi2.sf(statistic, 1))
    except Exception:
        p_value = 1.0
    return {"statistic": float(statistic), "p_value": float(p_value), "discordant_pairs": {"a_correct_b_incorrect": int(n_01), "a_incorrect_b_correct": int(n_10)}}
