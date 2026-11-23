from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

SUPPORTED_LABELS = ("SUPPORT", "CONTRADICT", "NOT ENOUGH INFO")


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> Dict[str, Dict[str, int]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    label_order = [label for label in labels if label in set(y_true) or label in set(y_pred)]
    if not label_order:
        label_order = list(dict.fromkeys([str(value) for value in y_true + y_pred if value]))
    matrix = {label: {other: 0 for other in label_order} for label in label_order}
    for actual, predicted in zip(y_true, y_pred):
        actual = str(actual)
        predicted = str(predicted)
        if actual not in matrix:
            matrix[actual] = {other: 0 for other in label_order}
        if predicted not in matrix[actual]:
            matrix[actual][predicted] = 0
        matrix[actual][predicted] += 1
    return matrix


def _safe_divide(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def per_class_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> Dict[str, Dict[str, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    label_order = [label for label in labels if label in set(y_true) or label in set(y_pred)]
    if not label_order:
        label_order = list(dict.fromkeys([str(value) for value in y_true + y_pred if value]))
    per_class: Dict[str, Dict[str, float]] = {}
    for label in label_order:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if str(actual) == label and str(predicted) == label)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if str(actual) != label and str(predicted) == label)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if str(actual) == label and str(predicted) != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for value in y_true if str(value) == label)}
    return per_class


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    return sum(1 for actual, predicted in zip(y_true, y_pred) if str(actual) == str(predicted)) / float(len(y_true))


def macro_precision(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> float:
    metrics = per_class_metrics(y_true, y_pred, labels)
    if not metrics:
        return 0.0
    return sum(item["precision"] for item in metrics.values()) / float(len(metrics))


def macro_recall(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> float:
    metrics = per_class_metrics(y_true, y_pred, labels)
    if not metrics:
        return 0.0
    return sum(item["recall"] for item in metrics.values()) / float(len(metrics))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> float:
    metrics = per_class_metrics(y_true, y_pred, labels)
    if not metrics:
        return 0.0
    return sum(item["f1"] for item in metrics.values()) / float(len(metrics))


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = SUPPORTED_LABELS) -> Dict[str, object]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": macro_precision(y_true, y_pred, labels),
        "macro_recall": macro_recall(y_true, y_pred, labels),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "class_distribution": dict(Counter(str(label) for label in y_true)),
    }
