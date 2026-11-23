import pytest

from src.evaluation.classification_metrics import accuracy, confusion_matrix, macro_f1, macro_precision, macro_recall, per_class_metrics


def test_classification_metrics_without_missing_labels():
    y_true = ["SUPPORT", "CONTRADICT", "SUPPORT", "NOT ENOUGH INFO"]
    y_pred = ["SUPPORT", "CONTRADICT", "CONTRADICT", "NOT ENOUGH INFO"]
    assert accuracy(y_true, y_pred) == 0.75
    assert macro_precision(y_true, y_pred) == pytest.approx(0.8333333333333333)
    assert macro_recall(y_true, y_pred) == pytest.approx(0.8333333333333333)
    assert macro_f1(y_true, y_pred) == pytest.approx(0.7777777777777777)
    assert confusion_matrix(y_true, y_pred)["SUPPORT"]["SUPPORT"] == 1
    assert per_class_metrics(y_true, y_pred)["CONTRADICT"]["f1"] == pytest.approx(0.6666666666666666)


def test_missing_dataset_labels_are_handled_gracefully():
    y_true = ["SUPPORT", "CONTRADICT"]
    y_pred = ["SUPPORT", "SUPPORT"]
    metrics = per_class_metrics(y_true, y_pred)
    assert set(metrics) == {"SUPPORT", "CONTRADICT"}
    assert metrics["CONTRADICT"]["support"] == 1
