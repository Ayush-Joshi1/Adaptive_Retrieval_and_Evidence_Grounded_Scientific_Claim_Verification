"""Reusable evaluation utilities for retrieval, classification, and statistical analysis."""

from .classification_metrics import (
    SUPPORTED_LABELS,
    accuracy,
    classification_report,
    confusion_matrix,
    macro_f1,
    macro_precision,
    macro_recall,
    per_class_metrics,
)
from .retrieval_metrics import (
    average_precision_at_k,
    compute_retrieval_metrics,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .statistical_tests import paired_bootstrap_ci, paired_system_comparison
