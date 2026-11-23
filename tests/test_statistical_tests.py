from src.evaluation.statistical_tests import holm_bonferroni_adjust


def test_holm_bonferroni_adjusts_in_original_order_with_monotone_values():
    assert holm_bonferroni_adjust([0.04, 0.01, 0.03]) == [0.06, 0.03, 0.06]


def test_holm_bonferroni_handles_tied_p_values():
    assert holm_bonferroni_adjust([0.01, 0.01]) == [0.02, 0.02]
