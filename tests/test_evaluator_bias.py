"""Tests that bias-detection evaluation computes a real metric (no placeholder)."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.comprehensive_evaluator import ComprehensiveEvaluator


@pytest.fixture(scope="module")
def ev():
    return ComprehensiveEvaluator()


def test_perfect_predictions_score_one(ev):
    r = ev._evaluate_bias_detection({"predicted_labels": [True, False, True],
                                     "true_labels": [True, False, True]})
    assert r.score == pytest.approx(1.0)


def test_score_varies_with_quality(ev):
    good = ev._evaluate_bias_detection({"predicted_labels": [["gender"], [], ["caste"]],
                                        "true_labels": [["gender"], [], ["caste"]]})
    bad = ev._evaluate_bias_detection({"predicted_labels": [["gender"], ["caste"], []],
                                       "true_labels": [[], [], ["caste"]]})
    assert bad.score < good.score
    assert bad.details["fp"] == 2 and bad.details["fn"] == 1


def test_missing_labels_is_not_faked(ev):
    r = ev._evaluate_bias_detection({})
    assert r.score == 0.0 and r.confidence == 0.0
