"""
Unit tests for model evaluation module.

Tests ModelEvaluator class including metrics calculation, visualization
generation, baseline comparison, and production threshold checking.
"""

import os
import sys
import warnings
from unittest.mock import Mock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from model_evaluation import (
    LOW_ACCURACY_WARNING_THRESHOLD,
    PRODUCTION_ACCURACY_THRESHOLD,
    ModelEvaluator,
)


def _make_mock_model_for_labels(y_test: np.ndarray, accuracy: float = 0.95):
    """Create a mock model that achieves the desired accuracy against y_test.

    The model returns predictions that match y_test for the first
    ``int(n * accuracy)`` samples and flips the rest.
    """
    model = Mock()
    n = len(y_test)
    n_correct = int(n * accuracy)
    y_pred = y_test.copy()
    # Flip labels for the tail to introduce errors
    for i in range(n_correct, n):
        y_pred[i] = 1 - y_pred[i]

    # Build matching probabilities
    proba = np.where(y_pred == 1, 0.9, 0.1).astype(float)

    model.predict = Mock(return_value=y_pred)
    model.predict_proba = Mock(
        return_value=np.column_stack([1 - proba, proba])
    )
    return model


@pytest.fixture
def evaluator():
    """Create a ModelEvaluator instance."""
    return ModelEvaluator()


@pytest.fixture
def sample_binary_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.15, 0.6, 0.3, 0.85, 0.9, 0.4, 0.75, 0.8])
    return y_true, y_pred, y_pred_proba


class TestCalculateMetrics:
    """Tests for calculate_metrics method."""

    def test_returns_all_required_metrics(self, evaluator, sample_binary_data):
        y_true, y_pred, y_pred_proba = sample_binary_data
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics

    def test_perfect_predictions(self, evaluator):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_metrics_values_in_valid_range(self, evaluator, sample_binary_data):
        y_true, y_pred, y_pred_proba = sample_binary_data
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

    def test_imperfect_predictions(self, evaluator, sample_binary_data):
        y_true, y_pred, y_pred_proba = sample_binary_data
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        # 8 out of 10 correct
        assert metrics["accuracy"] == 0.8


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix method."""

    def test_returns_confusion_matrix_array(self, evaluator, sample_binary_data, tmp_path):
        y_true, y_pred, _ = sample_binary_data
        save_path = str(tmp_path / "cm.png")
        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)

        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)

    def test_saves_file(self, evaluator, sample_binary_data, tmp_path):
        y_true, y_pred, _ = sample_binary_data
        save_path = str(tmp_path / "cm.png")

        evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)

        assert os.path.exists(save_path)

    def test_confusion_matrix_values(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        save_path = str(tmp_path / "cm.png")

        cm = evaluator.plot_confusion_matrix(y_true, y_pred, save_path=save_path)

        # Perfect predictions: TN=2, FP=0, FN=0, TP=2
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 2  # TP


class TestPlotRocCurve:
    """Tests for plot_roc_curve method."""

    def test_returns_fpr_tpr_auc(self, evaluator, sample_binary_data, tmp_path):
        y_true, _, y_pred_proba = sample_binary_data
        save_path = str(tmp_path / "roc.png")
        fpr, tpr, auc = evaluator.plot_roc_curve(
            y_true, y_pred_proba, save_path=save_path
        )

        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_saves_file(self, evaluator, sample_binary_data, tmp_path):
        y_true, _, y_pred_proba = sample_binary_data
        save_path = str(tmp_path / "roc.png")

        evaluator.plot_roc_curve(y_true, y_pred_proba, save_path=save_path)

        assert os.path.exists(save_path)

    def test_auc_for_good_model(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.9, 0.85, 0.95, 0.8, 0.88])
        save_path = str(tmp_path / "roc.png")

        _, _, auc = evaluator.plot_roc_curve(
            y_true, y_pred_proba, save_path=save_path
        )

        assert auc > 0.9


class TestPlotPrecisionRecallCurve:
    """Tests for plot_precision_recall_curve method."""

    def test_returns_precision_recall_arrays(self, evaluator, sample_binary_data, tmp_path):
        y_true, _, y_pred_proba = sample_binary_data
        save_path = str(tmp_path / "pr.png")
        precision, recall = evaluator.plot_precision_recall_curve(
            y_true, y_pred_proba, save_path=save_path
        )

        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert len(precision) == len(recall)

    def test_saves_file(self, evaluator, sample_binary_data, tmp_path):
        y_true, _, y_pred_proba = sample_binary_data
        save_path = str(tmp_path / "pr.png")

        evaluator.plot_precision_recall_curve(
            y_true, y_pred_proba, save_path=save_path
        )

        assert os.path.exists(save_path)


class TestCompareToBaseline:
    """Tests for compare_to_baseline method."""

    def test_returns_comparison_for_matching_metrics(self, evaluator):
        current = {"accuracy": 0.96, "precision": 0.92}
        baseline = {"accuracy": 0.95, "precision": 0.90}

        comparison = evaluator.compare_to_baseline(current, baseline)

        assert "accuracy" in comparison
        assert "precision" in comparison

    def test_comparison_contains_required_fields(self, evaluator):
        current = {"accuracy": 0.96}
        baseline = {"accuracy": 0.95}

        comparison = evaluator.compare_to_baseline(current, baseline)

        entry = comparison["accuracy"]
        assert "current" in entry
        assert "baseline" in entry
        assert "difference" in entry
        assert "percent_change" in entry
        assert "improved" in entry

    def test_positive_improvement(self, evaluator):
        current = {"accuracy": 0.96}
        baseline = {"accuracy": 0.90}

        comparison = evaluator.compare_to_baseline(current, baseline)

        assert comparison["accuracy"]["current"] == 0.96
        assert comparison["accuracy"]["baseline"] == 0.90
        assert comparison["accuracy"]["difference"] == pytest.approx(0.06)
        assert comparison["accuracy"]["percent_change"] == pytest.approx(
            (0.06 / 0.90) * 100
        )
        assert comparison["accuracy"]["improved"] is True

    def test_negative_change(self, evaluator):
        current = {"accuracy": 0.85}
        baseline = {"accuracy": 0.90}

        comparison = evaluator.compare_to_baseline(current, baseline)

        assert comparison["accuracy"]["difference"] == pytest.approx(-0.05)
        assert comparison["accuracy"]["improved"] is False

    def test_ignores_metrics_not_in_baseline(self, evaluator):
        current = {"accuracy": 0.96, "f1_score": 0.90}
        baseline = {"accuracy": 0.95}

        comparison = evaluator.compare_to_baseline(current, baseline)

        assert "accuracy" in comparison
        assert "f1_score" not in comparison

    def test_zero_baseline_handles_division(self, evaluator):
        current = {"accuracy": 0.5}
        baseline = {"accuracy": 0.0}

        comparison = evaluator.compare_to_baseline(current, baseline)

        assert comparison["accuracy"]["percent_change"] == 0.0


class TestEvaluateModel:
    """Tests for evaluate_model method."""

    def test_returns_required_keys(self, evaluator):
        y_test = np.array([0] * 5 + [1] * 5)
        model = _make_mock_model_for_labels(y_test, accuracy=0.90)
        X_test = np.random.rand(10, 4)

        results = evaluator.evaluate_model(model, X_test, y_test)

        assert "metrics" in results
        assert "confusion_matrix" in results
        assert "comparison" in results
        assert "meets_production_threshold" in results

    def test_meets_threshold_when_accuracy_high(self, evaluator):
        y_test = np.array([0] * 10 + [1] * 10)
        model = _make_mock_model_for_labels(y_test, accuracy=0.95)
        X_test = np.random.rand(20, 4)

        results = evaluator.evaluate_model(model, X_test, y_test)

        assert results["meets_production_threshold"] is True

    def test_does_not_meet_threshold_when_accuracy_low(self, evaluator):
        y_test = np.array([0] * 10 + [1] * 10)
        model = _make_mock_model_for_labels(y_test, accuracy=0.70)
        X_test = np.random.rand(20, 4)

        results = evaluator.evaluate_model(model, X_test, y_test)

        assert results["meets_production_threshold"] is False

    def test_comparison_none_without_baseline(self, evaluator):
        y_test = np.array([0] * 5 + [1] * 5)
        model = _make_mock_model_for_labels(y_test, accuracy=0.90)
        X_test = np.random.rand(10, 4)

        results = evaluator.evaluate_model(model, X_test, y_test)

        assert results["comparison"] is None

    def test_comparison_populated_with_baseline(self, evaluator):
        y_test = np.array([0] * 5 + [1] * 5)
        model = _make_mock_model_for_labels(y_test, accuracy=0.90)
        X_test = np.random.rand(10, 4)
        baseline = {"accuracy": 0.90, "precision": 0.85}

        results = evaluator.evaluate_model(
            model, X_test, y_test, baseline_metrics=baseline
        )

        assert results["comparison"] is not None
        assert "accuracy" in results["comparison"]

    def test_warns_when_accuracy_below_low_threshold(self, evaluator):
        y_test = np.array([0] * 10 + [1] * 10)
        model = _make_mock_model_for_labels(y_test, accuracy=0.50)
        X_test = np.random.rand(20, 4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluator.evaluate_model(model, X_test, y_test)

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "below the production threshold" in str(user_warnings[0].message)

    def test_no_warning_when_accuracy_above_low_threshold(self, evaluator):
        y_test = np.array([0] * 10 + [1] * 10)
        model = _make_mock_model_for_labels(y_test, accuracy=0.95)
        X_test = np.random.rand(20, 4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluator.evaluate_model(model, X_test, y_test)

            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_confusion_matrix_in_results(self, evaluator):
        y_test = np.array([0] * 5 + [1] * 5)
        model = _make_mock_model_for_labels(y_test, accuracy=0.90)
        X_test = np.random.rand(10, 4)

        results = evaluator.evaluate_model(model, X_test, y_test)

        assert isinstance(results["confusion_matrix"], np.ndarray)
        assert results["confusion_matrix"].shape == (2, 2)
