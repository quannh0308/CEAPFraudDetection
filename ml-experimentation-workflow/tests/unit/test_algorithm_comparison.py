"""
Unit tests for algorithm comparison module.

Tests AlgorithmComparator class including algorithm comparison,
visualization generation, ExperimentTracker integration, and custom algorithms.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock sagemaker modules before importing
sys.modules['sagemaker'] = MagicMock()
sys.modules['sagemaker.experiments'] = MagicMock()

from algorithm_comparison import AlgorithmComparator, _evaluate_model, _get_model_params


def _make_mock_model(accuracy=0.95):
    """Create a mock model that returns deterministic predictions."""
    model = Mock()
    model.fit = Mock(return_value=model)
    model.get_params = Mock(return_value={"param_a": 1})

    def predict(X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        preds = np.ones(n, dtype=int)
        n_errors = int(n * (1 - accuracy))
        preds[:n_errors] = 0
        return preds

    def predict_proba(X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.column_stack([np.full(n, 0.1), np.full(n, 0.9)])

    model.predict = predict
    model.predict_proba = predict_proba
    return model


@pytest.fixture
def sample_data():
    """Create small sample dataset for testing."""
    np.random.seed(42)
    X_train = np.random.rand(30, 4)
    y_train = np.array([0] * 15 + [1] * 15)
    X_test = np.random.rand(10, 4)
    y_test = np.array([0] * 5 + [1] * 5)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def mock_tracker():
    """Create mock ExperimentTracker."""
    tracker = Mock()
    tracker.start_experiment = Mock(return_value="exp-001")
    tracker.log_parameters = Mock()
    tracker.log_metrics = Mock()
    tracker.close_experiment = Mock()
    return tracker


class TestEvaluateModel:
    """Tests for the _evaluate_model helper."""

    def test_returns_all_metrics(self):
        model = _make_mock_model()
        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([1, 1, 1, 1])

        metrics = _evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics

    def test_handles_model_without_predict_proba(self):
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 1, 0, 1]))
        del model.predict_proba

        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([1, 1, 0, 1])

        metrics = _evaluate_model(model, X_test, y_test)
        assert metrics["auc_roc"] == 0.0
        assert metrics["accuracy"] == 1.0


class TestGetModelParams:
    """Tests for _get_model_params helper."""

    def test_extracts_params(self):
        model = Mock()
        model.get_params = Mock(return_value={"a": 1, "b": 2})
        assert _get_model_params(model) == {"a": 1, "b": 2}

    def test_returns_empty_dict_on_error(self):
        model = Mock(spec=[])  # no get_params
        assert _get_model_params(model) == {}


class TestAlgorithmComparatorInit:
    """Tests for AlgorithmComparator initialization."""

    def test_init_without_tracker(self):
        comparator = AlgorithmComparator()
        assert comparator.tracker is None

    def test_init_with_tracker(self):
        tracker = Mock()
        comparator = AlgorithmComparator(tracker=tracker)
        assert comparator.tracker is tracker


class TestCompareAlgorithms:
    """Tests for compare_algorithms method."""

    def test_compare_custom_algorithms(self, sample_data):
        """Test comparison with custom algorithm dict."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        custom_algorithms = {
            "ModelA": _make_mock_model(0.9),
            "ModelB": _make_mock_model(0.8),
        }

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms=custom_algorithms,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df["algorithm"]) == ["ModelA", "ModelB"]
        assert "accuracy" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "auc_roc" in df.columns
        assert "training_time_seconds" in df.columns

    def test_compare_single_algorithm(self, sample_data):
        """Test comparison with a single algorithm."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"Solo": _make_mock_model()},
        )

        assert len(df) == 1
        assert df.iloc[0]["algorithm"] == "Solo"

    def test_training_time_is_recorded(self, sample_data):
        """Test that training time is measured and included."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"Fast": _make_mock_model()},
        )

        assert df.iloc[0]["training_time_seconds"] >= 0

    def test_tracker_integration(self, sample_data, mock_tracker):
        """Test that each algorithm run is logged to ExperimentTracker."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator(tracker=mock_tracker)

        algorithms = {
            "A": _make_mock_model(),
            "B": _make_mock_model(),
            "C": _make_mock_model(),
        }

        comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms=algorithms,
            experiment_name="test-comparison",
        )

        assert mock_tracker.start_experiment.call_count == 3
        assert mock_tracker.log_parameters.call_count == 3
        assert mock_tracker.log_metrics.call_count == 3
        assert mock_tracker.close_experiment.call_count == 3

        # Verify experiment names match
        for call_obj in mock_tracker.start_experiment.call_args_list:
            assert call_obj[1]["experiment_name"] == "test-comparison"

    def test_tracker_receives_correct_algorithm_names(self, sample_data, mock_tracker):
        """Test that tracker receives the correct algorithm name for each run."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator(tracker=mock_tracker)

        comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"XGB": _make_mock_model()},
            experiment_name="algo-test",
        )

        mock_tracker.start_experiment.assert_called_once_with(
            experiment_name="algo-test",
            algorithm="XGB",
        )

    def test_no_tracker_does_not_fail(self, sample_data):
        """Test that comparison works without a tracker."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"M": _make_mock_model()},
        )

        assert len(df) == 1

    @patch('algorithm_comparison._get_default_algorithms')
    def test_default_algorithms_used_when_none(self, mock_defaults, sample_data):
        """Test that default algorithms are used when algorithms=None."""
        X_train, y_train, X_test, y_test = sample_data
        mock_defaults.return_value = {"Default": _make_mock_model()}

        comparator = AlgorithmComparator()
        df = comparator.compare_algorithms(X_train, y_train, X_test, y_test)

        mock_defaults.assert_called_once()
        assert len(df) == 1
        assert df.iloc[0]["algorithm"] == "Default"


class TestVisualizeComparison:
    """Tests for visualize_comparison method."""

    def test_visualize_saves_file(self, sample_data, tmp_path):
        """Test that visualization saves to the specified path."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"A": _make_mock_model(), "B": _make_mock_model()},
        )

        save_path = str(tmp_path / "test_comparison.png")
        comparator.visualize_comparison(df, save_path=save_path)

        assert os.path.exists(save_path)

    def test_visualize_with_single_algorithm(self, sample_data, tmp_path):
        """Test visualization with a single algorithm."""
        X_train, y_train, X_test, y_test = sample_data
        comparator = AlgorithmComparator()

        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"Solo": _make_mock_model()},
        )

        save_path = str(tmp_path / "single_algo.png")
        comparator.visualize_comparison(df, save_path=save_path)

        assert os.path.exists(save_path)

    def test_visualize_default_save_path(self, sample_data, tmp_path, monkeypatch):
        """Test visualization with default save path."""
        X_train, y_train, X_test, y_test = sample_data
        monkeypatch.chdir(tmp_path)

        comparator = AlgorithmComparator()
        df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={"A": _make_mock_model()},
        )

        comparator.visualize_comparison(df)

        assert os.path.exists(tmp_path / "algorithm_comparison.png")
