"""
Unit tests for hyperparameter tuning module.

Tests grid search, random search, and SageMaker Bayesian optimization
with mocked models and ExperimentTracker.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock sagemaker modules before importing
sys.modules['sagemaker'] = MagicMock()
sys.modules['sagemaker.experiments'] = MagicMock()
sys.modules['sagemaker.tuner'] = MagicMock()

from hyperparameter_tuning import HyperparameterTuner, _evaluate_model


def _make_mock_model(accuracy=0.95):
    """Create a mock model that returns deterministic predictions."""
    model = Mock()
    model.fit = Mock(return_value=model)

    def predict(X):
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        preds = np.ones(n, dtype=int)
        # Introduce some errors to match desired accuracy
        n_errors = int(n * (1 - accuracy))
        preds[:n_errors] = 0
        return preds

    def predict_proba(X):
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        proba = np.column_stack([np.full(n, 0.1), np.full(n, 0.9)])
        return proba

    model.predict = predict
    model.predict_proba = predict_proba
    return model


def _make_model_class(accuracy=0.95):
    """Return a callable that creates mock models with given accuracy."""
    def factory(**kwargs):
        return _make_mock_model(accuracy)
    factory.__name__ = "MockModel"
    return factory


class TestEvaluateModel:
    """Tests for the _evaluate_model helper."""

    def test_returns_all_metrics(self):
        """Test that all standard metrics are returned."""
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
        """Test graceful handling when predict_proba is unavailable."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 1, 0, 1]))
        del model.predict_proba  # Remove predict_proba

        X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_test = np.array([1, 1, 0, 1])

        metrics = _evaluate_model(model, X_test, y_test)
        assert metrics["auc_roc"] == 0.0
        assert metrics["accuracy"] == 1.0



class TestHyperparameterTunerInit:
    """Tests for HyperparameterTuner initialization."""

    def test_init_without_tracker(self):
        """Test initialization without ExperimentTracker."""
        tuner = HyperparameterTuner()
        assert tuner.tracker is None

    def test_init_with_tracker(self):
        """Test initialization with ExperimentTracker."""
        tracker = Mock()
        tuner = HyperparameterTuner(tracker=tracker)
        assert tuner.tracker is tracker


class TestGridSearch:
    """Tests for grid_search method."""

    @pytest.fixture
    def sample_data(self):
        """Create small sample dataset."""
        np.random.seed(42)
        X_train = np.random.rand(20, 3)
        y_train = np.array([0] * 10 + [1] * 10)
        X_test = np.random.rand(10, 3)
        y_test = np.array([0] * 5 + [1] * 5)
        return X_train, y_train, X_test, y_test

    @pytest.fixture
    def mock_tracker(self):
        """Create mock ExperimentTracker."""
        tracker = Mock()
        tracker.start_experiment = Mock(return_value="exp-001")
        tracker.log_parameters = Mock()
        tracker.log_metrics = Mock()
        tracker.close_experiment = Mock()
        return tracker

    def test_grid_search_all_combinations(self, sample_data):
        """Test that grid search evaluates all parameter combinations."""
        X_train, y_train, X_test, y_test = sample_data
        model_class = _make_model_class()
        tuner = HyperparameterTuner()

        param_grid = {
            "param_a": [1, 2],
            "param_b": [10, 20, 30],
        }

        results = tuner.grid_search(
            model_class=model_class,
            param_grid=param_grid,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        # 2 * 3 = 6 combinations
        assert len(results["all_results"]) == 6
        assert results["best_params"] is not None
        assert results["best_score"] > 0

    def test_grid_search_returns_best(self, sample_data):
        """Test that grid search returns the best scoring combination."""
        X_train, y_train, X_test, y_test = sample_data

        def factory(**kwargs):
            model = Mock()
            model.fit = Mock(return_value=model)
            if kwargs.get("param_a") == 2:
                # Perfect predictions
                model.predict = Mock(return_value=y_test.copy())
                proba = np.column_stack([1 - y_test, y_test]).astype(float)
                model.predict_proba = Mock(return_value=proba)
            else:
                # All-zeros predictions (bad)
                model.predict = Mock(return_value=np.zeros_like(y_test))
                model.predict_proba = Mock(return_value=np.column_stack([np.ones(len(y_test)), np.zeros(len(y_test))]))
            return model
        factory.__name__ = "MockModel"

        param_grid = {"param_a": [1, 2]}

        tuner = HyperparameterTuner()
        results = tuner.grid_search(
            model_class=factory,
            param_grid=param_grid,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert results["best_params"]["param_a"] == 2

    def test_grid_search_tracker_integration(self, sample_data, mock_tracker):
        """Test that grid search logs each trial to ExperimentTracker."""
        X_train, y_train, X_test, y_test = sample_data
        model_class = _make_model_class()
        tuner = HyperparameterTuner(tracker=mock_tracker)

        param_grid = {"param_a": [1, 2], "param_b": [10]}

        tuner.grid_search(
            model_class=model_class,
            param_grid=param_grid,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            experiment_name="test-grid",
        )

        # 2 combinations → 2 experiment starts
        assert mock_tracker.start_experiment.call_count == 2
        assert mock_tracker.log_parameters.call_count == 2
        assert mock_tracker.log_metrics.call_count == 2
        assert mock_tracker.close_experiment.call_count == 2

    def test_grid_search_empty_param_grid(self, sample_data):
        """Test grid search with empty parameter grid."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.grid_search(
            model_class=_make_model_class(),
            param_grid={},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert results["all_results"] == []
        assert results["best_params"] == {}
        assert results["best_score"] == 0.0

    def test_grid_search_single_parameter(self, sample_data):
        """Test grid search with a single parameter and single value."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.grid_search(
            model_class=_make_model_class(),
            param_grid={"param_a": [42]},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert len(results["all_results"]) == 1
        assert results["best_params"] == {"param_a": 42}

    def test_grid_search_without_tracker(self, sample_data):
        """Test grid search works without tracker (experiment_id is None)."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.grid_search(
            model_class=_make_model_class(),
            param_grid={"param_a": [1]},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert results["all_results"][0]["experiment_id"] is None

    def test_grid_search_custom_scoring(self, sample_data):
        """Test grid search with a custom scoring metric."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.grid_search(
            model_class=_make_model_class(),
            param_grid={"param_a": [1]},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            scoring="f1",
        )

        # Score should come from f1 metric
        assert "f1" in results["all_results"][0]["metrics"]


class TestRandomSearch:
    """Tests for random_search method."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X_train = np.random.rand(20, 3)
        y_train = np.array([0] * 10 + [1] * 10)
        X_test = np.random.rand(10, 3)
        y_test = np.array([0] * 5 + [1] * 5)
        return X_train, y_train, X_test, y_test

    @pytest.fixture
    def mock_tracker(self):
        tracker = Mock()
        tracker.start_experiment = Mock(return_value="exp-rand-001")
        tracker.log_parameters = Mock()
        tracker.log_metrics = Mock()
        tracker.close_experiment = Mock()
        return tracker

    def test_random_search_n_iter_trials(self, sample_data):
        """Test that random search runs exactly n_iter trials."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.random_search(
            model_class=_make_model_class(),
            param_distributions={"param_a": [1, 2, 3]},
            n_iter=5,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert len(results["all_results"]) == 5

    def test_random_search_callable_distribution(self, sample_data):
        """Test random search with callable distributions."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.random_search(
            model_class=_make_model_class(),
            param_distributions={
                "param_a": lambda: np.random.choice([1, 2, 3]),
            },
            n_iter=3,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert len(results["all_results"]) == 3
        for r in results["all_results"]:
            assert r["params"]["param_a"] in [1, 2, 3]

    def test_random_search_tracker_integration(self, sample_data, mock_tracker):
        """Test that random search logs each trial to ExperimentTracker."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner(tracker=mock_tracker)

        tuner.random_search(
            model_class=_make_model_class(),
            param_distributions={"param_a": [1, 2]},
            n_iter=4,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            experiment_name="test-random",
        )

        assert mock_tracker.start_experiment.call_count == 4
        assert mock_tracker.log_parameters.call_count == 4
        assert mock_tracker.log_metrics.call_count == 4
        assert mock_tracker.close_experiment.call_count == 4

    def test_random_search_zero_iter(self, sample_data):
        """Test random search with n_iter=0."""
        X_train, y_train, X_test, y_test = sample_data
        tuner = HyperparameterTuner()

        results = tuner.random_search(
            model_class=_make_model_class(),
            param_distributions={"param_a": [1]},
            n_iter=0,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert results["all_results"] == []
        assert results["best_score"] == 0.0

    def test_random_search_returns_best(self, sample_data):
        """Test that random search returns the best scoring trial."""
        X_train, y_train, X_test, y_test = sample_data

        def factory(**kwargs):
            model = Mock()
            model.fit = Mock(return_value=model)
            if kwargs.get("param_a") == 99:
                # Perfect predictions matching y_test
                model.predict = Mock(return_value=y_test.copy())
                proba = np.column_stack([1 - y_test, y_test]).astype(float)
                model.predict_proba = Mock(return_value=proba)
            else:
                # All-zeros predictions (bad for the positive class)
                model.predict = Mock(return_value=np.zeros_like(y_test))
                model.predict_proba = Mock(return_value=np.column_stack([np.ones(len(y_test)), np.zeros(len(y_test))]))
            return model
        factory.__name__ = "MockModel"

        tuner = HyperparameterTuner()
        results = tuner.random_search(
            model_class=factory,
            param_distributions={"param_a": [99]},
            n_iter=2,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

        assert results["best_params"]["param_a"] == 99
        assert results["best_score"] > 0.9


class TestBayesianOptimization:
    """Tests for bayesian_optimization method (SageMaker tuning)."""

    @patch('hyperparameter_tuning.SAGEMAKER_AVAILABLE', True)
    @patch('hyperparameter_tuning.SageMakerTuner')
    def test_bayesian_optimization_creates_tuner(self, mock_tuner_class):
        """Test that bayesian_optimization creates and runs a SageMaker tuner."""
        mock_tuner_instance = MagicMock()
        mock_tuner_instance.best_training_job.return_value = "best-job-001"
        mock_tuner_instance.best_estimator.return_value.hyperparameters.return_value = {
            "max_depth": "7",
            "eta": "0.15",
        }
        mock_tuner_instance.latest_tuning_job.name = "tuning-job-001"
        mock_tuner_class.return_value = mock_tuner_instance

        tuner = HyperparameterTuner()
        estimator = Mock()
        ranges = {"max_depth": Mock(), "eta": Mock()}

        results = tuner.bayesian_optimization(
            estimator=estimator,
            objective_metric_name="validation:auc",
            hyperparameter_ranges=ranges,
            max_jobs=10,
            max_parallel_jobs=3,
            train_data_s3="s3://bucket/train",
            validation_data_s3="s3://bucket/validation",
        )

        # Verify tuner was created with correct args
        mock_tuner_class.assert_called_once_with(
            estimator=estimator,
            objective_metric_name="validation:auc",
            hyperparameter_ranges=ranges,
            max_jobs=10,
            max_parallel_jobs=3,
            strategy="Bayesian",
        )

        # Verify fit was called
        mock_tuner_instance.fit.assert_called_once_with(
            {"train": "s3://bucket/train", "validation": "s3://bucket/validation"},
            wait=True,
        )

        assert results["best_params"] == {"max_depth": "7", "eta": "0.15"}
        assert results["best_training_job"] == "best-job-001"
        assert results["tuning_job_name"] == "tuning-job-001"

    @patch('hyperparameter_tuning.SAGEMAKER_AVAILABLE', False)
    @patch('hyperparameter_tuning.SageMakerTuner', None)
    def test_bayesian_optimization_raises_without_sagemaker(self):
        """Test that bayesian_optimization raises when SageMaker is unavailable."""
        tuner = HyperparameterTuner()

        with pytest.raises(RuntimeError, match="SageMaker is not installed"):
            tuner.bayesian_optimization(
                estimator=Mock(),
                objective_metric_name="validation:auc",
                hyperparameter_ranges={},
            )

    @patch('hyperparameter_tuning.SAGEMAKER_AVAILABLE', True)
    @patch('hyperparameter_tuning.SageMakerTuner')
    def test_bayesian_optimization_without_data_paths(self, mock_tuner_class):
        """Test bayesian_optimization with no S3 data paths."""
        mock_tuner_instance = MagicMock()
        mock_tuner_instance.best_training_job.return_value = "job-002"
        mock_tuner_instance.best_estimator.return_value.hyperparameters.return_value = {}
        mock_tuner_instance.latest_tuning_job.name = "tuning-002"
        mock_tuner_class.return_value = mock_tuner_instance

        tuner = HyperparameterTuner()
        tuner.bayesian_optimization(
            estimator=Mock(),
            objective_metric_name="validation:auc",
            hyperparameter_ranges={},
        )

        # fit should be called with empty dict
        mock_tuner_instance.fit.assert_called_once_with({}, wait=True)
