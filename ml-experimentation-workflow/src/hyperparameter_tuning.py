"""
Hyperparameter tuning module for ML experimentation workflow.

This module provides the HyperparameterTuner class for performing grid search,
random search, and Bayesian optimization (via SageMaker) hyperparameter tuning
with ExperimentTracker integration.

Example:
    from experiment_tracking import ExperimentTracker
    from hyperparameter_tuning import HyperparameterTuner
    from xgboost import XGBClassifier

    tracker = ExperimentTracker()
    tuner = HyperparameterTuner(tracker=tracker)

    results = tuner.grid_search(
        model_class=XGBClassifier,
        param_grid={'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        experiment_name="xgboost-grid-search"
    )
    print(results['best_params'])
"""

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import sagemaker
    from sagemaker.tuner import HyperparameterTuner as SageMakerTuner
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False
    sagemaker = None
    SageMakerTuner = None


def _evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """
    Evaluate a trained model and return all standard metrics.

    Args:
        model: A trained model with predict and predict_proba methods.
        X_test: Test feature matrix.
        y_test: Test labels.
        scoring: Primary scoring metric name.

    Returns:
        Dictionary of metric name to value.
    """
    y_pred = model.predict(X_test)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # AUC-ROC requires probability estimates
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        metrics["auc_roc"] = 0.0

    return metrics


class HyperparameterTuner:
    """
    Hyperparameter tuner supporting grid search, random search, and
    SageMaker Bayesian optimization.

    Integrates with ExperimentTracker to log every trial.

    Args:
        tracker: Optional ExperimentTracker instance for logging trials.

    Example:
        tuner = HyperparameterTuner(tracker=tracker)
        results = tuner.grid_search(
            model_class=XGBClassifier,
            param_grid={'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            experiment_name="grid-search-xgb",
        )
    """

    def __init__(self, tracker: Optional[Any] = None) -> None:
        """
        Initialize HyperparameterTuner.

        Args:
            tracker: Optional ExperimentTracker instance. When provided,
                     every trial is logged as an experiment.
        """
        self.tracker = tracker

    # ------------------------------------------------------------------
    # Grid Search
    # ------------------------------------------------------------------
    def grid_search(
        self,
        model_class: Union[Type, Callable[..., Any]],
        param_grid: Dict[str, List[Any]],
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        experiment_name: str = "grid-search",
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Exhaustive grid search over all parameter combinations.

        Args:
            model_class: Model class or factory callable that accepts keyword
                         arguments matching the parameter names.
            param_grid: Dictionary mapping parameter names to lists of values.
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Test feature matrix.
            y_test: Test labels.
            experiment_name: Name used when logging to ExperimentTracker.
            scoring: Primary metric for selecting the best model
                     (accuracy | precision | recall | f1 | auc_roc).

        Returns:
            Dictionary with keys:
                - best_params: parameter dict of the best trial
                - best_score: best primary metric value
                - all_results: list of dicts with params and metrics

        Example:
            results = tuner.grid_search(
                model_class=XGBClassifier,
                param_grid={'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
        """
        if not param_grid:
            return {"best_params": {}, "best_score": 0.0, "all_results": []}

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        all_results: List[Dict[str, Any]] = []
        best_score = -1.0
        best_params: Dict[str, Any] = {}

        for combo in all_combinations:
            params = dict(zip(param_names, combo))

            # Train
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            metrics = _evaluate_model(model, X_test, y_test, scoring)

            # Log to tracker
            experiment_id = None
            if self.tracker is not None:
                experiment_id = self.tracker.start_experiment(
                    experiment_name=experiment_name,
                    algorithm=model_class.__name__ if hasattr(model_class, '__name__') else str(model_class),
                )
                self.tracker.log_parameters(experiment_id, params)
                self.tracker.log_metrics(experiment_id, metrics)
                self.tracker.close_experiment(experiment_id)

            trial_result = {
                "params": params,
                "metrics": metrics,
                "score": metrics.get(scoring, metrics["accuracy"]),
                "experiment_id": experiment_id,
            }
            all_results.append(trial_result)

            if trial_result["score"] > best_score:
                best_score = trial_result["score"]
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # Random Search
    # ------------------------------------------------------------------
    def random_search(
        self,
        model_class: Union[Type, Callable[..., Any]],
        param_distributions: Dict[str, Any],
        n_iter: int,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        experiment_name: str = "random-search",
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Random search sampling from parameter distributions.

        Each distribution value can be:
          - a list  – a random element is chosen uniformly
          - a callable (e.g. ``lambda: random.uniform(0.01, 0.3)``) – called
            each iteration to produce a value

        Args:
            model_class: Model class or factory callable.
            param_distributions: Dict mapping parameter names to lists or
                                 callables that produce random values.
            n_iter: Number of random trials to run.
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Test feature matrix.
            y_test: Test labels.
            experiment_name: Name used when logging to ExperimentTracker.
            scoring: Primary metric for selecting the best model.

        Returns:
            Dictionary with keys:
                - best_params: parameter dict of the best trial
                - best_score: best primary metric value
                - all_results: list of dicts with params and metrics

        Example:
            results = tuner.random_search(
                model_class=XGBClassifier,
                param_distributions={
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': lambda: random.uniform(0.01, 0.3),
                },
                n_iter=10,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
        """
        if n_iter <= 0:
            return {"best_params": {}, "best_score": 0.0, "all_results": []}

        all_results: List[Dict[str, Any]] = []
        best_score = -1.0
        best_params: Dict[str, Any] = {}

        for _ in range(n_iter):
            params: Dict[str, Any] = {}
            for name, dist in param_distributions.items():
                if callable(dist) and not isinstance(dist, list):
                    params[name] = dist()
                elif isinstance(dist, list):
                    params[name] = random.choice(dist)
                else:
                    params[name] = dist

            # Train
            model = model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            metrics = _evaluate_model(model, X_test, y_test, scoring)

            # Log to tracker
            experiment_id = None
            if self.tracker is not None:
                experiment_id = self.tracker.start_experiment(
                    experiment_name=experiment_name,
                    algorithm=model_class.__name__ if hasattr(model_class, '__name__') else str(model_class),
                )
                self.tracker.log_parameters(experiment_id, params)
                self.tracker.log_metrics(experiment_id, metrics)
                self.tracker.close_experiment(experiment_id)

            trial_result = {
                "params": params,
                "metrics": metrics,
                "score": metrics.get(scoring, metrics["accuracy"]),
                "experiment_id": experiment_id,
            }
            all_results.append(trial_result)

            if trial_result["score"] > best_score:
                best_score = trial_result["score"]
                best_params = params

        return {
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # Bayesian Optimization (SageMaker Automatic Model Tuning)
    # ------------------------------------------------------------------
    def bayesian_optimization(
        self,
        estimator: Any,
        objective_metric_name: str,
        hyperparameter_ranges: Dict[str, Any],
        max_jobs: int = 20,
        max_parallel_jobs: int = 5,
        train_data_s3: Optional[str] = None,
        validation_data_s3: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization using SageMaker Automatic Model Tuning.

        Requires the ``sagemaker`` package to be installed.

        Args:
            estimator: A SageMaker estimator (e.g. ``sagemaker.estimator.Estimator``).
            objective_metric_name: Metric to optimize (e.g. ``'validation:auc'``).
            hyperparameter_ranges: Dict of SageMaker parameter range objects
                (``IntegerParameter``, ``ContinuousParameter``, etc.).
            max_jobs: Maximum number of tuning jobs.
            max_parallel_jobs: Maximum parallel jobs.
            train_data_s3: S3 URI for training data.
            validation_data_s3: S3 URI for validation data.

        Returns:
            Dictionary with keys:
                - best_params: best hyperparameters found
                - best_training_job: name of the best training job
                - tuning_job_name: name of the tuning job

        Raises:
            RuntimeError: If the ``sagemaker`` package is not available.

        Example:
            from sagemaker.tuner import IntegerParameter, ContinuousParameter

            results = tuner.bayesian_optimization(
                estimator=xgb_estimator,
                objective_metric_name='validation:auc',
                hyperparameter_ranges={
                    'max_depth': IntegerParameter(3, 10),
                    'eta': ContinuousParameter(0.01, 0.3),
                },
                train_data_s3='s3://bucket/train',
                validation_data_s3='s3://bucket/validation',
            )
        """
        if not SAGEMAKER_AVAILABLE or SageMakerTuner is None:
            raise RuntimeError(
                "SageMaker is not installed. Install it with: pip install sagemaker"
            )

        tuner = SageMakerTuner(
            estimator=estimator,
            objective_metric_name=objective_metric_name,
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            strategy="Bayesian",
        )

        fit_inputs: Dict[str, str] = {}
        if train_data_s3:
            fit_inputs["train"] = train_data_s3
        if validation_data_s3:
            fit_inputs["validation"] = validation_data_s3

        tuner.fit(fit_inputs, wait=True)

        best_training_job = tuner.best_training_job()
        best_params = tuner.best_estimator().hyperparameters()

        return {
            "best_params": best_params,
            "best_training_job": best_training_job,
            "tuning_job_name": tuner.latest_tuning_job.name
            if hasattr(tuner, "latest_tuning_job") and tuner.latest_tuning_job
            else None,
        }
