"""
Algorithm comparison module for ML experimentation workflow.

This module provides the AlgorithmComparator class for training and comparing
multiple ML algorithms on the same dataset, with optional ExperimentTracker
integration and visualization utilities.

Example:
    from experiment_tracking import ExperimentTracker
    from algorithm_comparison import AlgorithmComparator

    tracker = ExperimentTracker()
    comparator = AlgorithmComparator(tracker=tracker)

    results_df = comparator.compare_algorithms(
        X_train, y_train, X_test, y_test,
        experiment_name="fraud-detection-comparison"
    )
    comparator.visualize_comparison(results_df, save_path="comparison.png")
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def _evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a trained model and return standard classification metrics.

    Args:
        model: A trained model with predict and predict_proba methods.
        X_test: Test feature matrix.
        y_test: Test labels.

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

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc_roc"] = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        metrics["auc_roc"] = 0.0

    return metrics


def _get_default_algorithms() -> Dict[str, Any]:
    """
    Return the default set of algorithms for comparison.

    Returns:
        Dictionary mapping algorithm name to model instance.
    """
    algorithms: Dict[str, Any] = {}

    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        algorithms["XGBoost"] = XGBClassifier(
            max_depth=5,
            learning_rate=0.2,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    if LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        algorithms["LightGBM"] = LGBMClassifier(
            max_depth=5,
            learning_rate=0.2,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )

    algorithms["RandomForest"] = RandomForestClassifier(
        max_depth=10,
        n_estimators=100,
        max_features="sqrt",
    )

    algorithms["NeuralNetwork"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=200,
    )

    return algorithms


class AlgorithmComparator:
    """
    Compare multiple ML algorithms on the same dataset.

    Trains each algorithm, calculates metrics, and optionally logs results
    to an ExperimentTracker.

    Args:
        tracker: Optional ExperimentTracker instance for logging runs.

    Example:
        comparator = AlgorithmComparator(tracker=tracker)
        df = comparator.compare_algorithms(X_train, y_train, X_test, y_test)
        comparator.visualize_comparison(df)
    """

    def __init__(self, tracker: Optional[Any] = None) -> None:
        self.tracker = tracker

    def compare_algorithms(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        algorithms: Optional[Dict[str, Any]] = None,
        experiment_name: str = "algorithm-comparison",
    ) -> pd.DataFrame:
        """
        Train and evaluate multiple algorithms, returning a comparison DataFrame.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Test feature matrix.
            y_test: Test labels.
            algorithms: Optional dict mapping algorithm name to model instance
                        or class. When None, default algorithms are used.
            experiment_name: Name used when logging to ExperimentTracker.

        Returns:
            pandas DataFrame with one row per algorithm and columns for each
            metric plus training_time_seconds.

        Example:
            df = comparator.compare_algorithms(
                X_train, y_train, X_test, y_test,
                algorithms={"MyModel": MyModelClass()},
            )
        """
        if algorithms is None:
            algorithms = _get_default_algorithms()

        results: List[Dict[str, Any]] = []

        for name, model in algorithms.items():
            # Train and measure time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Evaluate
            metrics = _evaluate_model(model, X_test, y_test)
            metrics["training_time_seconds"] = training_time

            # Log to tracker if available
            if self.tracker is not None:
                experiment_id = self.tracker.start_experiment(
                    experiment_name=experiment_name,
                    algorithm=name,
                )
                self.tracker.log_parameters(experiment_id, _get_model_params(model))
                self.tracker.log_metrics(experiment_id, metrics)
                self.tracker.close_experiment(experiment_id)

            row = {"algorithm": name}
            row.update(metrics)
            results.append(row)

        return pd.DataFrame(results)

    def visualize_comparison(
        self,
        results_df: pd.DataFrame,
        save_path: str = "algorithm_comparison.png",
    ) -> None:
        """
        Generate bar charts comparing metrics across algorithms.

        Args:
            results_df: DataFrame returned by compare_algorithms.
            save_path: File path to save the figure.

        Example:
            comparator.visualize_comparison(df, save_path="comparison.png")
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        metrics = ["accuracy", "precision", "recall", "f1", "auc_roc", "training_time_seconds"]
        available_metrics = [m for m in metrics if m in results_df.columns]

        n_metrics = len(available_metrics)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx // ncols, idx % ncols]
            sns.barplot(data=results_df, x="algorithm", y=metric, ax=ax)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

        # Hide unused axes
        for idx in range(n_metrics, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


def _get_model_params(model: Any) -> Dict[str, Any]:
    """Extract parameters from a model, falling back to an empty dict."""
    try:
        return model.get_params()
    except (AttributeError, TypeError):
        return {}
