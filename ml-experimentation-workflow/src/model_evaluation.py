"""
Model evaluation module for ML experimentation workflow.

This module provides the ModelEvaluator class for standardized model
evaluation with metrics calculation, visualization generation, baseline
comparison, and production threshold checking.

Example:
    from model_evaluation import ModelEvaluator

    evaluator = ModelEvaluator()

    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

    # Generate visualizations
    cm = evaluator.plot_confusion_matrix(y_true, y_pred)
    fpr, tpr, auc = evaluator.plot_roc_curve(y_true, y_pred_proba)

    # Compare to baseline
    comparison = evaluator.compare_to_baseline(metrics, baseline_metrics)

    # Full evaluation with threshold checking
    results = evaluator.evaluate_model(model, X_test, y_test, baseline_metrics)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Production threshold constants
PRODUCTION_ACCURACY_THRESHOLD = 0.90
LOW_ACCURACY_WARNING_THRESHOLD = 0.80


class ModelEvaluator:
    """
    Standardized model evaluation framework.

    Provides methods for calculating classification metrics, generating
    evaluation visualizations, comparing against baseline metrics, and
    checking production readiness thresholds.

    Example:
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        print(metrics)
        # {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88,
        #  'f1_score': 0.90, 'auc_roc': 0.96}
    """

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all standard classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            Dictionary with accuracy, precision, recall, f1_score, and auc_roc.

        Example:
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(
                np.array([0, 1, 1, 0]),
                np.array([0, 1, 0, 0]),
                np.array([0.1, 0.9, 0.4, 0.2]),
            )
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
        }

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "confusion_matrix.png",
    ) -> np.ndarray:
        """
        Generate and save a confusion matrix visualization.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: File path to save the figure.

        Returns:
            The confusion matrix as a numpy array.

        Example:
            evaluator = ModelEvaluator()
            cm = evaluator.plot_confusion_matrix(y_true, y_pred, "cm.png")
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(save_path)
        plt.close()

        return cm

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = "roc_curve.png",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate and save an ROC curve visualization.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities for the positive class.
            save_path: File path to save the figure.

        Returns:
            Tuple of (fpr, tpr, auc) where fpr and tpr are arrays and auc
            is the area under the ROC curve.

        Example:
            evaluator = ModelEvaluator()
            fpr, tpr, auc = evaluator.plot_roc_curve(y_true, y_proba)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

        return fpr, tpr, auc

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: str = "pr_curve.png",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate and save a precision-recall curve visualization.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities for the positive class.
            save_path: File path to save the figure.

        Returns:
            Tuple of (precision, recall) arrays.

        Example:
            evaluator = ModelEvaluator()
            precision, recall = evaluator.plot_precision_recall_curve(
                y_true, y_proba
            )
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

        return precision, recall

    def compare_to_baseline(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare current experiment metrics to production baseline.

        Args:
            current_metrics: Metrics from the current experiment.
            baseline_metrics: Baseline metrics from production.

        Returns:
            Dictionary keyed by metric name, each containing current, baseline,
            difference, percent_change, and improved values.

        Example:
            evaluator = ModelEvaluator()
            comparison = evaluator.compare_to_baseline(
                {"accuracy": 0.96, "precision": 0.92},
                {"accuracy": 0.95, "precision": 0.90},
            )
            # comparison["accuracy"]["improved"] == True
        """
        comparison: Dict[str, Dict[str, Any]] = {}
        for metric in current_metrics:
            if metric in baseline_metrics:
                current = current_metrics[metric]
                baseline = baseline_metrics[metric]
                diff = current - baseline
                pct_change = (diff / baseline) * 100 if baseline != 0 else 0.0

                comparison[metric] = {
                    "current": current,
                    "baseline": baseline,
                    "difference": diff,
                    "percent_change": pct_change,
                    "improved": diff > 0,
                }

        return comparison

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Complete model evaluation with metrics, visualizations, and threshold check.

        Generates predictions, calculates all metrics, creates visualizations,
        optionally compares to baseline, and checks production thresholds.

        A warning is issued via ``warnings.warn`` if accuracy falls below 0.80.

        Args:
            model: A trained model with predict and predict_proba methods.
            X_test: Test feature matrix.
            y_test: Test labels.
            baseline_metrics: Optional baseline metrics for comparison.

        Returns:
            Dictionary with keys: metrics, confusion_matrix, comparison,
            meets_production_threshold.

        Example:
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_model(model, X_test, y_test)
            print(results["meets_production_threshold"])
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # Generate visualizations
        cm = self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_precision_recall_curve(y_test, y_pred_proba)

        # Compare to baseline if provided
        comparison = None
        if baseline_metrics is not None:
            comparison = self.compare_to_baseline(metrics, baseline_metrics)

        # Check production threshold
        meets_threshold = metrics["accuracy"] >= PRODUCTION_ACCURACY_THRESHOLD

        # Warn if accuracy is below low threshold
        if metrics["accuracy"] < LOW_ACCURACY_WARNING_THRESHOLD:
            warnings.warn(
                f"Model accuracy {metrics['accuracy']:.4f} is below the "
                f"production threshold of {LOW_ACCURACY_WARNING_THRESHOLD}. "
                "Results are below production quality.",
                UserWarning,
                stacklevel=2,
            )

        return {
            "metrics": metrics,
            "confusion_matrix": cm,
            "comparison": comparison,
            "meets_production_threshold": meets_threshold,
        }
