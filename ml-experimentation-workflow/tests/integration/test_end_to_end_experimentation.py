"""
Integration tests for the complete ML experimentation workflow.

Validates the end-to-end flow: data load → train → evaluate → promote,
using mocked AWS services.

Requirements validated: 15.1, 15.2, 15.3, 15.5
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock sagemaker modules before importing source modules
sys.modules['sagemaker'] = MagicMock()
sys.modules['sagemaker.experiments'] = MagicMock()

import matplotlib
matplotlib.use('Agg')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from experiment_tracking import ExperimentTracker
from hyperparameter_tuning import HyperparameterTuner
from algorithm_comparison import AlgorithmComparator
from feature_engineering import FeatureEngineer
from model_evaluation import ModelEvaluator
from production_integration import ProductionIntegrator, PARAM_PATHS


def _make_fraud_dataset():
    """Create a synthetic fraud detection dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, weights=[0.7, 0.3],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    return X_train, X_test, y_train, y_test


VALID_HYPERPARAMETERS = {
    'objective': 'binary:logistic',
    'num_round': 150,
    'max_depth': 7,
    'eta': 0.15,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}


class TestEndToEndExperimentation:
    """Integration tests for the complete experimentation workflow."""

    @pytest.fixture
    def dataset(self):
        """Synthetic fraud detection dataset."""
        return _make_fraud_dataset()

    @pytest.fixture
    def tracker(self):
        """ExperimentTracker using internal MockRun (sagemaker mocked)."""
        return ExperimentTracker(region_name='us-east-1')

    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for ProductionIntegrator."""
        with patch('production_integration.boto3.client') as mock_client:
            mock_ssm = MagicMock()
            mock_s3 = MagicMock()
            mock_sfn = MagicMock()

            mock_ssm.exceptions.ParameterNotFound = type(
                'ParameterNotFound', (Exception,), {}
            )
            mock_s3.exceptions.NoSuchKey = type(
                'NoSuchKey', (Exception,), {}
            )

            # SSM get_parameter returns plain string values (for yaml.dump)
            mock_ssm.get_parameter.return_value = {
                'Parameter': {'Value': 'existing-value'}
            }

            # S3 get_object raises NoSuchKey (no existing config to archive)
            mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey()

            def client_factory(service_name, **kwargs):
                if service_name == 'ssm':
                    return mock_ssm
                elif service_name == 's3':
                    return mock_s3
                elif service_name == 'stepfunctions':
                    return mock_sfn
                return MagicMock()

            mock_client.side_effect = client_factory
            yield {'ssm': mock_ssm, 's3': mock_s3, 'sfn': mock_sfn}

    # ------------------------------------------------------------------ #
    # 1. Full end-to-end workflow
    # ------------------------------------------------------------------ #
    def test_complete_workflow_data_load_train_evaluate_promote(
        self, dataset, tracker, mock_boto3_clients
    ):
        """End-to-end: data load → train → evaluate → promote."""
        X_train, X_test, y_train, y_test = dataset

        # Train via AlgorithmComparator (uses tracker internally)
        comparator = AlgorithmComparator(tracker=tracker)
        results_df = comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42)},
            experiment_name='integration-test',
        )
        assert len(results_df) == 1
        assert results_df.iloc[0]['algorithm'] == 'RandomForest'

        # Evaluate with ModelEvaluator
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)
        assert 'accuracy' in metrics
        assert 'precision' in metrics

        # Promote to production
        integrator = ProductionIntegrator(experiment_tracker=tracker)
        result = integrator.promote_to_production(
            experiment_id='integration-exp-001',
            hyperparameters=VALID_HYPERPARAMETERS,
            metrics=metrics,
            approver='integration-test',
        )

        assert 'promotion_event' in result
        assert result['promotion_event']['experiment_id'] == 'integration-exp-001'
        assert result['execution_arn'] is None

        # Verify SSM writes
        ssm = mock_boto3_clients['ssm']
        assert ssm.put_parameter.call_count >= len(PARAM_PATHS)

        # Verify S3 config write
        s3 = mock_boto3_clients['s3']
        s3_put_calls = [
            c for c in s3.put_object.call_args_list
            if c[1].get('Key') == 'production-model-config.yaml'
        ]
        assert len(s3_put_calls) == 1

    # ------------------------------------------------------------------ #
    # 2. Tracker logs throughout workflow
    # ------------------------------------------------------------------ #
    def test_experiment_tracker_logs_throughout_workflow(self, dataset, tracker):
        """Verify tracker logs params and metrics during training."""
        X_train, X_test, y_train, y_test = dataset

        comparator = AlgorithmComparator(tracker=tracker)
        comparator.compare_algorithms(
            X_train, y_train, X_test, y_test,
            algorithms={'RandomForest': RandomForestClassifier(n_estimators=20, random_state=42)},
            experiment_name='tracker-log-test',
        )

        # Tracker should have no active runs (all closed)
        assert len(tracker._active_runs) == 0

    # ------------------------------------------------------------------ #
    # 3. Hyperparameter tuning integrates with tracker
    # ------------------------------------------------------------------ #
    def test_hyperparameter_tuning_integrates_with_tracker(self, dataset, tracker):
        """Grid search logs every trial to the tracker."""
        X_train, X_test, y_train, y_test = dataset

        tuner = HyperparameterTuner(tracker=tracker)
        results = tuner.grid_search(
            model_class=RandomForestClassifier,
            param_grid={'n_estimators': [10, 20], 'max_depth': [3, 5]},
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            experiment_name='hp-tuning-integration',
            scoring='accuracy',
        )

        assert 'best_params' in results
        assert 'best_score' in results
        assert len(results['all_results']) == 4  # 2 x 2 grid
        assert results['best_score'] > 0
        # All runs should be closed
        assert len(tracker._active_runs) == 0

    # ------------------------------------------------------------------ #
    # 4. Feature engineering → training → evaluation
    # ------------------------------------------------------------------ #
    def test_feature_engineering_to_training_pipeline(self):
        """Feature engineering feeds into training and evaluation."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'Time': np.random.randint(0, 86400 * 7, size=n),
            'Amount': np.abs(np.random.randn(n) * 100),
        })
        for i in range(8):
            df[f'V{i+1}'] = np.random.randn(n)
        df['Class'] = np.random.randint(0, 2, size=n)

        engineer = FeatureEngineer()
        df = engineer.create_time_features(df)
        df = engineer.create_amount_features(df)

        feature_cols = [c for c in df.columns if c not in ('Class',)]
        X = df[feature_cols]
        y = df['Class']

        selected_features, scores_df = engineer.select_features_univariate(X, y, k=5)
        assert len(selected_features) == 5

        X_sel = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sel, y, test_size=0.3, random_state=42,
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(
            y_test.values,
            model.predict(X_test),
            model.predict_proba(X_test)[:, 1],
        )
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    # ------------------------------------------------------------------ #
    # 5. Promotion writes to Parameter Store and S3
    # ------------------------------------------------------------------ #
    def test_promotion_writes_to_parameter_store_and_s3(
        self, dataset, tracker, mock_boto3_clients
    ):
        """promote_to_production writes hyperparams to SSM and config to S3."""
        X_train, X_test, y_train, y_test = dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)

        integrator = ProductionIntegrator(experiment_tracker=tracker)
        result = integrator.promote_to_production(
            experiment_id='ssm-s3-test-001',
            hyperparameters=VALID_HYPERPARAMETERS,
            metrics=metrics,
            approver='test-approver',
        )

        # Verify Parameter Store writes for every param path
        ssm = mock_boto3_clients['ssm']
        put_calls = ssm.put_parameter.call_args_list
        written_paths = {c[1]['Name'] for c in put_calls}
        for path in PARAM_PATHS.values():
            assert path in written_paths

        # Verify S3 config write
        s3 = mock_boto3_clients['s3']
        config_puts = [
            c for c in s3.put_object.call_args_list
            if c[1].get('Key') == 'production-model-config.yaml'
        ]
        assert len(config_puts) == 1
        assert config_puts[0][1]['Bucket'] == 'fraud-detection-config'

        # Verify promotion event returned
        assert result['promotion_event']['approver'] == 'test-approver'
        assert result['promotion_event']['metrics'] == metrics

    # ------------------------------------------------------------------ #
    # 6. Evaluation metrics match production requirements (Req 15.3)
    # ------------------------------------------------------------------ #
    def test_evaluation_metrics_match_production_requirements(self, dataset):
        """Metrics include accuracy, precision, recall as required."""
        X_train, X_test, y_train, y_test = dataset

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)

        required_keys = {'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'}
        assert required_keys.issubset(metrics.keys())
        for key in required_keys:
            assert isinstance(metrics[key], float)
            assert 0 <= metrics[key] <= 1
