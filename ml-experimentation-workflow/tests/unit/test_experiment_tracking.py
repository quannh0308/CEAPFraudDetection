"""
Unit tests for experiment tracking module.

Tests experiment creation, parameter/metric logging, and query functionality
with mocked SageMaker Experiments.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


# Mock sagemaker modules before importing
sys.modules['sagemaker'] = MagicMock()
sys.modules['sagemaker.experiments'] = MagicMock()

from experiment_tracking import ExperimentTracker


class TestExperimentTracker:
    """Test suite for ExperimentTracker class."""
    
    @pytest.fixture
    def mock_sagemaker_session(self):
        """Create mock SageMaker session."""
        with patch('experiment_tracking.sagemaker.Session') as mock_session:
            yield mock_session.return_value
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        with patch('experiment_tracking.boto3.client') as mock_client:
            yield mock_client.return_value
    
    @pytest.fixture
    def mock_run(self):
        """Create mock Run object."""
        with patch('experiment_tracking.Run') as mock_run_class:
            mock_run_instance = MagicMock()
            mock_run_class.return_value = mock_run_instance
            yield mock_run_instance
    
    @pytest.fixture
    def tracker(self, mock_sagemaker_session, mock_s3_client):
        """Create ExperimentTracker instance with mocked dependencies."""
        return ExperimentTracker(region_name='us-east-1')
    
    def test_init(self, tracker, mock_sagemaker_session):
        """Test ExperimentTracker initialization."""
        assert tracker.sagemaker_session is not None
        assert tracker.s3_client is not None
        assert tracker.region_name == 'us-east-1'
        assert tracker._active_runs == {}
    
    def test_start_experiment_creates_unique_id(self, tracker, mock_run):
        """Test that start_experiment creates a unique experiment ID."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        # Verify ID format: algorithm-timestamp-uuid
        assert experiment_id.startswith("xgboost-")
        assert len(experiment_id.split('-')) >= 4  # algorithm-YYYYMMDD-HHMMSS-uuid
        
        # Verify run was created and stored
        assert experiment_id in tracker._active_runs
    
    def test_start_experiment_logs_metadata(self, tracker, mock_run):
        """Test that start_experiment logs initial metadata."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost",
            user="test-user",
            dataset_version="v1.0.0",
            code_version="abc123"
        )
        
        # Verify log_parameter was called with correct values
        run = tracker._active_runs[experiment_id]
        
        # Check that parameters were logged
        assert run.log_parameter.called
        
        # Verify specific parameters
        call_args_list = [call[0] for call in run.log_parameter.call_args_list]
        param_names = [args[0] for args in call_args_list]
        
        assert "experiment_id" in param_names
        assert "algorithm" in param_names
        assert "start_timestamp" in param_names
        assert "user" in param_names
        assert "dataset_version" in param_names
        assert "code_version" in param_names
    
    def test_log_parameters(self, tracker, mock_run):
        """Test parameter logging."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        parameters = {
            "max_depth": 7,
            "eta": 0.15,
            "num_round": 150,
            "subsample": 0.8
        }
        
        tracker.log_parameters(experiment_id, parameters)
        
        run = tracker._active_runs[experiment_id]
        
        # Verify each parameter was logged
        for param_name, param_value in parameters.items():
            run.log_parameter.assert_any_call(param_name, param_value)
    
    def test_log_parameters_invalid_experiment(self, tracker):
        """Test that logging parameters for invalid experiment raises error."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            tracker.log_parameters("invalid-id", {"max_depth": 7})
    
    def test_log_metrics(self, tracker, mock_run):
        """Test metric logging."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        metrics = {
            "accuracy": 0.961,
            "precision": 0.92,
            "recall": 0.88,
            "f1_score": 0.90,
            "auc_roc": 0.95
        }
        
        tracker.log_metrics(experiment_id, metrics)
        
        run = tracker._active_runs[experiment_id]
        
        # Verify each metric was logged
        for metric_name, metric_value in metrics.items():
            run.log_metric.assert_any_call(metric_name, metric_value)
    
    def test_log_metrics_invalid_experiment(self, tracker):
        """Test that logging metrics for invalid experiment raises error."""
        with pytest.raises(ValueError, match="Experiment .* not found"):
            tracker.log_metrics("invalid-id", {"accuracy": 0.95})
    
    def test_log_artifacts(self, tracker, mock_run, mock_s3_client):
        """Test artifact logging and S3 upload."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        artifact_paths = ["model.pkl", "confusion_matrix.png"]
        s3_bucket = "test-bucket"
        s3_prefix = "experiments"
        
        s3_uris = tracker.log_artifacts(
            experiment_id,
            artifact_paths,
            s3_bucket,
            s3_prefix
        )
        
        # Verify S3 uploads
        assert tracker.s3_client.upload_file.call_count == 2
        
        # Verify S3 URIs
        assert len(s3_uris) == 2
        assert s3_uris[0] == f"s3://{s3_bucket}/{s3_prefix}/{experiment_id}/model.pkl"
        assert s3_uris[1] == f"s3://{s3_bucket}/{s3_prefix}/{experiment_id}/confusion_matrix.png"
        
        # Verify artifacts logged to run
        run = tracker._active_runs[experiment_id]
        assert run.log_file.call_count == 2
    
    def test_log_artifacts_without_prefix(self, tracker, mock_run, mock_s3_client):
        """Test artifact logging without S3 prefix."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        artifact_paths = ["model.pkl"]
        s3_bucket = "test-bucket"
        
        s3_uris = tracker.log_artifacts(
            experiment_id,
            artifact_paths,
            s3_bucket
        )
        
        # Verify S3 URI without prefix
        assert s3_uris[0] == f"s3://{s3_bucket}/{experiment_id}/model.pkl"
    
    def test_log_artifacts_invalid_experiment(self, tracker):
        """Test that logging artifacts for invalid experiment raises error."""
        with pytest.raises(ValueError, match="not in active runs"):
            tracker.log_artifacts("invalid-id", ["model.pkl"], "test-bucket")
    
    @patch('experiment_tracking.sagemaker.experiments.search')
    def test_query_experiments_no_filters(self, mock_search, tracker):
        """Test querying experiments without filters."""
        # Mock search results
        mock_search.return_value = [
            {
                'TrialComponentName': 'test-run-1',
                'CreationTime': datetime(2024, 1, 15, 10, 0, 0),
                'Parameters': {
                    'experiment_id': 'xgboost-20240115-001',
                    'algorithm': 'xgboost',
                    'max_depth': '7',
                    'eta': '0.15'
                },
                'Metrics': {
                    'accuracy': {'Max': 0.961},
                    'auc_roc': {'Max': 0.95}
                }
            }
        ]
        
        results = tracker.query_experiments(
            experiment_name="test-experiment"
        )
        
        assert len(results) == 1
        assert results[0]['experiment_id'] == 'xgboost-20240115-001'
        assert results[0]['algorithm'] == 'xgboost'
        assert results[0]['hyperparameters']['max_depth'] == '7'
        assert results[0]['metrics']['accuracy'] == 0.961
    
    @patch('experiment_tracking.sagemaker.experiments.search')
    def test_query_experiments_with_accuracy_filter(self, mock_search, tracker):
        """Test querying experiments with minimum accuracy filter."""
        mock_search.return_value = []
        
        tracker.query_experiments(
            experiment_name="test-experiment",
            min_accuracy=0.95
        )
        
        # Verify search was called
        assert mock_search.called
        call_args = mock_search.call_args
        
        # Verify search_expression was passed (filters were applied)
        assert 'search_expression' in call_args[1]
        search_exp = call_args[1]['search_expression']
        assert search_exp is not None
    
    @patch('experiment_tracking.sagemaker.experiments.search')
    def test_query_experiments_with_date_filters(self, mock_search, tracker):
        """Test querying experiments with date range filters."""
        mock_search.return_value = []
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        tracker.query_experiments(
            experiment_name="test-experiment",
            start_date=start_date,
            end_date=end_date
        )
        
        # Verify search was called
        assert mock_search.called
        call_args = mock_search.call_args
        
        # Verify search_expression was passed (filters were applied)
        assert 'search_expression' in call_args[1]
        search_exp = call_args[1]['search_expression']
        assert search_exp is not None
    
    @patch('experiment_tracking.sagemaker.experiments.search')
    def test_query_experiments_with_hyperparameter_filters(self, mock_search, tracker):
        """Test querying experiments with hyperparameter filters."""
        mock_search.return_value = []
        
        hyperparameter_filters = {
            "max_depth": 7,
            "eta": 0.15
        }
        
        tracker.query_experiments(
            experiment_name="test-experiment",
            hyperparameter_filters=hyperparameter_filters
        )
        
        # Verify search was called
        assert mock_search.called
        call_args = mock_search.call_args
        
        # Verify search_expression was passed (filters were applied)
        assert 'search_expression' in call_args[1]
        search_exp = call_args[1]['search_expression']
        assert search_exp is not None
    
    def test_close_experiment(self, tracker, mock_run):
        """Test closing an experiment."""
        experiment_id = tracker.start_experiment(
            experiment_name="test-experiment",
            algorithm="xgboost"
        )
        
        assert experiment_id in tracker._active_runs
        
        tracker.close_experiment(experiment_id)
        
        # Verify run was closed and removed
        assert experiment_id not in tracker._active_runs
        mock_run.close.assert_called_once()
    
    def test_close_nonexistent_experiment(self, tracker):
        """Test closing a nonexistent experiment (should not raise error)."""
        # Should not raise an error
        tracker.close_experiment("nonexistent-id")
