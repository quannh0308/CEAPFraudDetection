"""
Experiment tracking module for ML experimentation workflow.

This module provides the ExperimentTracker class for logging and versioning
experiments using SageMaker Experiments. It supports creating experiment runs
with unique IDs, logging hyperparameters and metrics, uploading artifacts to
S3, and querying past experiments with flexible filters.

Key capabilities:
    - Create experiment runs with unique IDs and automatic timestamps
    - Log hyperparameters, performance metrics, and model artifacts
    - Query experiments by date range, metric thresholds, or hyperparameter values
    - Integrate with SageMaker Experiments for native AWS tracking

Example:
    from experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker()
    experiment_id = tracker.start_experiment(
        experiment_name="fraud-detection-optimization",
        algorithm="xgboost",
        dataset_version="v1.2.3",
    )
    tracker.log_parameters(experiment_id, {"max_depth": 7, "eta": 0.15})
    tracker.log_metrics(experiment_id, {"accuracy": 0.961, "auc_roc": 0.95})
    tracker.close_experiment(experiment_id)
"""

import boto3
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

try:
    import sagemaker
    from sagemaker.experiments import Run
    SAGEMAKER_AVAILABLE = True
except ImportError:
    # For testing without SageMaker installed
    SAGEMAKER_AVAILABLE = False
    sagemaker = None
    Run = None


class ExperimentTracker:
    """
    Experiment tracker with SageMaker Experiments integration.
    
    This class provides methods for creating experiment runs, logging parameters,
    metrics, and artifacts, and querying experiments with various filters.
    
    Example:
        tracker = ExperimentTracker()
        experiment_id = tracker.start_experiment(
            experiment_name="fraud-detection-optimization",
            algorithm="xgboost"
        )
        tracker.log_parameters(experiment_id, {"max_depth": 7, "eta": 0.15})
        tracker.log_metrics(experiment_id, {"accuracy": 0.961, "auc_roc": 0.95})
    """
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize ExperimentTracker with SageMaker session.
        
        Args:
            region_name: AWS region name for SageMaker session.
        
        Example:
            tracker = ExperimentTracker()
            tracker = ExperimentTracker(region_name='us-west-2')
        """
        if SAGEMAKER_AVAILABLE and sagemaker is not None:
            self.sagemaker_session = sagemaker.Session(boto3.Session(region_name=region_name))
        else:
            self.sagemaker_session = None
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.region_name = region_name
        self._active_runs: Dict[str, Any] = {}
    
    def start_experiment(
        self,
        experiment_name: str,
        algorithm: str,
        user: Optional[str] = None,
        dataset_version: Optional[str] = None,
        code_version: Optional[str] = None
    ) -> str:
        """
        Create experiment run with unique ID.
        
        Args:
            experiment_name: Name of the experiment (e.g., "fraud-detection-optimization")
            algorithm: Algorithm being used (e.g., "xgboost", "lightgbm")
            user: User running the experiment (defaults to current user)
            dataset_version: Version of the dataset being used
            code_version: Version of the code being used
        
        Returns:
            Unique experiment ID for this run
        
        Example:
            experiment_id = tracker.start_experiment(
                experiment_name="fraud-detection-optimization",
                algorithm="xgboost",
                dataset_version="v1.2.3"
            )
        """
        # Generate unique experiment ID with timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        experiment_id = f"{algorithm}-{timestamp}-{unique_id}"
        
        # Create run name
        run_name = f"{experiment_name}-{experiment_id}"
        
        # Create SageMaker Experiments run
        if SAGEMAKER_AVAILABLE and Run is not None:
            run = Run(
                experiment_name=experiment_name,
                run_name=run_name,
                sagemaker_session=self.sagemaker_session
            )
        else:
            # For testing: create a mock-compatible object
            run = type('MockRun', (), {
                'log_parameter': lambda *args, **kwargs: None,
                'log_metric': lambda *args, **kwargs: None,
                'log_file': lambda *args, **kwargs: None,
                'close': lambda *args, **kwargs: None
            })()
        
        # Store active run
        self._active_runs[experiment_id] = run
        
        # Log initial metadata
        run.log_parameter("experiment_id", experiment_id)
        run.log_parameter("algorithm", algorithm)
        run.log_parameter("start_timestamp", datetime.now().isoformat())
        
        if user:
            run.log_parameter("user", user)
        if dataset_version:
            run.log_parameter("dataset_version", dataset_version)
        if code_version:
            run.log_parameter("code_version", code_version)
        
        return experiment_id
    
    def log_parameters(self, experiment_id: str, parameters: Dict[str, Any]) -> None:
        """
        Log hyperparameters for an experiment.
        
        Args:
            experiment_id: Unique experiment ID from start_experiment
            parameters: Dictionary of hyperparameters to log
        
        Example:
            tracker.log_parameters(experiment_id, {
                "max_depth": 7,
                "eta": 0.15,
                "num_round": 150,
                "subsample": 0.8
            })
        """
        if experiment_id not in self._active_runs:
            raise ValueError(f"Experiment {experiment_id} not found. Call start_experiment first.")
        
        run = self._active_runs[experiment_id]
        
        for param_name, param_value in parameters.items():
            run.log_parameter(param_name, param_value)
    
    def log_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        """
        Log performance metrics for an experiment.
        
        Args:
            experiment_id: Unique experiment ID from start_experiment
            metrics: Dictionary of metrics to log (e.g., accuracy, precision, recall)
        
        Example:
            tracker.log_metrics(experiment_id, {
                "accuracy": 0.961,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90,
                "auc_roc": 0.95
            })
        """
        if experiment_id not in self._active_runs:
            raise ValueError(f"Experiment {experiment_id} not found. Call start_experiment first.")
        
        run = self._active_runs[experiment_id]
        
        for metric_name, metric_value in metrics.items():
            run.log_metric(metric_name, metric_value)
    
    def log_artifacts(
        self,
        experiment_id: str,
        artifact_paths: List[str],
        s3_bucket: str,
        s3_prefix: Optional[str] = None
    ) -> List[str]:
        """
        Upload artifact files to S3 and log to experiment.
        
        Args:
            experiment_id: Unique experiment ID from start_experiment
            artifact_paths: List of local file paths to upload
            s3_bucket: S3 bucket name for artifact storage
            s3_prefix: Optional S3 prefix (folder) for artifacts
        
        Returns:
            List of S3 URIs for uploaded artifacts
        
        Example:
            s3_uris = tracker.log_artifacts(
                experiment_id,
                ["model.pkl", "confusion_matrix.png"],
                s3_bucket="fraud-detection-models",
                s3_prefix="experiments"
            )
        """
        if experiment_id not in self._active_runs:
            raise ValueError(f"Experiment {experiment_id} not in active runs. Call start_experiment first.")
        
        run = self._active_runs[experiment_id]
        s3_uris = []
        
        # Determine S3 prefix
        if s3_prefix:
            full_prefix = f"{s3_prefix}/{experiment_id}"
        else:
            full_prefix = experiment_id
        
        # Upload each artifact
        for artifact_path in artifact_paths:
            # Extract filename from path
            filename = artifact_path.split('/')[-1]
            s3_key = f"{full_prefix}/{filename}"
            
            # Upload to S3
            self.s3_client.upload_file(artifact_path, s3_bucket, s3_key)
            
            # Create S3 URI
            s3_uri = f"s3://{s3_bucket}/{s3_key}"
            s3_uris.append(s3_uri)
            
            # Log artifact to run
            run.log_file(artifact_path, name=filename, is_output=True)
        
        return s3_uris
    
    def query_experiments(
        self,
        experiment_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_accuracy: Optional[float] = None,
        hyperparameter_filters: Optional[Dict[str, Any]] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query experiments with filtering by date, metrics, and hyperparameters.
        
        Args:
            experiment_name: Name of the experiment to query
            start_date: Filter experiments after this date
            end_date: Filter experiments before this date
            min_accuracy: Filter experiments with accuracy >= this value
            hyperparameter_filters: Dictionary of hyperparameter filters
            max_results: Maximum number of results to return
        
        Returns:
            List of experiment metadata dictionaries
        
        Example:
            # Find experiments with accuracy >= 0.95
            results = tracker.query_experiments(
                experiment_name="fraud-detection-optimization",
                min_accuracy=0.95
            )
            
            # Find experiments with specific hyperparameters
            results = tracker.query_experiments(
                experiment_name="fraud-detection-optimization",
                hyperparameter_filters={"max_depth": 7, "eta": 0.15}
            )
        """
        from sagemaker.experiments import search_expression
        
        # Build search filters
        filters = []
        
        # Date filters
        if start_date:
            filters.append(
                search_expression.Filter(
                    name="CreationTime",
                    operator=search_expression.Operator.GREATER_THAN_OR_EQUAL,
                    value=start_date.isoformat()
                )
            )
        
        if end_date:
            filters.append(
                search_expression.Filter(
                    name="CreationTime",
                    operator=search_expression.Operator.LESS_THAN_OR_EQUAL,
                    value=end_date.isoformat()
                )
            )
        
        # Accuracy filter
        if min_accuracy is not None:
            filters.append(
                search_expression.Filter(
                    name="Metrics.accuracy.Max",
                    operator=search_expression.Operator.GREATER_THAN_OR_EQUAL,
                    value=str(min_accuracy)
                )
            )
        
        # Hyperparameter filters
        if hyperparameter_filters:
            for param_name, param_value in hyperparameter_filters.items():
                filters.append(
                    search_expression.Filter(
                        name=f"Parameters.{param_name}",
                        operator=search_expression.Operator.EQUALS,
                        value=str(param_value)
                    )
                )
        
        # Create search expression
        if filters:
            search_exp = search_expression.SearchExpression(filters=filters)
        else:
            search_exp = None
        
        # Execute search
        search_results = sagemaker.experiments.search(
            search_expression=search_exp,
            sort_by="CreationTime",
            sort_order="Descending",
            max_results=max_results,
            sagemaker_session=self.sagemaker_session
        )
        
        # Format results
        experiments = []
        for result in search_results:
            experiment_data = {
                'experiment_id': result.get('Parameters', {}).get('experiment_id'),
                'run_name': result.get('TrialComponentName'),
                'start_timestamp': result.get('CreationTime'),
                'algorithm': result.get('Parameters', {}).get('algorithm'),
                'hyperparameters': {},
                'metrics': {},
                'dataset_version': result.get('Parameters', {}).get('dataset_version'),
                'code_version': result.get('Parameters', {}).get('code_version')
            }
            
            # Extract hyperparameters
            for param_name, param_value in result.get('Parameters', {}).items():
                if param_name not in ['experiment_id', 'algorithm', 'start_timestamp', 
                                      'user', 'dataset_version', 'code_version']:
                    experiment_data['hyperparameters'][param_name] = param_value
            
            # Extract metrics
            for metric_name, metric_data in result.get('Metrics', {}).items():
                experiment_data['metrics'][metric_name] = metric_data.get('Max')
            
            experiments.append(experiment_data)
        
        return experiments
    
    def close_experiment(self, experiment_id: str) -> None:
        """
        Close an experiment run and clean up resources.
        
        Args:
            experiment_id: Unique experiment ID to close.
        
        Example:
            tracker.close_experiment(experiment_id)
        """
        if experiment_id in self._active_runs:
            run = self._active_runs[experiment_id]
            run.close()
            del self._active_runs[experiment_id]
