"""
A/B testing module for comparing challenger models against production champions.

This module provides utilities for deploying challenger model endpoints,
configuring traffic splits for gradual rollout, comparing endpoint performance,
and promoting winning challengers to production. It integrates with AWS SageMaker
for endpoint management and inference.

Key capabilities:
    - Deploy challenger models to separate SageMaker endpoints
    - Generate traffic split configurations with staged rollout plans
    - Compare champion and challenger endpoint latency and predictions
    - Promote challenger models to the production champion endpoint

Example:
    from ab_testing import ABTestingManager

    manager = ABTestingManager()

    # Deploy a challenger endpoint
    challenger = manager.deploy_challenger_endpoint(
        experiment_id='exp-20240115-001',
        model_artifact_s3_path='s3://fraud-detection-models/exp-20240115-001/model.tar.gz'
    )

    # Compare endpoints
    comparison, champ_preds, chal_preds = manager.compare_endpoints(
        champion_endpoint='fraud-detection-production',
        challenger_endpoint=challenger,
        test_data=[{'feature1': 0.5}]
    )
"""

import json
import time

import boto3
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# SageMaker XGBoost container image URI
XGBOOST_IMAGE_URI: str = '382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest'

# IAM execution role for SageMaker
SAGEMAKER_EXECUTION_ROLE: str = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'

# Production endpoint name
PRODUCTION_ENDPOINT: str = 'fraud-detection-production'


# Traffic split rollout stages (challenger traffic percentages)
ROLLOUT_STAGES: List[Dict[str, int]] = [
    {'stage': 1, 'challenger_traffic': 1, 'duration_hours': 24},
    {'stage': 2, 'challenger_traffic': 10, 'duration_hours': 48},
    {'stage': 3, 'challenger_traffic': 50, 'duration_hours': 72},
    {'stage': 4, 'challenger_traffic': 100, 'duration_hours': 0},
]

# Success criteria for A/B tests
SUCCESS_CRITERIA: Dict[str, float] = {
    'min_accuracy': 0.90,
    'max_latency_ms': 100,
    'min_improvement_pct': 1.0,
}


class ABTestingManager:
    """
    Utilities for A/B testing challenger models against production champions.

    Provides methods to deploy challenger endpoints, generate traffic split
    configurations, compare endpoint performance, and promote challengers
    to production.

    Example:
        manager = ABTestingManager()
        challenger = manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz'
        )
        config = manager.generate_traffic_split_config(
            champion_endpoint='fraud-detection-production',
            challenger_endpoint=challenger
        )
    """

    def __init__(self) -> None:
        """
        Initialize ABTestingManager with boto3 SageMaker and S3 clients.

        Example:
            manager = ABTestingManager()
        """
        self.sagemaker_client = boto3.client('sagemaker')
        self.s3_client = boto3.client('s3')

    def deploy_challenger_endpoint(
        self,
        experiment_id: str,
        model_artifact_s3_path: str,
        instance_type: str = 'ml.m5.large',
    ) -> str:
        """
        Deploy a challenger model to a separate SageMaker endpoint.

        Creates a SageMaker model, endpoint configuration, and endpoint for
        the challenger model. Waits for the endpoint to be in service before
        returning.

        Args:
            experiment_id: Unique identifier for the experiment.
            model_artifact_s3_path: S3 path to the model artifact (model.tar.gz).
            instance_type: SageMaker instance type for the endpoint.
                Defaults to 'ml.m5.large'.

        Returns:
            The name of the deployed challenger endpoint.

        Raises:
            botocore.exceptions.ClientError: If SageMaker API calls fail.
            botocore.exceptions.WaiterError: If the endpoint fails to reach
                InService status.

        Example:
            manager = ABTestingManager()
            endpoint = manager.deploy_challenger_endpoint(
                experiment_id='exp-20240115-001',
                model_artifact_s3_path='s3://fraud-detection-models/model.tar.gz'
            )
        """
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f"fraud-detection-challenger-{experiment_id}-{timestamp}"
        model_name = f"fraud-detection-model-{experiment_id}"
        endpoint_config_name = f"fraud-detection-config-{experiment_id}-{timestamp}"

        # Create model
        self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': XGBOOST_IMAGE_URI,
                'ModelDataUrl': model_artifact_s3_path,
            },
            ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE,
        )

        # Create endpoint configuration
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type,
            }],
        )

        # Create endpoint
        self.sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

        print(f"Deploying challenger endpoint: {endpoint_name}")
        print("Waiting for endpoint to be in service...")

        # Wait for endpoint to be in service
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)

        print(f"✓ Challenger endpoint deployed: {endpoint_name}")

        return endpoint_name

    def generate_traffic_split_config(
        self,
        champion_endpoint: str,
        challenger_endpoint: str,
        challenger_traffic_pct: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a traffic splitting configuration template for gradual rollout.

        Creates a configuration dictionary with endpoint details, a staged
        rollout plan (1%, 10%, 50%, 100%), and success criteria for the
        A/B test.

        Args:
            champion_endpoint: Name of the production champion endpoint.
            challenger_endpoint: Name of the challenger endpoint.
            challenger_traffic_pct: Initial percentage of traffic to route
                to the challenger. Defaults to 10.

        Returns:
            A dictionary containing endpoint configuration, rollout plan
            stages, and success criteria.

        Example:
            manager = ABTestingManager()
            config = manager.generate_traffic_split_config(
                champion_endpoint='fraud-detection-production',
                challenger_endpoint='fraud-detection-challenger-exp-001-20240115',
                challenger_traffic_pct=10
            )
        """
        config: Dict[str, Any] = {
            'endpoints': {
                'champion': {
                    'name': champion_endpoint,
                    'traffic_percentage': 100 - challenger_traffic_pct,
                },
                'challenger': {
                    'name': challenger_endpoint,
                    'traffic_percentage': challenger_traffic_pct,
                },
            },
            'rollout_plan': list(ROLLOUT_STAGES),
            'success_criteria': dict(SUCCESS_CRITERIA),
        }

        return config

    def compare_endpoints(
        self,
        champion_endpoint: str,
        challenger_endpoint: str,
        test_data: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[float], List[float]]:
        """
        Compare champion and challenger endpoint performance on test data.

        Invokes both endpoints with each test record, measures latency,
        and returns comparison metrics including average, p95, and p99
        latencies along with predictions from each endpoint.

        Args:
            champion_endpoint: Name of the production champion endpoint.
            challenger_endpoint: Name of the challenger endpoint.
            test_data: List of test records to send to both endpoints.

        Returns:
            A tuple of (comparison_dict, champion_predictions, challenger_predictions)
            where comparison_dict contains latency metrics for each endpoint.

        Raises:
            botocore.exceptions.ClientError: If endpoint invocation fails.

        Example:
            manager = ABTestingManager()
            comparison, champ_preds, chal_preds = manager.compare_endpoints(
                champion_endpoint='fraud-detection-production',
                challenger_endpoint='fraud-detection-challenger-exp-001',
                test_data=[{'feature1': 0.5, 'feature2': 1.2}]
            )
        """
        runtime_client = boto3.client('sagemaker-runtime')

        champion_predictions: List[float] = []
        challenger_predictions: List[float] = []
        champion_latencies: List[float] = []
        challenger_latencies: List[float] = []

        for record in test_data:
            # Champion prediction
            start = time.time()
            champion_response = runtime_client.invoke_endpoint(
                EndpointName=champion_endpoint,
                Body=json.dumps(record),
                ContentType='application/json',
            )
            champion_latencies.append((time.time() - start) * 1000)
            champion_predictions.append(
                float(champion_response['Body'].read())
            )

            # Challenger prediction
            start = time.time()
            challenger_response = runtime_client.invoke_endpoint(
                EndpointName=challenger_endpoint,
                Body=json.dumps(record),
                ContentType='application/json',
            )
            challenger_latencies.append((time.time() - start) * 1000)
            challenger_predictions.append(
                float(challenger_response['Body'].read())
            )

        # Calculate latency metrics
        comparison: Dict[str, Any] = {
            'champion': {
                'avg_latency_ms': float(np.mean(champion_latencies)),
                'p95_latency_ms': float(np.percentile(champion_latencies, 95)),
                'p99_latency_ms': float(np.percentile(champion_latencies, 99)),
            },
            'challenger': {
                'avg_latency_ms': float(np.mean(challenger_latencies)),
                'p95_latency_ms': float(np.percentile(challenger_latencies, 95)),
                'p99_latency_ms': float(np.percentile(challenger_latencies, 99)),
            },
        }

        return comparison, champion_predictions, challenger_predictions

    def promote_challenger_to_champion(
        self,
        challenger_endpoint: str,
    ) -> None:
        """
        Promote a challenger endpoint to the production champion.

        Updates the production endpoint to use the challenger's endpoint
        configuration, waits for the update to complete, and cleans up
        the old challenger endpoint.

        Args:
            challenger_endpoint: Name of the challenger endpoint to promote.

        Raises:
            botocore.exceptions.ClientError: If SageMaker API calls fail.
            botocore.exceptions.WaiterError: If the production endpoint fails
                to reach InService status after update.

        Example:
            manager = ABTestingManager()
            manager.promote_challenger_to_champion(
                challenger_endpoint='fraud-detection-challenger-exp-001-20240115'
            )
        """
        # Get challenger endpoint config
        challenger_desc = self.sagemaker_client.describe_endpoint(
            EndpointName=challenger_endpoint,
        )
        challenger_config = challenger_desc['EndpointConfigName']

        # Update production endpoint to use challenger config
        self.sagemaker_client.update_endpoint(
            EndpointName=PRODUCTION_ENDPOINT,
            EndpointConfigName=challenger_config,
        )

        print(f"Promoting challenger {challenger_endpoint} to production...")
        print("Waiting for endpoint update...")

        # Wait for production endpoint to be in service
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=PRODUCTION_ENDPOINT)

        print("✓ Challenger promoted to production!")

        # Clean up old challenger endpoint
        self.sagemaker_client.delete_endpoint(EndpointName=challenger_endpoint)
        print(f"Cleaned up challenger endpoint: {challenger_endpoint}")
