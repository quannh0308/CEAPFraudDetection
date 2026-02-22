"""
Unit tests for A/B testing module.

Tests challenger deployment, traffic split configuration, endpoint comparison,
and challenger promotion with mocked SageMaker clients.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock sagemaker modules before importing
sys.modules['sagemaker'] = MagicMock()
sys.modules['sagemaker.experiments'] = MagicMock()

from ab_testing import (
    ABTestingManager,
    PRODUCTION_ENDPOINT,
    ROLLOUT_STAGES,
    SUCCESS_CRITERIA,
    XGBOOST_IMAGE_URI,
    SAGEMAKER_EXECUTION_ROLE,
)


class TestABTestingManager:
    """Test suite for ABTestingManager class."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Create mock boto3 clients for SageMaker and S3."""
        with patch('ab_testing.boto3.client') as mock_client:
            mock_sagemaker = MagicMock()
            mock_s3 = MagicMock()
            mock_runtime = MagicMock()

            # Set up waiter as a no-op mock
            mock_sagemaker.get_waiter.return_value = MagicMock()

            def client_factory(service_name, **kwargs):
                if service_name == 'sagemaker':
                    return mock_sagemaker
                elif service_name == 's3':
                    return mock_s3
                elif service_name == 'sagemaker-runtime':
                    return mock_runtime
                return MagicMock()

            mock_client.side_effect = client_factory
            yield {
                'sagemaker': mock_sagemaker,
                's3': mock_s3,
                'runtime': mock_runtime,
            }

    @pytest.fixture
    def manager(self, mock_boto3_clients):
        """Create ABTestingManager instance with mocked dependencies."""
        return ABTestingManager()

    # --- Initialization tests ---

    def test_init_creates_sagemaker_client(self, mock_boto3_clients):
        """Test that __init__ creates a SageMaker client."""
        manager = ABTestingManager()
        assert manager.sagemaker_client is not None

    def test_init_creates_s3_client(self, mock_boto3_clients):
        """Test that __init__ creates an S3 client."""
        manager = ABTestingManager()
        assert manager.s3_client is not None

    # --- Deploy challenger endpoint tests ---

    def test_deploy_creates_model_with_correct_params(self, manager, mock_boto3_clients):
        """Test deploy_challenger_endpoint creates model with correct image and role."""
        mock_sm = mock_boto3_clients['sagemaker']

        manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        mock_sm.create_model.assert_called_once()
        call_kwargs = mock_sm.create_model.call_args[1]
        assert call_kwargs['PrimaryContainer']['Image'] == XGBOOST_IMAGE_URI
        assert call_kwargs['PrimaryContainer']['ModelDataUrl'] == 's3://bucket/model.tar.gz'
        assert call_kwargs['ExecutionRoleArn'] == SAGEMAKER_EXECUTION_ROLE
        assert 'exp-001' in call_kwargs['ModelName']

    def test_deploy_creates_endpoint_config(self, manager, mock_boto3_clients):
        """Test deploy_challenger_endpoint creates an endpoint configuration."""
        mock_sm = mock_boto3_clients['sagemaker']

        manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        mock_sm.create_endpoint_config.assert_called_once()
        call_kwargs = mock_sm.create_endpoint_config.call_args[1]
        variants = call_kwargs['ProductionVariants']
        assert len(variants) == 1
        assert variants[0]['VariantName'] == 'AllTraffic'
        assert variants[0]['InitialInstanceCount'] == 1

    def test_deploy_creates_endpoint(self, manager, mock_boto3_clients):
        """Test deploy_challenger_endpoint creates an endpoint."""
        mock_sm = mock_boto3_clients['sagemaker']

        manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        mock_sm.create_endpoint.assert_called_once()
        call_kwargs = mock_sm.create_endpoint.call_args[1]
        assert 'EndpointName' in call_kwargs
        assert 'EndpointConfigName' in call_kwargs

    def test_deploy_waits_for_endpoint_in_service(self, manager, mock_boto3_clients):
        """Test deploy_challenger_endpoint waits for endpoint to be in service."""
        mock_sm = mock_boto3_clients['sagemaker']

        manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        mock_sm.get_waiter.assert_called_once_with('endpoint_in_service')
        waiter = mock_sm.get_waiter.return_value
        waiter.wait.assert_called_once()

    def test_deploy_returns_endpoint_name_with_experiment_id(self, manager, mock_boto3_clients):
        """Test deploy returns endpoint name containing the experiment ID."""
        endpoint_name = manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        assert 'exp-001' in endpoint_name
        assert endpoint_name.startswith('fraud-detection-challenger-')

    def test_deploy_returns_endpoint_name_with_timestamp(self, manager, mock_boto3_clients):
        """Test deploy returns endpoint name containing a timestamp."""
        endpoint_name = manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
        )

        # Endpoint name format: fraud-detection-challenger-{exp_id}-{YYYYMMDD-HHMMSS}
        parts = endpoint_name.split('-')
        # The timestamp portion should be at the end (YYYYMMDD-HHMMSS = 15 chars)
        timestamp_part = endpoint_name.split('exp-001-')[1]
        assert len(timestamp_part) == 15  # YYYYMMDD-HHMMSS

    def test_deploy_uses_custom_instance_type(self, manager, mock_boto3_clients):
        """Test deploy uses custom instance type when provided."""
        mock_sm = mock_boto3_clients['sagemaker']

        manager.deploy_challenger_endpoint(
            experiment_id='exp-001',
            model_artifact_s3_path='s3://bucket/model.tar.gz',
            instance_type='ml.c5.xlarge',
        )

        call_kwargs = mock_sm.create_endpoint_config.call_args[1]
        assert call_kwargs['ProductionVariants'][0]['InstanceType'] == 'ml.c5.xlarge'

    # --- Traffic split config tests ---

    def test_traffic_split_includes_champion_endpoint(self, manager):
        """Test traffic split config includes champion endpoint name."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        assert config['endpoints']['champion']['name'] == 'champion-ep'

    def test_traffic_split_includes_challenger_endpoint(self, manager):
        """Test traffic split config includes challenger endpoint name."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        assert config['endpoints']['challenger']['name'] == 'challenger-ep'

    def test_traffic_split_default_challenger_traffic_10_percent(self, manager):
        """Test default challenger traffic is 10%."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        assert config['endpoints']['challenger']['traffic_percentage'] == 10
        assert config['endpoints']['champion']['traffic_percentage'] == 90

    def test_traffic_split_custom_challenger_traffic(self, manager):
        """Test custom challenger traffic percentage."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            challenger_traffic_pct=25,
        )

        assert config['endpoints']['challenger']['traffic_percentage'] == 25
        assert config['endpoints']['champion']['traffic_percentage'] == 75

    def test_traffic_split_includes_rollout_plan_with_4_stages(self, manager):
        """Test rollout plan has exactly 4 stages."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        assert len(config['rollout_plan']) == 4

    def test_traffic_split_rollout_stages_are_1_10_50_100(self, manager):
        """Test rollout stages have challenger traffic of 1%, 10%, 50%, 100%."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        traffic_pcts = [s['challenger_traffic'] for s in config['rollout_plan']]
        assert traffic_pcts == [1, 10, 50, 100]

    def test_traffic_split_includes_success_criteria(self, manager):
        """Test traffic split config includes success criteria."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        assert config['success_criteria'] == SUCCESS_CRITERIA
        assert config['success_criteria']['min_accuracy'] == 0.90
        assert config['success_criteria']['max_latency_ms'] == 100
        assert config['success_criteria']['min_improvement_pct'] == 1.0

    # --- Compare endpoints tests ---

    def _make_invoke_response(self, value=b'0.95'):
        """Helper to create a mock invoke_endpoint response."""
        mock_body = MagicMock()
        mock_body.read.return_value = value
        return {'Body': mock_body}

    @patch('ab_testing.time.time')
    def test_compare_invokes_both_endpoints(self, mock_time, manager, mock_boto3_clients):
        """Test compare_endpoints invokes both champion and challenger."""
        mock_runtime = mock_boto3_clients['runtime']
        mock_time.side_effect = [0.0, 0.01, 0.0, 0.01]  # start/end pairs
        mock_runtime.invoke_endpoint.return_value = self._make_invoke_response()

        manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'feature1': 0.5}],
        )

        assert mock_runtime.invoke_endpoint.call_count == 2
        call_args_list = mock_runtime.invoke_endpoint.call_args_list
        assert call_args_list[0][1]['EndpointName'] == 'champion-ep'
        assert call_args_list[1][1]['EndpointName'] == 'challenger-ep'

    @patch('ab_testing.time.time')
    def test_compare_returns_champion_latency_metrics(self, mock_time, manager, mock_boto3_clients):
        """Test compare returns champion latency metrics (avg, p95, p99)."""
        mock_runtime = mock_boto3_clients['runtime']
        # Two records: champion start/end, challenger start/end x2
        mock_time.side_effect = [0.0, 0.010, 0.0, 0.020, 0.0, 0.015, 0.0, 0.025]
        mock_runtime.invoke_endpoint.return_value = self._make_invoke_response()

        comparison, _, _ = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'f': 1}, {'f': 2}],
        )

        assert 'avg_latency_ms' in comparison['champion']
        assert 'p95_latency_ms' in comparison['champion']
        assert 'p99_latency_ms' in comparison['champion']

    @patch('ab_testing.time.time')
    def test_compare_returns_challenger_latency_metrics(self, mock_time, manager, mock_boto3_clients):
        """Test compare returns challenger latency metrics (avg, p95, p99)."""
        mock_runtime = mock_boto3_clients['runtime']
        mock_time.side_effect = [0.0, 0.010, 0.0, 0.020, 0.0, 0.015, 0.0, 0.025]
        mock_runtime.invoke_endpoint.return_value = self._make_invoke_response()

        comparison, _, _ = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'f': 1}, {'f': 2}],
        )

        assert 'avg_latency_ms' in comparison['challenger']
        assert 'p95_latency_ms' in comparison['challenger']
        assert 'p99_latency_ms' in comparison['challenger']

    @patch('ab_testing.time.time')
    def test_compare_returns_champion_predictions(self, mock_time, manager, mock_boto3_clients):
        """Test compare returns champion predictions as floats."""
        mock_runtime = mock_boto3_clients['runtime']
        mock_time.side_effect = [0.0, 0.01, 0.0, 0.01]

        champion_resp = self._make_invoke_response(b'0.85')
        challenger_resp = self._make_invoke_response(b'0.92')
        mock_runtime.invoke_endpoint.side_effect = [champion_resp, challenger_resp]

        _, champion_preds, _ = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'feature1': 0.5}],
        )

        assert champion_preds == [0.85]

    @patch('ab_testing.time.time')
    def test_compare_returns_challenger_predictions(self, mock_time, manager, mock_boto3_clients):
        """Test compare returns challenger predictions as floats."""
        mock_runtime = mock_boto3_clients['runtime']
        mock_time.side_effect = [0.0, 0.01, 0.0, 0.01]

        champion_resp = self._make_invoke_response(b'0.85')
        challenger_resp = self._make_invoke_response(b'0.92')
        mock_runtime.invoke_endpoint.side_effect = [champion_resp, challenger_resp]

        _, _, challenger_preds = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'feature1': 0.5}],
        )

        assert challenger_preds == [0.92]

    @patch('ab_testing.time.time')
    def test_compare_handles_multiple_records(self, mock_time, manager, mock_boto3_clients):
        """Test compare handles multiple test records correctly."""
        mock_runtime = mock_boto3_clients['runtime']
        # 3 records x 2 endpoints x 2 time calls = 12 time calls
        mock_time.side_effect = [
            0.0, 0.01, 0.0, 0.02,  # record 1
            0.0, 0.015, 0.0, 0.025,  # record 2
            0.0, 0.012, 0.0, 0.022,  # record 3
        ]

        responses = [
            self._make_invoke_response(b'0.80'),  # champion rec 1
            self._make_invoke_response(b'0.90'),  # challenger rec 1
            self._make_invoke_response(b'0.82'),  # champion rec 2
            self._make_invoke_response(b'0.91'),  # challenger rec 2
            self._make_invoke_response(b'0.81'),  # champion rec 3
            self._make_invoke_response(b'0.93'),  # challenger rec 3
        ]
        mock_runtime.invoke_endpoint.side_effect = responses

        comparison, champion_preds, challenger_preds = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'f': 1}, {'f': 2}, {'f': 3}],
        )

        assert len(champion_preds) == 3
        assert len(challenger_preds) == 3
        assert champion_preds == [0.80, 0.82, 0.81]
        assert challenger_preds == [0.90, 0.91, 0.93]
        assert mock_runtime.invoke_endpoint.call_count == 6

    # --- Promote challenger tests ---

    def test_promote_describes_challenger_endpoint(self, manager, mock_boto3_clients):
        """Test promote describes the challenger endpoint to get its config."""
        mock_sm = mock_boto3_clients['sagemaker']
        mock_sm.describe_endpoint.return_value = {
            'EndpointConfigName': 'challenger-config-001',
        }

        manager.promote_challenger_to_champion(
            challenger_endpoint='challenger-ep-001',
        )

        mock_sm.describe_endpoint.assert_called_once_with(
            EndpointName='challenger-ep-001',
        )

    def test_promote_updates_production_endpoint(self, manager, mock_boto3_clients):
        """Test promote updates the production endpoint with challenger config."""
        mock_sm = mock_boto3_clients['sagemaker']
        mock_sm.describe_endpoint.return_value = {
            'EndpointConfigName': 'challenger-config-001',
        }

        manager.promote_challenger_to_champion(
            challenger_endpoint='challenger-ep-001',
        )

        mock_sm.update_endpoint.assert_called_once_with(
            EndpointName=PRODUCTION_ENDPOINT,
            EndpointConfigName='challenger-config-001',
        )

    def test_promote_waits_for_production_endpoint(self, manager, mock_boto3_clients):
        """Test promote waits for the production endpoint to be in service."""
        mock_sm = mock_boto3_clients['sagemaker']
        mock_sm.describe_endpoint.return_value = {
            'EndpointConfigName': 'challenger-config-001',
        }

        manager.promote_challenger_to_champion(
            challenger_endpoint='challenger-ep-001',
        )

        mock_sm.get_waiter.assert_called_with('endpoint_in_service')
        waiter = mock_sm.get_waiter.return_value
        waiter.wait.assert_called_with(EndpointName=PRODUCTION_ENDPOINT)

    def test_promote_deletes_challenger_endpoint(self, manager, mock_boto3_clients):
        """Test promote deletes the old challenger endpoint after promotion."""
        mock_sm = mock_boto3_clients['sagemaker']
        mock_sm.describe_endpoint.return_value = {
            'EndpointConfigName': 'challenger-config-001',
        }

        manager.promote_challenger_to_champion(
            challenger_endpoint='challenger-ep-001',
        )

        mock_sm.delete_endpoint.assert_called_once_with(
            EndpointName='challenger-ep-001',
        )
