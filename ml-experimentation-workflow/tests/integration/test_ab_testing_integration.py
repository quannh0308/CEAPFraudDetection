"""
Integration tests for the A/B testing workflow.

Validates the complete flow: deploy challenger → compare → promote,
using mocked SageMaker services.

Requirements validated: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

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
)


def _make_invoke_response(value=b'0.95'):
    """Helper to create a mock invoke_endpoint response."""
    mock_body = MagicMock()
    mock_body.read.return_value = value
    return {'Body': mock_body}


class TestABTestingIntegration:
    """Integration tests for the complete A/B testing workflow."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Mock boto3 clients for SageMaker, S3, and SageMaker Runtime."""
        with patch('ab_testing.boto3.client') as mock_client:
            mock_sagemaker = MagicMock()
            mock_s3 = MagicMock()
            mock_runtime = MagicMock()

            # Waiter that completes immediately
            mock_sagemaker.get_waiter.return_value = MagicMock()

            # describe_endpoint returns a config name for promotion
            mock_sagemaker.describe_endpoint.return_value = {
                'EndpointConfigName': 'challenger-config-001',
            }

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
        """ABTestingManager with mocked dependencies."""
        return ABTestingManager()

    # ------------------------------------------------------------------ #
    # 1. Complete A/B testing workflow
    # ------------------------------------------------------------------ #
    @patch('ab_testing.time.time')
    def test_complete_ab_testing_workflow(
        self, mock_time, manager, mock_boto3_clients
    ):
        """Full flow: deploy → generate config → compare → promote."""
        mock_sm = mock_boto3_clients['sagemaker']
        mock_runtime = mock_boto3_clients['runtime']

        # Deploy challenger
        challenger_ep = manager.deploy_challenger_endpoint(
            experiment_id='exp-integration-001',
            model_artifact_s3_path='s3://models/model.tar.gz',
        )
        assert challenger_ep.startswith('fraud-detection-challenger-')
        mock_sm.create_model.assert_called_once()
        mock_sm.create_endpoint.assert_called_once()

        # Generate traffic split config
        config = manager.generate_traffic_split_config(
            champion_endpoint=PRODUCTION_ENDPOINT,
            challenger_endpoint=challenger_ep,
        )
        assert config['endpoints']['champion']['name'] == PRODUCTION_ENDPOINT
        assert config['endpoints']['challenger']['name'] == challenger_ep
        assert len(config['rollout_plan']) == 4

        # Compare endpoints
        mock_time.side_effect = [0.0, 0.010, 0.0, 0.015]
        mock_runtime.invoke_endpoint.return_value = _make_invoke_response(b'0.92')

        comparison, champ_preds, chal_preds = manager.compare_endpoints(
            champion_endpoint=PRODUCTION_ENDPOINT,
            challenger_endpoint=challenger_ep,
            test_data=[{'feature1': 0.5}],
        )
        assert 'champion' in comparison
        assert 'challenger' in comparison
        assert len(champ_preds) == 1
        assert len(chal_preds) == 1

        # Promote challenger
        manager.promote_challenger_to_champion(challenger_endpoint=challenger_ep)
        mock_sm.update_endpoint.assert_called_once_with(
            EndpointName=PRODUCTION_ENDPOINT,
            EndpointConfigName='challenger-config-001',
        )
        mock_sm.delete_endpoint.assert_called_once_with(EndpointName=challenger_ep)

    # ------------------------------------------------------------------ #
    # 2. Deploy and compare endpoints
    # ------------------------------------------------------------------ #
    @patch('ab_testing.time.time')
    def test_deploy_and_compare_endpoints(
        self, mock_time, manager, mock_boto3_clients
    ):
        """Deploy challenger, then compare with champion."""
        mock_runtime = mock_boto3_clients['runtime']

        challenger_ep = manager.deploy_challenger_endpoint(
            experiment_id='exp-compare-001',
            model_artifact_s3_path='s3://models/model.tar.gz',
        )

        # Two records → 4 invoke calls, 8 time calls
        mock_time.side_effect = [
            0.0, 0.010, 0.0, 0.012,
            0.0, 0.011, 0.0, 0.013,
        ]
        responses = [
            _make_invoke_response(b'0.80'),
            _make_invoke_response(b'0.85'),
            _make_invoke_response(b'0.82'),
            _make_invoke_response(b'0.88'),
        ]
        mock_runtime.invoke_endpoint.side_effect = responses

        comparison, champ_preds, chal_preds = manager.compare_endpoints(
            champion_endpoint=PRODUCTION_ENDPOINT,
            challenger_endpoint=challenger_ep,
            test_data=[{'f': 1}, {'f': 2}],
        )

        assert len(champ_preds) == 2
        assert len(chal_preds) == 2
        assert champ_preds == [0.80, 0.82]
        assert chal_preds == [0.85, 0.88]
        assert mock_runtime.invoke_endpoint.call_count == 4

    # ------------------------------------------------------------------ #
    # 3. Traffic split config follows rollout stages
    # ------------------------------------------------------------------ #
    def test_traffic_split_config_follows_rollout_stages(self, manager):
        """Verify 4-stage rollout plan with correct traffic percentages."""
        config = manager.generate_traffic_split_config(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
        )

        plan = config['rollout_plan']
        assert len(plan) == 4
        traffic_pcts = [s['challenger_traffic'] for s in plan]
        assert traffic_pcts == [1, 10, 50, 100]

        # Verify success criteria are included
        assert config['success_criteria'] == SUCCESS_CRITERIA

    # ------------------------------------------------------------------ #
    # 4. Challenger promotion updates production
    # ------------------------------------------------------------------ #
    def test_challenger_promotion_updates_production(
        self, manager, mock_boto3_clients
    ):
        """Production endpoint gets updated and challenger cleaned up."""
        mock_sm = mock_boto3_clients['sagemaker']

        challenger_ep = manager.deploy_challenger_endpoint(
            experiment_id='exp-promote-001',
            model_artifact_s3_path='s3://models/model.tar.gz',
        )

        manager.promote_challenger_to_champion(challenger_endpoint=challenger_ep)

        # Verify describe → update → wait → delete sequence
        mock_sm.describe_endpoint.assert_called_once_with(
            EndpointName=challenger_ep,
        )
        mock_sm.update_endpoint.assert_called_once_with(
            EndpointName=PRODUCTION_ENDPOINT,
            EndpointConfigName='challenger-config-001',
        )
        mock_sm.get_waiter.return_value.wait.assert_called_with(
            EndpointName=PRODUCTION_ENDPOINT,
        )
        mock_sm.delete_endpoint.assert_called_once_with(
            EndpointName=challenger_ep,
        )

    # ------------------------------------------------------------------ #
    # 5. Challenger endpoint naming includes experiment ID (Req 11.2)
    # ------------------------------------------------------------------ #
    def test_challenger_endpoint_naming_includes_experiment_id(
        self, manager, mock_boto3_clients
    ):
        """Endpoint name follows fraud-detection-challenger-{exp_id}-{ts}."""
        endpoint_name = manager.deploy_challenger_endpoint(
            experiment_id='my-exp-42',
            model_artifact_s3_path='s3://models/model.tar.gz',
        )

        assert endpoint_name.startswith('fraud-detection-challenger-')
        assert 'my-exp-42' in endpoint_name
        # Timestamp portion after experiment id
        ts_part = endpoint_name.split('my-exp-42-')[1]
        assert len(ts_part) == 15  # YYYYMMDD-HHMMSS

    # ------------------------------------------------------------------ #
    # 6. Compare returns latency metrics for both endpoints
    # ------------------------------------------------------------------ #
    @patch('ab_testing.time.time')
    def test_compare_returns_latency_metrics_for_both_endpoints(
        self, mock_time, manager, mock_boto3_clients
    ):
        """Comparison dict has avg/p95/p99 latency for champion and challenger."""
        mock_runtime = mock_boto3_clients['runtime']

        # 3 records → 6 invocations, 12 time calls
        mock_time.side_effect = [
            0.0, 0.010, 0.0, 0.020,
            0.0, 0.015, 0.0, 0.025,
            0.0, 0.012, 0.0, 0.022,
        ]
        mock_runtime.invoke_endpoint.return_value = _make_invoke_response(b'0.90')

        comparison, _, _ = manager.compare_endpoints(
            champion_endpoint='champion-ep',
            challenger_endpoint='challenger-ep',
            test_data=[{'f': 1}, {'f': 2}, {'f': 3}],
        )

        for role in ('champion', 'challenger'):
            assert 'avg_latency_ms' in comparison[role]
            assert 'p95_latency_ms' in comparison[role]
            assert 'p99_latency_ms' in comparison[role]
            assert comparison[role]['avg_latency_ms'] > 0
