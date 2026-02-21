"""
Unit tests for production integration module.

Tests hyperparameter validation, Parameter Store writes, backup creation,
and promotion event logging with mocked AWS clients.
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

from production_integration import ProductionIntegrator, PARAM_PATHS, REQUIRED_PARAMS


class TestProductionIntegrator:
    """Test suite for ProductionIntegrator class."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Create mock boto3 clients for SSM, S3, and Step Functions."""
        with patch('production_integration.boto3.client') as mock_client:
            mock_ssm = MagicMock()
            mock_s3 = MagicMock()
            mock_sfn = MagicMock()

            # Set up ParameterNotFound exception on the SSM mock
            mock_ssm.exceptions.ParameterNotFound = type(
                'ParameterNotFound', (Exception,), {}
            )

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

    @pytest.fixture
    def integrator(self, mock_boto3_clients):
        """Create ProductionIntegrator instance with mocked dependencies."""
        return ProductionIntegrator()

    @pytest.fixture
    def integrator_with_tracker(self, mock_boto3_clients):
        """Create ProductionIntegrator with a mock ExperimentTracker."""
        mock_tracker = MagicMock()
        mock_tracker.start_experiment.return_value = 'promotion-exp-001'
        return ProductionIntegrator(experiment_tracker=mock_tracker)

    @pytest.fixture
    def valid_hyperparameters(self):
        """Valid hyperparameters fixture."""
        return {
            'objective': 'binary:logistic',
            'num_round': 150,
            'max_depth': 7,
            'eta': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

    # --- Initialization tests ---

    def test_init_creates_clients(self, mock_boto3_clients):
        """Test that __init__ creates SSM, S3, and Step Functions clients."""
        integrator = ProductionIntegrator()
        assert integrator.ssm_client is not None
        assert integrator.s3_client is not None
        assert integrator.sfn_client is not None

    def test_init_stores_experiment_tracker(self, mock_boto3_clients):
        """Test that experiment_tracker is stored when provided."""
        mock_tracker = MagicMock()
        integrator = ProductionIntegrator(experiment_tracker=mock_tracker)
        assert integrator.experiment_tracker is mock_tracker

    def test_init_defaults_experiment_tracker_to_none(self, mock_boto3_clients):
        """Test that experiment_tracker defaults to None."""
        integrator = ProductionIntegrator()
        assert integrator.experiment_tracker is None

    # --- Validation tests ---

    def test_validate_valid_hyperparameters(self, integrator, valid_hyperparameters):
        """Test valid hyperparameters return True."""
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    def test_validate_missing_required_parameter(self, integrator):
        """Test missing required parameter raises ValueError with descriptive message."""
        incomplete = {
            'objective': 'binary:logistic',
            'num_round': 150,
            # missing max_depth, eta, subsample, colsample_bytree
        }
        with pytest.raises(ValueError, match="Missing required hyperparameters"):
            integrator.validate_hyperparameters(incomplete)

    def test_validate_max_depth_too_low(self, integrator, valid_hyperparameters):
        """Test max_depth=0 raises ValueError."""
        valid_hyperparameters['max_depth'] = 0
        with pytest.raises(ValueError, match="max_depth"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_max_depth_too_high(self, integrator, valid_hyperparameters):
        """Test max_depth=21 raises ValueError."""
        valid_hyperparameters['max_depth'] = 21
        with pytest.raises(ValueError, match="max_depth"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_eta_zero(self, integrator, valid_hyperparameters):
        """Test eta=0.0 raises ValueError (exclusive lower bound)."""
        valid_hyperparameters['eta'] = 0.0
        with pytest.raises(ValueError, match="eta"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_eta_too_high(self, integrator, valid_hyperparameters):
        """Test eta=1.1 raises ValueError."""
        valid_hyperparameters['eta'] = 1.1
        with pytest.raises(ValueError, match="eta"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_num_round_too_low(self, integrator, valid_hyperparameters):
        """Test num_round=0 raises ValueError."""
        valid_hyperparameters['num_round'] = 0
        with pytest.raises(ValueError, match="num_round"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_num_round_too_high(self, integrator, valid_hyperparameters):
        """Test num_round=1001 raises ValueError."""
        valid_hyperparameters['num_round'] = 1001
        with pytest.raises(ValueError, match="num_round"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_subsample_zero(self, integrator, valid_hyperparameters):
        """Test subsample=0.0 raises ValueError (exclusive lower bound)."""
        valid_hyperparameters['subsample'] = 0.0
        with pytest.raises(ValueError, match="subsample"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_subsample_too_high(self, integrator, valid_hyperparameters):
        """Test subsample=1.1 raises ValueError."""
        valid_hyperparameters['subsample'] = 1.1
        with pytest.raises(ValueError, match="subsample"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_colsample_bytree_zero(self, integrator, valid_hyperparameters):
        """Test colsample_bytree=0.0 raises ValueError (exclusive lower bound)."""
        valid_hyperparameters['colsample_bytree'] = 0.0
        with pytest.raises(ValueError, match="colsample_bytree"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_colsample_bytree_too_high(self, integrator, valid_hyperparameters):
        """Test colsample_bytree=1.1 raises ValueError."""
        valid_hyperparameters['colsample_bytree'] = 1.1
        with pytest.raises(ValueError, match="colsample_bytree"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_invalid_type_for_max_depth(self, integrator, valid_hyperparameters):
        """Test non-numeric string for max_depth raises ValueError."""
        valid_hyperparameters['max_depth'] = 'not_a_number'
        with pytest.raises(ValueError, match="max_depth"):
            integrator.validate_hyperparameters(valid_hyperparameters)

    def test_validate_boundary_max_depth_min(self, integrator, valid_hyperparameters):
        """Test max_depth=1 is valid (lower boundary)."""
        valid_hyperparameters['max_depth'] = 1
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    def test_validate_boundary_max_depth_max(self, integrator, valid_hyperparameters):
        """Test max_depth=20 is valid (upper boundary)."""
        valid_hyperparameters['max_depth'] = 20
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    def test_validate_boundary_eta_max(self, integrator, valid_hyperparameters):
        """Test eta=1.0 is valid (upper boundary, inclusive)."""
        valid_hyperparameters['eta'] = 1.0
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    def test_validate_boundary_num_round_min(self, integrator, valid_hyperparameters):
        """Test num_round=1 is valid (lower boundary)."""
        valid_hyperparameters['num_round'] = 1
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    def test_validate_boundary_num_round_max(self, integrator, valid_hyperparameters):
        """Test num_round=1000 is valid (upper boundary)."""
        valid_hyperparameters['num_round'] = 1000
        assert integrator.validate_hyperparameters(valid_hyperparameters) is True

    # --- Backup tests ---

    def test_backup_reads_all_parameter_paths(self, integrator, mock_boto3_clients):
        """Test backup reads all parameter paths from SSM."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'some_value'}
        }

        backup, backup_key = integrator.backup_current_parameters()

        # Should read all 6 parameter paths
        assert mock_ssm.get_parameter.call_count == len(PARAM_PATHS)
        for param_path in PARAM_PATHS.values():
            mock_ssm.get_parameter.assert_any_call(Name=param_path)

        # All values should be in the backup
        for param_path in PARAM_PATHS.values():
            assert param_path in backup
            assert backup[param_path] == 'some_value'

    def test_backup_handles_parameter_not_found(self, integrator, mock_boto3_clients):
        """Test backup handles ParameterNotFound by setting None."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.side_effect = mock_ssm.exceptions.ParameterNotFound(
            {'Error': {'Code': 'ParameterNotFound'}}, 'GetParameter'
        )

        backup, backup_key = integrator.backup_current_parameters()

        # All values should be None
        for param_path in PARAM_PATHS.values():
            assert backup[param_path] is None

    def test_backup_saves_yaml_to_s3(self, integrator, mock_boto3_clients):
        """Test backup saves YAML to S3 with correct key format."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_s3 = mock_boto3_clients['s3']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': '7'}
        }

        backup, backup_key = integrator.backup_current_parameters()

        # Verify S3 put_object was called
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'fraud-detection-config'
        assert call_kwargs['Key'].startswith('parameter-store-backups/backup-')
        assert call_kwargs['Key'].endswith('.yaml')

    def test_backup_returns_correct_dict_and_key(self, integrator, mock_boto3_clients):
        """Test backup returns correct dict and key."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'test_val'}
        }

        backup, backup_key = integrator.backup_current_parameters()

        assert isinstance(backup, dict)
        assert isinstance(backup_key, str)
        assert backup_key.startswith('parameter-store-backups/backup-')

    # --- Write tests ---

    def test_write_all_parameters_to_correct_paths(
        self, integrator, mock_boto3_clients, valid_hyperparameters
    ):
        """Test writes all 6 parameters to correct paths."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'old_value'}
        }

        integrator.write_hyperparameters_to_parameter_store(valid_hyperparameters)

        # Verify put_parameter was called for each parameter
        assert mock_ssm.put_parameter.call_count == len(PARAM_PATHS)
        for param_name, param_path in PARAM_PATHS.items():
            mock_ssm.put_parameter.assert_any_call(
                Name=param_path,
                Value=str(valid_hyperparameters[param_name]),
                Type='String',
                Overwrite=True,
            )

    def test_write_creates_backup_before_writing(
        self, integrator, mock_boto3_clients, valid_hyperparameters
    ):
        """Test creates backup before writing."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_s3 = mock_boto3_clients['s3']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'old_value'}
        }

        integrator.write_hyperparameters_to_parameter_store(valid_hyperparameters)

        # S3 put_object (backup) should have been called before SSM put_parameter
        mock_s3.put_object.assert_called_once()

    def test_write_validates_before_writing(
        self, integrator, mock_boto3_clients
    ):
        """Test validates before writing (invalid params should not write)."""
        mock_ssm = mock_boto3_clients['ssm']
        invalid_params = {
            'objective': 'binary:logistic',
            'num_round': 150,
            'max_depth': 999,  # out of range
            'eta': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        with pytest.raises(ValueError):
            integrator.write_hyperparameters_to_parameter_store(invalid_params)

        # put_parameter should NOT have been called
        mock_ssm.put_parameter.assert_not_called()

    def test_write_logs_promotion_event_with_tracker(
        self, integrator_with_tracker, mock_boto3_clients, valid_hyperparameters
    ):
        """Test logs promotion event when experiment_tracker is provided."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'old_value'}
        }

        integrator_with_tracker.write_hyperparameters_to_parameter_store(
            valid_hyperparameters, experiment_id='exp-001'
        )

        tracker = integrator_with_tracker.experiment_tracker
        tracker.start_experiment.assert_called_once()
        tracker.log_parameters.assert_called_once()
        tracker.close_experiment.assert_called_once()

    def test_write_does_not_log_without_tracker(
        self, integrator, mock_boto3_clients, valid_hyperparameters
    ):
        """Test does not log when experiment_tracker is None."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'old_value'}
        }

        # Should not raise even without tracker
        backup_key = integrator.write_hyperparameters_to_parameter_store(
            valid_hyperparameters
        )
        assert integrator.experiment_tracker is None
        assert isinstance(backup_key, str)

    def test_write_returns_backup_key(
        self, integrator, mock_boto3_clients, valid_hyperparameters
    ):
        """Test returns backup key."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_ssm.get_parameter.return_value = {
            'Parameter': {'Value': 'old_value'}
        }

        backup_key = integrator.write_hyperparameters_to_parameter_store(
            valid_hyperparameters
        )

        assert isinstance(backup_key, str)
        assert backup_key.startswith('parameter-store-backups/backup-')

    # --- Promotion event logging tests ---

    def test_log_promotion_starts_experiment(
        self, integrator_with_tracker, valid_hyperparameters
    ):
        """Test starts experiment with correct name."""
        tracker = integrator_with_tracker.experiment_tracker

        integrator_with_tracker._log_promotion_event(
            hyperparameters=valid_hyperparameters,
            backup_key='parameter-store-backups/backup-20240115-120000.yaml',
        )

        tracker.start_experiment.assert_called_once_with(
            experiment_name='production-promotion',
            algorithm='promotion',
        )

    def test_log_promotion_logs_all_parameters(
        self, integrator_with_tracker, valid_hyperparameters
    ):
        """Test logs all promotion parameters."""
        tracker = integrator_with_tracker.experiment_tracker

        integrator_with_tracker._log_promotion_event(
            hyperparameters=valid_hyperparameters,
            backup_key='parameter-store-backups/backup-20240115-120000.yaml',
        )

        tracker.log_parameters.assert_called_once()
        logged_params = tracker.log_parameters.call_args[0][1]

        assert logged_params['action'] == 'promote_hyperparameters'
        assert logged_params['backup_key'] == 'parameter-store-backups/backup-20240115-120000.yaml'
        assert 'timestamp' in logged_params
        # Promoted hyperparameters should be included
        for param_name, param_value in valid_hyperparameters.items():
            assert logged_params[f'promoted_{param_name}'] == param_value

    def test_log_promotion_closes_experiment(
        self, integrator_with_tracker, valid_hyperparameters
    ):
        """Test closes experiment after logging."""
        tracker = integrator_with_tracker.experiment_tracker

        integrator_with_tracker._log_promotion_event(
            hyperparameters=valid_hyperparameters,
            backup_key='parameter-store-backups/backup-20240115-120000.yaml',
        )

        tracker.close_experiment.assert_called_once_with('promotion-exp-001')

    def test_log_promotion_includes_source_experiment_id(
        self, integrator_with_tracker, valid_hyperparameters
    ):
        """Test includes source_experiment_id when provided."""
        tracker = integrator_with_tracker.experiment_tracker

        integrator_with_tracker._log_promotion_event(
            hyperparameters=valid_hyperparameters,
            backup_key='parameter-store-backups/backup-20240115-120000.yaml',
            experiment_id='exp-source-001',
        )

        logged_params = tracker.log_parameters.call_args[0][1]
        assert logged_params['source_experiment_id'] == 'exp-source-001'

    def test_log_promotion_excludes_source_experiment_id_when_none(
        self, integrator_with_tracker, valid_hyperparameters
    ):
        """Test source_experiment_id is not in params when not provided."""
        tracker = integrator_with_tracker.experiment_tracker

        integrator_with_tracker._log_promotion_event(
            hyperparameters=valid_hyperparameters,
            backup_key='parameter-store-backups/backup-20240115-120000.yaml',
            experiment_id=None,
        )

        logged_params = tracker.log_parameters.call_args[0][1]
        assert 'source_experiment_id' not in logged_params
