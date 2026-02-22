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

from production_integration import (
    ProductionIntegrator,
    PARAM_PATHS,
    REQUIRED_PARAMS,
    S3AccessError,
    ParameterStoreError,
    SageMakerTrainingError,
    handle_sagemaker_training_error,
)


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


class TestConfigurationFileManagement:
    """Test suite for configuration file generation, validation, and S3 writes."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Create mock boto3 clients for SSM, S3, and Step Functions."""
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
    def sample_hyperparameters(self):
        """Sample hyperparameters for config generation."""
        return {
            'objective': 'binary:logistic',
            'num_round': 150,
            'max_depth': 7,
            'eta': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

    @pytest.fixture
    def sample_metrics(self):
        """Sample performance metrics for config generation."""
        return {
            'accuracy': 0.961,
            'precision': 0.92,
            'recall': 0.88,
            'f1_score': 0.90,
            'auc_roc': 0.95,
        }

    @pytest.fixture
    def valid_config(self, integrator, sample_hyperparameters, sample_metrics):
        """Generate a valid production config."""
        return integrator.generate_production_config(
            experiment_id='exp-20240115-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='data-science-team',
        )

    # --- generate_production_config tests ---

    def test_generate_config_returns_dict_with_model_key(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config has top-level 'model' key."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        assert 'model' in config
        assert isinstance(config['model'], dict)

    def test_generate_config_includes_algorithm(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes algorithm field set to 'xgboost'."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        assert config['model']['algorithm'] == 'xgboost'

    def test_generate_config_includes_version(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes version matching experiment_id."""
        config = integrator.generate_production_config(
            experiment_id='exp-20240115-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        assert config['model']['version'] == 'exp-20240115-001'

    def test_generate_config_includes_hyperparameters(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes hyperparameters dict."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        assert config['model']['hyperparameters'] == sample_hyperparameters

    def test_generate_config_includes_performance(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes performance metrics."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        assert config['model']['performance'] == sample_metrics

    def test_generate_config_includes_tested_date(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes tested_date in YYYY-MM-DD format."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='approver-name',
        )
        tested_date = config['model']['tested_date']
        assert isinstance(tested_date, str)
        # Verify date format YYYY-MM-DD
        datetime.strptime(tested_date, '%Y-%m-%d')

    def test_generate_config_includes_approved_by(
        self, integrator, sample_hyperparameters, sample_metrics
    ):
        """Test generated config includes approved_by field."""
        config = integrator.generate_production_config(
            experiment_id='exp-001',
            hyperparameters=sample_hyperparameters,
            metrics=sample_metrics,
            approver='data-science-team',
        )
        assert config['model']['approved_by'] == 'data-science-team'

    # --- validate_config_schema tests ---

    def test_validate_valid_config(self, integrator, valid_config):
        """Test valid config passes validation."""
        assert integrator.validate_config_schema(valid_config) is True

    def test_validate_missing_model_key(self, integrator):
        """Test config without 'model' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required top-level 'model' key"):
            integrator.validate_config_schema({'not_model': {}})

    def test_validate_missing_algorithm(self, integrator, valid_config):
        """Test config missing 'algorithm' raises ValueError."""
        del valid_config['model']['algorithm']
        with pytest.raises(ValueError, match="algorithm"):
            integrator.validate_config_schema(valid_config)

    def test_validate_missing_hyperparameters(self, integrator, valid_config):
        """Test config missing 'hyperparameters' raises ValueError."""
        del valid_config['model']['hyperparameters']
        with pytest.raises(ValueError, match="hyperparameters"):
            integrator.validate_config_schema(valid_config)

    def test_validate_missing_performance(self, integrator, valid_config):
        """Test config missing 'performance' raises ValueError."""
        del valid_config['model']['performance']
        with pytest.raises(ValueError, match="performance"):
            integrator.validate_config_schema(valid_config)

    def test_validate_missing_tested_date(self, integrator, valid_config):
        """Test config missing 'tested_date' raises ValueError."""
        del valid_config['model']['tested_date']
        with pytest.raises(ValueError, match="tested_date"):
            integrator.validate_config_schema(valid_config)

    def test_validate_missing_approved_by(self, integrator, valid_config):
        """Test config missing 'approved_by' raises ValueError."""
        del valid_config['model']['approved_by']
        with pytest.raises(ValueError, match="approved_by"):
            integrator.validate_config_schema(valid_config)

    def test_validate_wrong_type_algorithm(self, integrator, valid_config):
        """Test config with non-string algorithm raises ValueError."""
        valid_config['model']['algorithm'] = 123
        with pytest.raises(ValueError, match="must be str"):
            integrator.validate_config_schema(valid_config)

    def test_validate_wrong_type_hyperparameters(self, integrator, valid_config):
        """Test config with non-dict hyperparameters raises ValueError."""
        valid_config['model']['hyperparameters'] = 'not_a_dict'
        with pytest.raises(ValueError, match="must be dict"):
            integrator.validate_config_schema(valid_config)

    def test_validate_wrong_type_performance(self, integrator, valid_config):
        """Test config with non-dict performance raises ValueError."""
        valid_config['model']['performance'] = [0.95]
        with pytest.raises(ValueError, match="must be dict"):
            integrator.validate_config_schema(valid_config)

    def test_validate_wrong_type_tested_date(self, integrator, valid_config):
        """Test config with non-string tested_date raises ValueError."""
        valid_config['model']['tested_date'] = 20240115
        with pytest.raises(ValueError, match="must be str"):
            integrator.validate_config_schema(valid_config)

    def test_validate_wrong_type_approved_by(self, integrator, valid_config):
        """Test config with non-string approved_by raises ValueError."""
        valid_config['model']['approved_by'] = ['team']
        with pytest.raises(ValueError, match="must be str"):
            integrator.validate_config_schema(valid_config)

    # --- write_config_to_s3 tests ---

    def test_write_config_writes_yaml_to_s3(
        self, integrator, mock_boto3_clients, valid_config
    ):
        """Test writes config as YAML to production-model-config.yaml."""
        mock_s3 = mock_boto3_clients['s3']
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )

        integrator.write_config_to_s3(valid_config)

        # Should write to production-model-config.yaml
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs['Bucket'] == 'fraud-detection-config'
        assert call_kwargs['Key'] == 'production-model-config.yaml'
        assert isinstance(call_kwargs['Body'], str)

    def test_write_config_archives_existing_config(
        self, integrator, mock_boto3_clients, valid_config
    ):
        """Test archives existing config before writing new one."""
        mock_s3 = mock_boto3_clients['s3']
        mock_body = MagicMock()
        mock_body.read.return_value = b'old config content'
        mock_s3.get_object.return_value = {'Body': mock_body}

        integrator.write_config_to_s3(valid_config)

        # Should have 2 put_object calls: archive + new config
        assert mock_s3.put_object.call_count == 2

        # First call should be the archive
        archive_call = mock_s3.put_object.call_args_list[0][1]
        assert archive_call['Bucket'] == 'fraud-detection-config'
        assert archive_call['Key'].startswith('archive/production-model-config-')
        assert archive_call['Key'].endswith('.yaml')
        assert archive_call['Body'] == b'old config content'

        # Second call should be the new config
        new_call = mock_s3.put_object.call_args_list[1][1]
        assert new_call['Key'] == 'production-model-config.yaml'

    def test_write_config_skips_archive_when_no_existing(
        self, integrator, mock_boto3_clients, valid_config
    ):
        """Test skips archive when no existing config in S3."""
        mock_s3 = mock_boto3_clients['s3']
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )

        integrator.write_config_to_s3(valid_config)

        # Only 1 put_object call (no archive)
        assert mock_s3.put_object.call_count == 1
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs['Key'] == 'production-model-config.yaml'

    def test_write_config_validates_schema_first(
        self, integrator, mock_boto3_clients
    ):
        """Test validates schema before writing (invalid config should not write)."""
        mock_s3 = mock_boto3_clients['s3']
        invalid_config = {'not_model': {}}

        with pytest.raises(ValueError):
            integrator.write_config_to_s3(invalid_config)

        # S3 should not have been called
        mock_s3.get_object.assert_not_called()
        mock_s3.put_object.assert_not_called()

    def test_write_config_yaml_is_parseable(
        self, integrator, mock_boto3_clients, valid_config
    ):
        """Test written YAML body can be parsed back to original config."""
        import yaml as yaml_mod

        mock_s3 = mock_boto3_clients['s3']
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )

        integrator.write_config_to_s3(valid_config)

        written_body = mock_s3.put_object.call_args[1]['Body']
        parsed = yaml_mod.safe_load(written_body)
        assert parsed['model']['algorithm'] == valid_config['model']['algorithm']
        assert parsed['model']['approved_by'] == valid_config['model']['approved_by']


class TestPipelineTrigger:
    """Test suite for pipeline trigger, status checking, and promotion workflow."""

    @pytest.fixture
    def mock_boto3_clients(self):
        """Create mock boto3 clients for SSM, S3, and Step Functions."""
        with patch('production_integration.boto3.client') as mock_client:
            mock_ssm = MagicMock()
            mock_s3 = MagicMock()
            mock_sfn = MagicMock()

            mock_ssm.exceptions.ParameterNotFound = type(
                'ParameterNotFound', (Exception,), {}
            )
            mock_sfn.exceptions.ClientError = type(
                'ClientError', (Exception,), {}
            )
            mock_s3.exceptions.NoSuchKey = type(
                'NoSuchKey', (Exception,), {}
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

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics fixture."""
        return {
            'accuracy': 0.96,
            'precision': 0.92,
            'recall': 0.88,
        }

    # --- Pipeline trigger tests ---

    def test_trigger_returns_execution_arn(self, integrator, mock_boto3_clients):
        """Test pipeline trigger returns the execution ARN on success."""
        mock_sfn = mock_boto3_clients['sfn']
        expected_arn = 'arn:aws:states:us-east-1:123456789012:execution:fraud-detection-training-pipeline:experiment-exp-001-20240115-120000'
        mock_sfn.start_execution.return_value = {'executionArn': expected_arn}

        result = integrator.trigger_production_pipeline('exp-001')

        assert result == expected_arn

    def test_trigger_passes_correct_experiment_id(self, integrator, mock_boto3_clients):
        """Test trigger passes correct experiment ID in execution input."""
        import json as json_mod

        mock_sfn = mock_boto3_clients['sfn']
        mock_sfn.start_execution.return_value = {'executionArn': 'arn:test'}

        integrator.trigger_production_pipeline('my-experiment-42')

        call_kwargs = mock_sfn.start_execution.call_args[1]
        input_data = json_mod.loads(call_kwargs['input'])
        assert input_data['experimentId'] == 'my-experiment-42'
        assert input_data['triggeredBy'] == 'experimentation-workflow'

    def test_trigger_execution_name_format(self, integrator, mock_boto3_clients):
        """Test trigger execution name follows expected format."""
        mock_sfn = mock_boto3_clients['sfn']
        mock_sfn.start_execution.return_value = {'executionArn': 'arn:test'}

        integrator.trigger_production_pipeline('exp-001')

        call_kwargs = mock_sfn.start_execution.call_args[1]
        execution_name = call_kwargs['name']
        assert execution_name.startswith('experiment-exp-001-')
        # Verify timestamp portion is 15 chars (YYYYMMDD-HHMMSS)
        timestamp_part = execution_name[len('experiment-exp-001-'):]
        assert len(timestamp_part) == 15

    def test_trigger_failure_raises_runtime_error(self, integrator, mock_boto3_clients):
        """Test trigger failure raises RuntimeError with descriptive message."""
        mock_sfn = mock_boto3_clients['sfn']
        mock_sfn.start_execution.side_effect = mock_sfn.exceptions.ClientError(
            'Execution limit exceeded'
        )

        with pytest.raises(RuntimeError, match="Failed to trigger production pipeline"):
            integrator.trigger_production_pipeline('exp-001')

    # --- Pipeline status tests ---

    def test_check_status_returns_correct_fields(self, integrator, mock_boto3_clients):
        """Test check_pipeline_status returns correct fields for completed execution."""
        mock_sfn = mock_boto3_clients['sfn']
        start_date = datetime(2024, 1, 15, 12, 0, 0)
        stop_date = datetime(2024, 1, 15, 13, 0, 0)
        mock_sfn.describe_execution.return_value = {
            'status': 'SUCCEEDED',
            'startDate': start_date,
            'stopDate': stop_date,
            'output': '{"result": "success"}',
        }

        result = integrator.check_pipeline_status('arn:test:execution')

        assert result['status'] == 'SUCCEEDED'
        assert result['startDate'] == start_date
        assert result['stopDate'] == stop_date
        assert result['output'] == '{"result": "success"}'

    def test_check_status_running_execution(self, integrator, mock_boto3_clients):
        """Test check_pipeline_status with running execution (no stopDate/output)."""
        mock_sfn = mock_boto3_clients['sfn']
        start_date = datetime(2024, 1, 15, 12, 0, 0)
        mock_sfn.describe_execution.return_value = {
            'status': 'RUNNING',
            'startDate': start_date,
        }

        result = integrator.check_pipeline_status('arn:test:execution')

        assert result['status'] == 'RUNNING'
        assert result['startDate'] == start_date
        assert result['stopDate'] is None
        assert result['output'] is None

    # --- Promote to production tests ---

    def test_promote_without_pipeline_trigger(
        self, integrator, mock_boto3_clients, valid_hyperparameters, sample_metrics
    ):
        """Test promote_to_production without triggering pipeline."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_s3 = mock_boto3_clients['s3']
        mock_sfn = mock_boto3_clients['sfn']

        mock_ssm.get_parameter.side_effect = mock_ssm.exceptions.ParameterNotFound(
            'not found'
        )
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )

        result = integrator.promote_to_production(
            experiment_id='exp-001',
            hyperparameters=valid_hyperparameters,
            metrics=sample_metrics,
            approver='data-science-team',
            trigger_pipeline=False,
        )

        assert result['execution_arn'] is None
        mock_sfn.start_execution.assert_not_called()

    def test_promote_with_pipeline_trigger(
        self, integrator, mock_boto3_clients, valid_hyperparameters, sample_metrics
    ):
        """Test promote_to_production with pipeline trigger enabled."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_s3 = mock_boto3_clients['s3']
        mock_sfn = mock_boto3_clients['sfn']

        mock_ssm.get_parameter.side_effect = mock_ssm.exceptions.ParameterNotFound(
            'not found'
        )
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )
        expected_arn = 'arn:aws:states:us-east-1:123456789012:execution:pipeline:run-1'
        mock_sfn.start_execution.return_value = {'executionArn': expected_arn}

        result = integrator.promote_to_production(
            experiment_id='exp-001',
            hyperparameters=valid_hyperparameters,
            metrics=sample_metrics,
            approver='data-science-team',
            trigger_pipeline=True,
        )

        assert result['execution_arn'] == expected_arn
        mock_sfn.start_execution.assert_called_once()

    def test_promote_returns_correct_structure(
        self, integrator, mock_boto3_clients, valid_hyperparameters, sample_metrics
    ):
        """Test promote_to_production returns dict with expected keys and values."""
        mock_ssm = mock_boto3_clients['ssm']
        mock_s3 = mock_boto3_clients['s3']

        mock_ssm.get_parameter.side_effect = mock_ssm.exceptions.ParameterNotFound(
            'not found'
        )
        mock_s3.get_object.side_effect = mock_s3.exceptions.NoSuchKey(
            {'Error': {'Code': 'NoSuchKey'}}, 'GetObject'
        )

        result = integrator.promote_to_production(
            experiment_id='exp-001',
            hyperparameters=valid_hyperparameters,
            metrics=sample_metrics,
            approver='data-science-team',
        )

        assert 'promotion_event' in result
        assert 'execution_arn' in result

        event = result['promotion_event']
        assert event['experiment_id'] == 'exp-001'
        assert event['approver'] == 'data-science-team'
        assert event['metrics'] == sample_metrics
        assert 'timestamp' in event
        assert 'backup_key' in event


class TestErrorHandlingAndRollback:
    """Test suite for error handling and rollback utilities.

    Validates: Requirements 14.2, 14.3, 14.4
    """

    @pytest.fixture
    def mock_boto3_clients(self):
        """Create mock boto3 clients for SSM, S3, and Step Functions."""
        with patch('production_integration.boto3.client') as mock_client:
            mock_ssm = MagicMock()
            mock_s3 = MagicMock()
            mock_sfn = MagicMock()

            mock_ssm.exceptions.ParameterNotFound = type(
                'ParameterNotFound', (Exception,), {}
            )

            # Set up NoSuchKey exception on the S3 mock
            mock_s3.exceptions.NoSuchKey = type(
                'NoSuchKey', (Exception,), {}
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

    # --- Parameter Store rollback tests ---

    def test_rollback_parameter_store_restores_values(
        self, integrator, mock_boto3_clients
    ):
        """Test that rollback reads backup from S3 and writes values to Parameter Store."""
        mock_s3 = mock_boto3_clients['s3']
        mock_ssm = mock_boto3_clients['ssm']

        backup_data = {
            '/fraud-detection/hyperparameters/max_depth': '5',
            '/fraud-detection/hyperparameters/eta': '0.1',
        }
        import yaml as _yaml
        mock_body = MagicMock()
        mock_body.read.return_value = _yaml.dump(backup_data).encode()
        mock_s3.get_object.return_value = {'Body': mock_body}

        result = integrator.rollback_parameter_store(
            'parameter-store-backups/backup-20240115-120000.yaml'
        )

        assert result == backup_data
        assert mock_ssm.put_parameter.call_count == 2
        mock_ssm.put_parameter.assert_any_call(
            Name='/fraud-detection/hyperparameters/max_depth',
            Value='5',
            Type='String',
            Overwrite=True,
        )
        mock_ssm.put_parameter.assert_any_call(
            Name='/fraud-detection/hyperparameters/eta',
            Value='0.1',
            Type='String',
            Overwrite=True,
        )

    def test_rollback_parameter_store_skips_none_values(
        self, integrator, mock_boto3_clients
    ):
        """Test that rollback skips parameters that were None in the backup."""
        mock_s3 = mock_boto3_clients['s3']
        mock_ssm = mock_boto3_clients['ssm']

        backup_data = {
            '/fraud-detection/hyperparameters/max_depth': '5',
            '/fraud-detection/hyperparameters/eta': None,
        }
        import yaml as _yaml
        mock_body = MagicMock()
        mock_body.read.return_value = _yaml.dump(backup_data).encode()
        mock_s3.get_object.return_value = {'Body': mock_body}

        integrator.rollback_parameter_store(
            'parameter-store-backups/backup-20240115-120000.yaml'
        )

        # Only max_depth should be written, eta was None
        assert mock_ssm.put_parameter.call_count == 1
        mock_ssm.put_parameter.assert_called_once_with(
            Name='/fraud-detection/hyperparameters/max_depth',
            Value='5',
            Type='String',
            Overwrite=True,
        )

    def test_rollback_parameter_store_s3_read_failure(
        self, integrator, mock_boto3_clients
    ):
        """Test that S3 read failure during rollback raises S3AccessError."""
        mock_s3 = mock_boto3_clients['s3']
        mock_s3.get_object.side_effect = Exception('Access Denied')

        with pytest.raises(S3AccessError, match='Failed to read backup'):
            integrator.rollback_parameter_store(
                'parameter-store-backups/backup-20240115-120000.yaml'
            )

    def test_rollback_parameter_store_ssm_write_failure(
        self, integrator, mock_boto3_clients
    ):
        """Test that SSM write failure during rollback raises ParameterStoreError."""
        mock_s3 = mock_boto3_clients['s3']
        mock_ssm = mock_boto3_clients['ssm']

        backup_data = {
            '/fraud-detection/hyperparameters/max_depth': '5',
        }
        import yaml as _yaml
        mock_body = MagicMock()
        mock_body.read.return_value = _yaml.dump(backup_data).encode()
        mock_s3.get_object.return_value = {'Body': mock_body}
        mock_ssm.put_parameter.side_effect = Exception('AccessDeniedException')

        with pytest.raises(ParameterStoreError, match='Failed to restore parameter'):
            integrator.rollback_parameter_store(
                'parameter-store-backups/backup-20240115-120000.yaml'
            )

    # --- Config file rollback tests ---

    def test_rollback_config_file_copies_backup(
        self, integrator, mock_boto3_clients
    ):
        """Test that rollback copies the backup to the production config location."""
        mock_s3 = mock_boto3_clients['s3']

        integrator.rollback_config_file(
            'archive/production-model-config-20240115-120000.yaml'
        )

        mock_s3.copy_object.assert_called_once_with(
            Bucket='fraud-detection-config',
            CopySource={
                'Bucket': 'fraud-detection-config',
                'Key': 'archive/production-model-config-20240115-120000.yaml',
            },
            Key='production-model-config.yaml',
        )

    def test_rollback_config_file_s3_failure(
        self, integrator, mock_boto3_clients
    ):
        """Test that S3 copy failure during config rollback raises S3AccessError."""
        mock_s3 = mock_boto3_clients['s3']
        mock_s3.copy_object.side_effect = Exception('Access Denied')

        with pytest.raises(S3AccessError, match='Failed to rollback config'):
            integrator.rollback_config_file(
                'archive/production-model-config-20240115-120000.yaml'
            )

    # --- S3 access error handling tests ---

    def test_s3_access_error_includes_permission_info(self):
        """Test that S3AccessError messages include permission requirements."""
        error = S3AccessError(
            "Failed to read from s3://bucket/key. "
            "Ensure your IAM role has s3:GetObject permission for this bucket."
        )
        assert 's3:GetObject' in str(error)
        assert 'permission' in str(error).lower()

    def test_s3_access_error_is_exception(self):
        """Test that S3AccessError is a proper Exception subclass."""
        assert issubclass(S3AccessError, Exception)

    # --- Parameter Store error handling tests ---

    def test_parameter_store_error_includes_rollback_instructions(self):
        """Test that ParameterStoreError messages include rollback instructions."""
        error = ParameterStoreError(
            "Failed to restore parameter '/fraud-detection/hyperparameters/max_depth' "
            "during rollback. Error: AccessDeniedException. "
            "Manual rollback may be required for remaining parameters."
        )
        assert 'rollback' in str(error).lower()
        assert '/fraud-detection/hyperparameters/max_depth' in str(error)

    def test_parameter_store_error_is_exception(self):
        """Test that ParameterStoreError is a proper Exception subclass."""
        assert issubclass(ParameterStoreError, Exception)

    # --- SageMaker training error handling tests ---

    def test_sagemaker_training_error_includes_cloudwatch_reference(self):
        """Test that SageMakerTrainingError messages include CloudWatch log references."""
        error = SageMakerTrainingError(
            "Training job 'my-job' failed: AlgorithmError. "
            "Check CloudWatch logs for details: /aws/sagemaker/TrainingJobs/my-job"
        )
        assert 'CloudWatch' in str(error)
        assert '/aws/sagemaker/TrainingJobs/my-job' in str(error)

    def test_sagemaker_training_error_is_exception(self):
        """Test that SageMakerTrainingError is a proper Exception subclass."""
        assert issubclass(SageMakerTrainingError, Exception)

    def test_handle_sagemaker_training_error_raises_on_failure(self):
        """Test that handle_sagemaker_training_error raises for failed jobs."""
        mock_client = MagicMock()
        mock_client.describe_training_job.return_value = {
            'TrainingJobStatus': 'Failed',
            'FailureReason': 'AlgorithmError: data format issue',
        }

        with pytest.raises(SageMakerTrainingError, match='AlgorithmError'):
            handle_sagemaker_training_error(mock_client, 'my-training-job')

    def test_handle_sagemaker_training_error_includes_log_group(self):
        """Test that the error includes the CloudWatch log group path."""
        mock_client = MagicMock()
        mock_client.describe_training_job.return_value = {
            'TrainingJobStatus': 'Failed',
            'FailureReason': 'ResourceLimitExceeded',
        }

        with pytest.raises(
            SageMakerTrainingError,
            match='/aws/sagemaker/TrainingJobs/my-training-job',
        ):
            handle_sagemaker_training_error(mock_client, 'my-training-job')

    def test_handle_sagemaker_training_error_no_raise_on_success(self):
        """Test that handle_sagemaker_training_error does not raise for completed jobs."""
        mock_client = MagicMock()
        mock_client.describe_training_job.return_value = {
            'TrainingJobStatus': 'Completed',
        }

        # Should not raise
        handle_sagemaker_training_error(mock_client, 'my-training-job')

    def test_handle_sagemaker_training_error_describe_failure(self):
        """Test that describe_training_job failure raises SageMakerTrainingError."""
        mock_client = MagicMock()
        mock_client.describe_training_job.side_effect = Exception('ServiceUnavailable')

        with pytest.raises(SageMakerTrainingError, match='Failed to describe training job'):
            handle_sagemaker_training_error(mock_client, 'my-training-job')

    def test_handle_sagemaker_training_error_unknown_failure_reason(self):
        """Test that unknown failure reason is handled gracefully."""
        mock_client = MagicMock()
        mock_client.describe_training_job.return_value = {
            'TrainingJobStatus': 'Failed',
        }

        with pytest.raises(SageMakerTrainingError, match='Unknown failure'):
            handle_sagemaker_training_error(mock_client, 'my-training-job')
