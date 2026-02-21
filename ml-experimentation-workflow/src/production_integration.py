"""
Production integration module for promoting ML experiments to production.

This module provides utilities for promoting winning experiment configurations
to the production fraud-detection-ml-pipeline via AWS Systems Manager Parameter Store.
It handles hyperparameter validation, backup creation, and promotion event logging
to ensure safe and traceable production updates.

Key capabilities:
    - Validate hyperparameter names and value ranges before writing
    - Backup current Parameter Store values to S3 before updates
    - Write validated hyperparameters to Parameter Store paths
    - Log promotion events to ExperimentTracker for audit trail

Example:
    from production_integration import ProductionIntegrator
    from experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker()
    integrator = ProductionIntegrator(experiment_tracker=tracker)

    hyperparameters = {
        'objective': 'binary:logistic',
        'num_round': 150,
        'max_depth': 7,
        'eta': 0.15,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    backup_key = integrator.write_hyperparameters_to_parameter_store(
        hyperparameters, experiment_id='exp-20240115-001'
    )
"""

import boto3
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# Parameter Store path mapping for fraud detection hyperparameters
PARAM_PATHS: Dict[str, str] = {
    'objective': '/fraud-detection/hyperparameters/objective',
    'num_round': '/fraud-detection/hyperparameters/num_round',
    'max_depth': '/fraud-detection/hyperparameters/max_depth',
    'eta': '/fraud-detection/hyperparameters/eta',
    'subsample': '/fraud-detection/hyperparameters/subsample',
    'colsample_bytree': '/fraud-detection/hyperparameters/colsample_bytree',
}

# Required hyperparameter names
REQUIRED_PARAMS: List[str] = [
    'objective', 'num_round', 'max_depth',
    'eta', 'subsample', 'colsample_bytree',
]


class ProductionIntegrator:
    """
    Utilities for promoting experiments to production.

    Provides methods to validate hyperparameters, back up current Parameter Store
    values, write new hyperparameters, and log promotion events via an optional
    ExperimentTracker instance.

    Args:
        experiment_tracker: Optional ExperimentTracker instance for logging
            promotion events. If not provided, promotion events are not logged.

    Example:
        integrator = ProductionIntegrator()
        integrator.validate_hyperparameters({
            'objective': 'binary:logistic',
            'num_round': 150,
            'max_depth': 7,
            'eta': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        })
    """

    def __init__(self, experiment_tracker: Optional[Any] = None) -> None:
        """
        Initialize ProductionIntegrator with boto3 clients.

        Args:
            experiment_tracker: Optional ExperimentTracker instance for logging
                promotion events. When provided, promotion events are recorded
                as experiments with relevant metadata.

        Example:
            from experiment_tracking import ExperimentTracker

            tracker = ExperimentTracker()
            integrator = ProductionIntegrator(experiment_tracker=tracker)
        """
        self.ssm_client = boto3.client('ssm')
        self.s3_client = boto3.client('s3')
        self.sfn_client = boto3.client('stepfunctions')
        self.config_bucket = 'fraud-detection-config'
        self.experiment_tracker = experiment_tracker

    def backup_current_parameters(self) -> Tuple[Dict[str, Optional[str]], str]:
        """
        Backup current Parameter Store values before updating.

        Reads all fraud-detection hyperparameter values from Parameter Store
        and saves them as a YAML file in S3. Parameters that do not exist
        are recorded as None in the backup.

        Returns:
            A tuple of (backup_dict, s3_backup_key) where backup_dict maps
            Parameter Store paths to their current values (or None if not found),
            and s3_backup_key is the S3 key where the backup was stored.

        Example:
            integrator = ProductionIntegrator()
            backup, backup_key = integrator.backup_current_parameters()
            print(f"Backup saved to s3://{integrator.config_bucket}/{backup_key}")
        """
        backup: Dict[str, Optional[str]] = {}
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        for param_path in PARAM_PATHS.values():
            try:
                response = self.ssm_client.get_parameter(Name=param_path)
                backup[param_path] = response['Parameter']['Value']
            except self.ssm_client.exceptions.ParameterNotFound:
                backup[param_path] = None

        # Save backup to S3
        backup_key = f'parameter-store-backups/backup-{timestamp}.yaml'
        self.s3_client.put_object(
            Bucket=self.config_bucket,
            Key=backup_key,
            Body=yaml.dump(backup),
        )

        return backup, backup_key

    def validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> bool:
        """
        Validate hyperparameter names and value ranges.

        Checks that all required hyperparameters are present and that numeric
        parameters fall within their valid ranges:
            - max_depth: integer in [1, 20]
            - eta: float in (0.0, 1.0]
            - num_round: integer in [1, 1000]
            - subsample: float in (0.0, 1.0]
            - colsample_bytree: float in (0.0, 1.0]

        Args:
            hyperparameters: Dictionary of hyperparameter names to values.

        Returns:
            True if all validations pass.

        Raises:
            ValueError: If a required parameter is missing, a value cannot be
                converted to the expected type, or a value is out of range.

        Example:
            integrator = ProductionIntegrator()
            integrator.validate_hyperparameters({
                'objective': 'binary:logistic',
                'num_round': 150,
                'max_depth': 7,
                'eta': 0.15,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            })
        """
        # Check all required parameters are present
        missing = [p for p in REQUIRED_PARAMS if p not in hyperparameters]
        if missing:
            raise ValueError(
                f"Missing required hyperparameters: {', '.join(missing)}"
            )

        # Validation rules: (min, max, type_func, inclusive_min)
        # inclusive_min=False means strictly greater than min (0.0 < x)
        # inclusive_min=True means greater than or equal (1 <= x)
        validations: Dict[str, Tuple[float, float, type, bool]] = {
            'max_depth': (1, 20, int, True),
            'eta': (0.0, 1.0, float, False),
            'num_round': (1, 1000, int, True),
            'subsample': (0.0, 1.0, float, False),
            'colsample_bytree': (0.0, 1.0, float, False),
        }

        for param, (min_val, max_val, type_func, inclusive_min) in validations.items():
            if param in hyperparameters:
                raw_value = hyperparameters[param]
                try:
                    value = type_func(raw_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid value for hyperparameter '{param}': "
                        f"cannot convert {raw_value!r} to {type_func.__name__}: {e}"
                    )

                if inclusive_min:
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"Hyperparameter '{param}' value {value} out of valid range "
                            f"[{min_val}, {max_val}]"
                        )
                else:
                    if not (min_val < value <= max_val):
                        raise ValueError(
                            f"Hyperparameter '{param}' value {value} out of valid range "
                            f"({min_val}, {max_val}]"
                        )

        return True

    def write_hyperparameters_to_parameter_store(
        self,
        hyperparameters: Dict[str, Any],
        experiment_id: Optional[str] = None,
    ) -> str:
        """
        Write validated hyperparameters to Parameter Store.

        Validates the hyperparameters, creates a backup of current values,
        writes new values to all required Parameter Store paths, and optionally
        logs the promotion event to ExperimentTracker.

        Args:
            hyperparameters: Dictionary of hyperparameter names to values.
                Must include all required parameters: objective, num_round,
                max_depth, eta, subsample, colsample_bytree.
            experiment_id: Optional experiment ID to associate with the
                promotion event. Used for logging to ExperimentTracker.

        Returns:
            The S3 key where the backup was stored.

        Raises:
            ValueError: If hyperparameters fail validation.

        Example:
            integrator = ProductionIntegrator(experiment_tracker=tracker)
            backup_key = integrator.write_hyperparameters_to_parameter_store(
                hyperparameters={
                    'objective': 'binary:logistic',
                    'num_round': 150,
                    'max_depth': 7,
                    'eta': 0.15,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                },
                experiment_id='xgboost-20240115-abc12345',
            )
        """
        # Validate first
        self.validate_hyperparameters(hyperparameters)

        # Backup current values
        backup, backup_key = self.backup_current_parameters()

        # Write new values to Parameter Store
        for param_name, param_path in PARAM_PATHS.items():
            value = str(hyperparameters[param_name])
            self.ssm_client.put_parameter(
                Name=param_path,
                Value=value,
                Type='String',
                Overwrite=True,
            )

        # Log promotion event to ExperimentTracker if available
        if self.experiment_tracker is not None:
            self._log_promotion_event(
                hyperparameters=hyperparameters,
                backup_key=backup_key,
                experiment_id=experiment_id,
            )

        return backup_key

    def _log_promotion_event(
        self,
        hyperparameters: Dict[str, Any],
        backup_key: str,
        experiment_id: Optional[str] = None,
    ) -> None:
        """
        Log a promotion event to ExperimentTracker.

        Creates a new experiment run to record the promotion, logs the
        hyperparameters and metadata, then closes the run.

        Args:
            hyperparameters: The hyperparameters that were promoted.
            backup_key: The S3 key of the parameter backup.
            experiment_id: Optional source experiment ID.
        """
        tracker = self.experiment_tracker
        promotion_exp_id = tracker.start_experiment(
            experiment_name='production-promotion',
            algorithm='promotion',
        )

        promotion_params: Dict[str, Any] = {
            'action': 'promote_hyperparameters',
            'backup_key': backup_key,
            'timestamp': datetime.now().isoformat(),
        }
        if experiment_id is not None:
            promotion_params['source_experiment_id'] = experiment_id

        # Log the promoted hyperparameters
        for param_name, param_value in hyperparameters.items():
            promotion_params[f'promoted_{param_name}'] = param_value

        tracker.log_parameters(promotion_exp_id, promotion_params)
        tracker.close_experiment(promotion_exp_id)
    def generate_production_config(
        self,
        experiment_id: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        approver: str,
    ) -> Dict[str, Any]:
        """
        Generate production configuration dictionary.

        Creates a configuration dictionary containing model algorithm,
        hyperparameters, performance metrics, test date, and approver
        information suitable for writing to S3 as a YAML config file.

        Args:
            experiment_id: Unique identifier for the experiment that
                produced the winning configuration.
            hyperparameters: Dictionary of hyperparameter names to values.
            metrics: Dictionary of performance metric names to values
                (e.g. accuracy, precision, recall).
            approver: Name or identifier of the person approving the
                configuration for production use.

        Returns:
            A dictionary with a top-level 'model' key containing algorithm,
            version, hyperparameters, performance, tested_date, and
            approved_by fields.

        Example:
            integrator = ProductionIntegrator()
            config = integrator.generate_production_config(
                experiment_id='exp-20240115-001',
                hyperparameters={'max_depth': 7, 'eta': 0.15},
                metrics={'accuracy': 0.96, 'precision': 0.92},
                approver='data-science-team',
            )
        """
        config: Dict[str, Any] = {
            'model': {
                'algorithm': 'xgboost',
                'version': experiment_id,
                'hyperparameters': dict(hyperparameters),
                'performance': dict(metrics),
                'tested_date': datetime.now().strftime('%Y-%m-%d'),
                'approved_by': approver,
            }
        }
        return config
    def validate_config_schema(self, config: Dict[str, Any]) -> bool:
        """
        Validate production configuration schema.

        Checks that the configuration dictionary has the required structure:
        a top-level 'model' key containing algorithm (str), hyperparameters
        (dict), performance (dict), tested_date (str), and approved_by (str).

        Args:
            config: Configuration dictionary to validate.

        Returns:
            True if the schema is valid.

        Raises:
            ValueError: If the 'model' key is missing, any required field
                is absent, or a field has an incorrect type.

        Example:
            integrator = ProductionIntegrator()
            config = integrator.generate_production_config(...)
            integrator.validate_config_schema(config)  # returns True
        """
        if 'model' not in config:
            raise ValueError("Configuration missing required top-level 'model' key")

        model = config['model']

        required_fields: Dict[str, type] = {
            'algorithm': str,
            'hyperparameters': dict,
            'performance': dict,
            'tested_date': str,
            'approved_by': str,
        }

        for field, expected_type in required_fields.items():
            if field not in model:
                raise ValueError(
                    f"Configuration 'model' section missing required field: '{field}'"
                )
            if not isinstance(model[field], expected_type):
                raise ValueError(
                    f"Configuration field 'model.{field}' must be {expected_type.__name__}, "
                    f"got {type(model[field]).__name__}"
                )

        return True
    def write_config_to_s3(self, config: Dict[str, Any]) -> None:
        """
        Write production configuration to S3.

        Validates the configuration schema, archives the current production
        config (if one exists), and writes the new config as YAML to
        ``s3://fraud-detection-config/production-model-config.yaml``.

        Args:
            config: Configuration dictionary to write. Must pass
                :meth:`validate_config_schema` validation.

        Raises:
            ValueError: If the configuration fails schema validation.

        Example:
            integrator = ProductionIntegrator()
            config = integrator.generate_production_config(
                experiment_id='exp-001',
                hyperparameters={'max_depth': 7},
                metrics={'accuracy': 0.96},
                approver='team-lead',
            )
            integrator.write_config_to_s3(config)
        """
        # Validate schema first
        self.validate_config_schema(config)

        # Archive current config if it exists
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        try:
            current_config = self.s3_client.get_object(
                Bucket=self.config_bucket,
                Key='production-model-config.yaml',
            )
            self.s3_client.put_object(
                Bucket=self.config_bucket,
                Key=f'archive/production-model-config-{timestamp}.yaml',
                Body=current_config['Body'].read(),
            )
        except self.s3_client.exceptions.NoSuchKey:
            pass  # No existing config to archive

        # Write new config
        self.s3_client.put_object(
            Bucket=self.config_bucket,
            Key='production-model-config.yaml',
            Body=yaml.dump(config),
        )
