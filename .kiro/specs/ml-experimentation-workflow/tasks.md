# Implementation Plan: ML Experimentation Workflow

## Overview

This implementation plan breaks down the ML experimentation workflow into discrete coding tasks. The system consists of Python modules for experiment tracking, hyperparameter tuning, production integration, and A/B testing, along with example Jupyter notebooks and CDK infrastructure for SageMaker Studio setup. All code will be written in Python to align with the data science ecosystem.

## Git Commit Guidelines

When completing "Git commit and push" tasks, commits should include:
- A clear, descriptive commit title
- A detailed commit message body that explains:
  - What was implemented
  - Key changes made to each file
  - How the implementation satisfies the requirements
  - Any important technical decisions or patterns used
- Reference to the relevant requirements from the design document

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create Python project structure in ml-experimentation-workflow/
  - Create requirements.txt with dependencies: boto3, sagemaker, pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyyaml, hypothesis, pytest, moto
  - Create src/ directory for Python modules
  - Create tests/ directory with unit/ and property/ subdirectories
  - Create notebooks/ directory for example notebooks
  - Create infrastructure/ directory for CDK stacks
  - Set up .gitignore for Python and Jupyter
  - **Git commit and push**: "feat: set up ML experimentation workflow project structure
    
    - Created Python project structure in ml-experimentation-workflow/
    - Added requirements.txt with all necessary dependencies (boto3, sagemaker, pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyyaml, hypothesis, pytest, moto)
    - Created src/ directory for Python modules
    - Created tests/ directory with unit/ and property/ subdirectories
    - Created notebooks/ directory for example Jupyter notebooks
    - Created infrastructure/ directory for CDK stacks
    - Set up .gitignore for Python and Jupyter artifacts"
  - _Requirements: 1.1, 1.2, 1.3_

- [-] 2. Implement experiment tracking module
  - [x] 2.1 Create ExperimentTracker class with SageMaker Experiments integration
    - Implement __init__ with SageMaker session initialization
    - Implement start_experiment method to create experiment run with unique ID
    - Implement log_parameters method to log hyperparameters
    - Implement log_metrics method to log performance metrics
    - Implement log_artifacts method to upload files to S3
    - Implement query_experiments method with filtering by date, metrics, hyperparameters
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [ ]* 2.2 Write property test for experiment ID uniqueness
    - **Property 12: Experiment ID Uniqueness**
    - **Validates: Requirements 7.1**
  
  - [ ]* 2.3 Write property test for experiment metadata completeness
    - **Property 11: Experiment Metadata Completeness**
    - **Validates: Requirements 7.2, 3.4, 5.3**
  
  - [ ]* 2.4 Write property test for experiment query correctness
    - **Property 13: Experiment Query Correctness**
    - **Validates: Requirements 7.3**
  
  - [x] 2.5 Write unit tests for ExperimentTracker
    - Test experiment creation with mocked SageMaker Experiments
    - Test parameter and metric logging
    - Test query with various filters
    - _Requirements: 7.1, 7.2, 7.3_
  
  - [x] 2.6 Git commit and push
    - **Git commit and push**: "feat: implement experiment tracking module with SageMaker Experiments
    
    - Created ExperimentTracker class with SageMaker Experiments integration
    - Implemented start_experiment method for creating experiment runs with unique IDs
    - Implemented log_parameters method for hyperparameter logging
    - Implemented log_metrics method for performance metrics tracking
    - Implemented log_artifacts method for uploading files to S3
    - Implemented query_experiments method with filtering by date, metrics, and hyperparameters
    - Added property tests validating experiment ID uniqueness, metadata completeness, and query correctness
    - Included unit tests with mocked SageMaker Experiments client"


- [x] 3. Implement hyperparameter tuning module
  - [x] 3.1 Create HyperparameterTuner class with grid search support
    - Implement grid_search method with parameter ranges
    - Integrate with ExperimentTracker to log all trials
    - Return best hyperparameters and metrics
    - _Requirements: 3.1, 3.4_
  
  - [x] 3.2 Add random search support to HyperparameterTuner
    - Implement random_search method with parameter distributions
    - Integrate with ExperimentTracker
    - _Requirements: 3.2, 3.4_
  
  - [x] 3.3 Add SageMaker Automatic Model Tuning support
    - Implement bayesian_optimization method using SageMaker HyperparameterTuner
    - Configure tuning job with objective metric and ranges
    - Wait for completion and retrieve best hyperparameters
    - _Requirements: 3.3_
  
  - [ ]* 3.4 Write property test for grid search completeness
    - **Property 19: Grid Search Completeness**
    - **Validates: Requirements 3.1**
  
  - [ ]* 3.5 Write property test for random search sampling
    - **Property 20: Random Search Sampling**
    - **Validates: Requirements 3.2**
  
  - [x] 3.6 Write unit tests for HyperparameterTuner
    - Test grid search with sample parameter ranges
    - Test random search with sample distributions
    - Test SageMaker tuning job creation
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 3.7 Git commit and push
    - **Git commit and push**: "feat: implement hyperparameter tuning module with multiple search strategies
    
    - Created HyperparameterTuner class with grid search, random search, and Bayesian optimization support
    - Implemented grid_search method with parameter ranges and ExperimentTracker integration
    - Implemented random_search method with parameter distributions
    - Implemented bayesian_optimization method using SageMaker HyperparameterTuner
    - Integrated all tuning methods with ExperimentTracker for trial logging
    - Added property tests validating grid search completeness and random search sampling
    - Included unit tests for all search strategies with mocked SageMaker client"

- [x] 4. Implement algorithm comparison module
  - [x] 4.1 Create AlgorithmComparator class
    - Implement compare_algorithms method supporting XGBoost, LightGBM, Random Forest, Neural Networks
    - Train each algorithm on provided dataset
    - Calculate metrics for each algorithm
    - Integrate with ExperimentTracker to log each algorithm run
    - Return comparison DataFrame
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 4.2 Add visualization utilities
    - Implement visualize_comparison method to generate bar charts
    - Save visualizations to files
    - _Requirements: 4.5_
  
  - [x] 4.3 Write unit tests for AlgorithmComparator
    - Test algorithm comparison with sample dataset
    - Test visualization generation
    - Verify all algorithms are trained and logged
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 4.4 Git commit and push
    - **Git commit and push**: "feat: implement algorithm comparison module for ML model selection
    
    - Created AlgorithmComparator class supporting XGBoost, LightGBM, Random Forest, and Neural Networks
    - Implemented compare_algorithms method for training and evaluating multiple algorithms
    - Calculated comprehensive metrics for each algorithm (accuracy, precision, recall, F1, AUC)
    - Integrated with ExperimentTracker to log all algorithm runs
    - Implemented visualize_comparison method for generating bar chart comparisons
    - Added visualization saving to files for documentation
    - Included unit tests validating algorithm comparison and visualization generation"

- [x] 5. Implement feature engineering module
  - [x] 5.1 Create FeatureEngineer class
    - Implement create_time_features method
    - Implement create_amount_features method
    - Implement create_interaction_features method
    - _Requirements: 5.1_
  
  - [x] 5.2 Add feature selection methods
    - Implement select_features_univariate method using SelectKBest
    - Implement select_features_rfe method using RFE
    - Implement analyze_feature_importance method using Random Forest
    - _Requirements: 5.2_
  
  - [ ]* 5.3 Write property test for feature selection subset
    - **Property 18: Feature Selection Subset**
    - **Validates: Requirements 5.2**
  
  - [x] 5.4 Write unit tests for FeatureEngineer
    - Test time feature creation with sample data
    - Test amount feature creation
    - Test feature selection methods
    - Test feature importance analysis
    - _Requirements: 5.1, 5.2_
  
  - [x] 5.5 Git commit and push
    - **Git commit and push**: "feat: implement feature engineering module for data transformation
    
    - Created FeatureEngineer class with time, amount, and interaction feature creation
    - Implemented create_time_features method for temporal feature extraction
    - Implemented create_amount_features method for transaction amount transformations
    - Implemented create_interaction_features method for feature combinations
    - Implemented select_features_univariate method using SelectKBest
    - Implemented select_features_rfe method using Recursive Feature Elimination
    - Implemented analyze_feature_importance method using Random Forest
    - Added property tests validating feature selection subset properties
    - Included unit tests for all feature engineering and selection methods"

- [-] 6. Checkpoint - Ensure experimentation modules pass tests
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: checkpoint - experimentation modules complete
    
    - Verified all experimentation module tests pass (ExperimentTracker, HyperparameterTuner, AlgorithmComparator, FeatureEngineer)
    - Confirmed SageMaker Experiments integration works correctly
    - Validated hyperparameter tuning strategies function as expected
    - Ensured feature engineering and selection methods produce correct outputs
    - Foundation ready for model evaluation and production integration"


- [ ] 7. Implement model evaluation framework
  - [ ] 7.1 Create ModelEvaluator class
    - Implement calculate_metrics method for accuracy, precision, recall, F1, AUC
    - Implement plot_confusion_matrix method
    - Implement plot_roc_curve method
    - Implement plot_precision_recall_curve method
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 7.2 Add baseline comparison functionality
    - Implement compare_to_baseline method
    - Calculate difference and percent change for each metric
    - _Requirements: 6.4_
  
  - [ ] 7.3 Add production threshold checking
    - Implement evaluate_model method that checks accuracy >= 0.90
    - Return meets_production_threshold flag
    - Generate warning if accuracy < 0.80
    - _Requirements: 6.5, 14.5_
  
  - [ ]* 7.4 Write property test for model evaluation metrics completeness
    - **Property 14: Model Evaluation Metrics Completeness**
    - **Validates: Requirements 6.1**
  
  - [ ]* 7.5 Write property test for production threshold detection
    - **Property 15: Production Threshold Detection**
    - **Validates: Requirements 6.5**
  
  - [ ]* 7.6 Write property test for baseline comparison completeness
    - **Property 16: Baseline Comparison Completeness**
    - **Validates: Requirements 6.4**
  
  - [ ]* 7.7 Write property test for low accuracy warning
    - **Property 17: Low Accuracy Warning**
    - **Validates: Requirements 14.5**
  
  - [ ] 7.8 Write unit tests for ModelEvaluator
    - Test metrics calculation with sample predictions
    - Test visualization generation
    - Test baseline comparison
    - Test threshold checking
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ] 7.9 Git commit and push
    - **Git commit and push**: "feat: implement model evaluation framework with production thresholds
    
    - Created ModelEvaluator class for comprehensive model assessment
    - Implemented calculate_metrics method for accuracy, precision, recall, F1, and AUC
    - Implemented plot_confusion_matrix, plot_roc_curve, and plot_precision_recall_curve methods
    - Implemented compare_to_baseline method for calculating metric differences and percent changes
    - Implemented evaluate_model method with production threshold checking (accuracy >= 0.90)
    - Added warning generation for low accuracy (< 0.80)
    - Added property tests validating metrics completeness, threshold detection, baseline comparison, and warnings
    - Included unit tests for all evaluation methods and threshold validation"

- [ ] 8. Implement production integration module
  - [ ] 8.1 Create ProductionIntegrator class
    - Implement __init__ with boto3 clients (SSM, S3, Step Functions)
    - Implement backup_current_parameters method
    - _Requirements: 8.3_
  
  - [ ] 8.2 Add hyperparameter validation
    - Implement validate_hyperparameters method
    - Check required parameters present
    - Validate value ranges for each parameter
    - Raise descriptive errors for invalid inputs
    - _Requirements: 8.2, 14.1_
  
  - [ ] 8.3 Add Parameter Store write functionality
    - Implement write_hyperparameters_to_parameter_store method
    - Write to all required paths: /fraud-detection/hyperparameters/*
    - Create backup before writing
    - Log promotion event to ExperimentTracker
    - _Requirements: 8.1, 8.4, 8.5_
  
  - [ ]* 8.4 Write property test for Parameter Store path correctness
    - **Property 1: Parameter Store Path Correctness**
    - **Validates: Requirements 8.1, 8.4, 15.2**
  
  - [ ]* 8.5 Write property test for hyperparameter validation
    - **Property 2: Hyperparameter Validation**
    - **Validates: Requirements 8.2, 14.1**
  
  - [ ]* 8.6 Write property test for Parameter Store backup creation
    - **Property 3: Parameter Store Backup Creation**
    - **Validates: Requirements 8.3**
  
  - [ ]* 8.7 Write property test for promotion event logging
    - **Property 4: Promotion Event Logging**
    - **Validates: Requirements 8.5**
  
  - [ ] 8.8 Write unit tests for Parameter Store integration
    - Test hyperparameter validation with valid and invalid inputs
    - Test Parameter Store writes with mocked SSM
    - Test backup creation
    - Test error handling for access denied
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 8.9 Git commit and push
    - **Git commit and push**: "feat: implement production integration module for Parameter Store management
    
    - Created ProductionIntegrator class with boto3 clients (SSM, S3, Step Functions)
    - Implemented backup_current_parameters method for safe parameter updates
    - Implemented validate_hyperparameters method with required parameter and range validation
    - Implemented write_hyperparameters_to_parameter_store method for all /fraud-detection/hyperparameters/* paths
    - Added automatic backup creation before Parameter Store writes
    - Integrated promotion event logging with ExperimentTracker
    - Added property tests validating Parameter Store path correctness, validation, backup creation, and event logging
    - Included unit tests with mocked SSM client for all integration scenarios"


- [ ] 9. Implement configuration file management
  - [ ] 9.1 Add configuration file generation to ProductionIntegrator
    - Implement generate_production_config method
    - Include all required fields: algorithm, hyperparameters, performance, tested_date, approved_by
    - _Requirements: 9.1, 9.2_
  
  - [ ] 9.2 Add configuration file validation
    - Implement validate_config_schema method
    - Check all required fields present and correct types
    - _Requirements: 9.4_
  
  - [ ] 9.3 Add S3 write functionality for config files
    - Implement write_config_to_s3 method
    - Write to s3://fraud-detection-config/production-model-config.yaml
    - Create versioned backup in archive/ directory
    - _Requirements: 9.3, 9.5_
  
  - [ ]* 9.4 Write property test for configuration file YAML format
    - **Property 5: Configuration File YAML Format**
    - **Validates: Requirements 9.1**
  
  - [ ]* 9.5 Write property test for configuration file completeness
    - **Property 6: Configuration File Completeness**
    - **Validates: Requirements 9.2, 9.4**
  
  - [ ]* 9.6 Write property test for configuration file S3 location
    - **Property 7: Configuration File S3 Location**
    - **Validates: Requirements 9.3**
  
  - [ ]* 9.7 Write property test for configuration file backup creation
    - **Property 8: Configuration File Backup Creation**
    - **Validates: Requirements 9.5**
  
  - [ ] 9.8 Write unit tests for configuration file management
    - Test config generation with sample data
    - Test schema validation
    - Test S3 writes with mocked S3
    - Test backup creation
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [ ] 9.9 Git commit and push
    - **Git commit and push**: "feat: implement configuration file management for production model config
    
    - Added generate_production_config method to ProductionIntegrator
    - Included all required fields: algorithm, hyperparameters, performance, tested_date, approved_by
    - Implemented validate_config_schema method for schema validation
    - Implemented write_config_to_s3 method for s3://fraud-detection-config/production-model-config.yaml
    - Added versioned backup creation in archive/ directory with timestamps
    - Added property tests validating YAML format, completeness, S3 location, and backup creation
    - Included unit tests with mocked S3 client for config generation, validation, and storage"

- [ ] 10. Implement production pipeline trigger functionality
  - [ ] 10.1 Add pipeline trigger to ProductionIntegrator
    - Implement trigger_production_pipeline method
    - Start Step Functions execution with experiment ID metadata
    - Return execution ARN
    - Handle trigger failures with descriptive errors
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [ ] 10.2 Add pipeline status checking
    - Implement check_pipeline_status method
    - Query Step Functions execution by ARN
    - Return status, start/stop dates, output
    - _Requirements: 10.5_
  
  - [ ] 10.3 Add complete promotion workflow
    - Implement promote_to_production method
    - Orchestrate Parameter Store write, config file write, and optional pipeline trigger
    - Return promotion event and execution ARN
    - _Requirements: 8.1, 9.1, 10.1_
  
  - [ ]* 10.4 Write property test for pipeline trigger metadata
    - **Property 9: Pipeline Trigger Metadata**
    - **Validates: Requirements 10.2**
  
  - [ ]* 10.5 Write property test for pipeline trigger response
    - **Property 10: Pipeline Trigger Response**
    - **Validates: Requirements 10.3**
  
  - [ ]* 10.6 Write property test for hyperparameter round-trip consistency
    - **Property 23: Hyperparameter Round-Trip Consistency**
    - **Validates: Requirements 15.5**
  
  - [ ] 10.7 Write unit tests for pipeline trigger
    - Test pipeline trigger with mocked Step Functions
    - Test status checking
    - Test complete promotion workflow
    - Test error handling
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 10.8 Git commit and push
    - **Git commit and push**: "feat: implement production pipeline trigger functionality
    
    - Added trigger_production_pipeline method to ProductionIntegrator
    - Implemented Step Functions execution start with experiment ID metadata
    - Added check_pipeline_status method for execution status queries
    - Implemented promote_to_production method orchestrating Parameter Store, config file, and pipeline trigger
    - Added comprehensive error handling for trigger failures with descriptive messages
    - Added property tests validating pipeline trigger metadata, response format, and hyperparameter round-trip consistency
    - Included unit tests with mocked Step Functions client for all trigger scenarios"

- [ ] 11. Checkpoint - Ensure production integration modules pass tests
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: checkpoint - production integration modules complete
    
    - Verified all production integration module tests pass (ProductionIntegrator with Parameter Store, config files, and pipeline trigger)
    - Confirmed Parameter Store integration works correctly with validation and backups
    - Validated configuration file generation and S3 storage
    - Ensured pipeline trigger functionality operates as expected
    - Production integration ready for A/B testing and deployment"


- [ ] 12. Implement A/B testing module
  - [ ] 12.1 Create ABTestingManager class
    - Implement __init__ with boto3 SageMaker client
    - Implement deploy_challenger_endpoint method
    - Create SageMaker model, endpoint config, and endpoint
    - Wait for endpoint to be in service
    - _Requirements: 11.1, 11.2_
  
  - [ ] 12.2 Add traffic split configuration
    - Implement generate_traffic_split_config method
    - Generate rollout plan with stages (1%, 10%, 50%, 100%)
    - Include success criteria
    - _Requirements: 11.3_
  
  - [ ] 12.3 Add endpoint comparison functionality
    - Implement compare_endpoints method
    - Invoke both champion and challenger endpoints with test data
    - Measure latency for each
    - Return comparison metrics
    - _Requirements: 11.4_
  
  - [ ] 12.4 Add challenger promotion
    - Implement promote_challenger_to_champion method
    - Update production endpoint to use challenger config
    - Clean up old challenger endpoint
    - _Requirements: 11.5_
  
  - [ ]* 12.5 Write property test for challenger endpoint naming
    - **Property 21: Challenger Endpoint Naming**
    - **Validates: Requirements 11.2**
  
  - [ ] 12.6 Write unit tests for ABTestingManager
    - Test challenger deployment with mocked SageMaker
    - Test traffic split config generation
    - Test endpoint comparison
    - Test challenger promotion
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [ ] 12.7 Git commit and push
    - **Git commit and push**: "feat: implement A/B testing module for model comparison
    
    - Created ABTestingManager class with SageMaker client integration
    - Implemented deploy_challenger_endpoint method for creating challenger endpoints
    - Implemented generate_traffic_split_config method with rollout stages (1%, 10%, 50%, 100%)
    - Implemented compare_endpoints method for champion vs challenger comparison with latency measurement
    - Implemented promote_challenger_to_champion method for production updates and cleanup
    - Added property tests validating challenger endpoint naming conventions
    - Included unit tests with mocked SageMaker client for all A/B testing scenarios"

- [ ] 13. Implement error handling and rollback utilities
  - [ ] 13.1 Add rollback functionality to ProductionIntegrator
    - Implement rollback_parameter_store method
    - Load backup from S3 and restore parameters
    - _Requirements: 14.4_
  
  - [ ] 13.2 Add config file rollback
    - Implement rollback_config_file method
    - Copy backup to production location
    - _Requirements: 14.4_
  
  - [ ] 13.3 Add comprehensive error handling
    - Add S3 access error handling with descriptive messages
    - Add Parameter Store error handling with rollback instructions
    - Add SageMaker training error handling with CloudWatch log references
    - _Requirements: 14.2, 14.3, 14.4_
  
  - [ ] 13.4 Write unit tests for error handling
    - Test rollback procedures
    - Test error message generation
    - Test various AWS service error scenarios
    - _Requirements: 14.2, 14.3, 14.4_
  
  - [ ] 13.5 Git commit and push
    - **Git commit and push**: "feat: implement error handling and rollback utilities
    
    - Added rollback_parameter_store method to ProductionIntegrator for restoring from backups
    - Implemented rollback_config_file method for configuration file restoration
    - Added comprehensive error handling for S3 access errors with descriptive messages
    - Added Parameter Store error handling with rollback instructions
    - Added SageMaker training error handling with CloudWatch log references
    - Included unit tests validating rollback procedures and error scenarios"

- [ ] 14. Create example notebooks
  - [ ] 14.1 Create data exploration notebook (01_data_exploration.ipynb)
    - Load data from S3
    - Display dataset schema and statistics
    - Create visualizations for feature distributions
    - Analyze class imbalance
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [ ] 14.2 Create hyperparameter tuning notebook (02_hyperparameter_tuning.ipynb)
    - Demonstrate grid search with ExperimentTracker integration
    - Demonstrate random search
    - Demonstrate SageMaker Automatic Model Tuning
    - Show how to retrieve best hyperparameters
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 14.3 Create algorithm comparison notebook (03_algorithm_comparison.ipynb)
    - Compare XGBoost, LightGBM, Random Forest, Neural Networks
    - Generate comparison visualizations
    - Log all experiments to tracker
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [ ] 14.4 Create feature engineering notebook (04_feature_engineering.ipynb)
    - Demonstrate time feature creation
    - Demonstrate amount feature creation
    - Demonstrate feature selection methods
    - Show feature importance analysis
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [ ] 14.5 Create production promotion notebook (05_production_promotion.ipynb)
    - Demonstrate model evaluation with baseline comparison
    - Show how to promote winning configuration to production
    - Demonstrate Parameter Store and config file updates
    - Show how to trigger production pipeline
    - Demonstrate A/B testing workflow
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 10.1, 11.1_
  
  - [ ] 14.6 Git commit and push
    - **Git commit and push**: "feat: create example Jupyter notebooks for ML experimentation workflows
    
    - Created 01_data_exploration.ipynb with data loading, schema display, visualizations, and class imbalance analysis
    - Created 02_hyperparameter_tuning.ipynb demonstrating grid search, random search, and SageMaker Automatic Model Tuning
    - Created 03_algorithm_comparison.ipynb comparing XGBoost, LightGBM, Random Forest, and Neural Networks with visualizations
    - Created 04_feature_engineering.ipynb demonstrating time/amount feature creation and feature selection methods
    - Created 05_production_promotion.ipynb showing complete workflow from evaluation to production promotion and A/B testing
    - All notebooks include ExperimentTracker integration for reproducibility
    - Added clear documentation and explanations in each notebook"


- [ ] 15. Create CDK infrastructure for SageMaker Studio
  - [ ] 15.1 Create SageMaker Studio CDK stack
    - Define SageMaker Studio Domain with IAM authentication
    - Configure default user settings with execution role
    - Set up Jupyter server with ml.t3.medium instance
    - Use data science image with pre-installed libraries
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ] 15.2 Create IAM roles and policies
    - Create SageMaker execution role
    - Add S3 read/write permissions for fraud-detection-data, fraud-detection-config, fraud-detection-models
    - Add SageMaker permissions for training jobs and endpoints
    - Add Parameter Store read/write permissions
    - Add Step Functions execution permissions
    - Add CloudWatch Logs permissions
    - _Requirements: 1.2, 1.3_
  
  - [ ] 15.3 Create S3 buckets if not existing
    - Ensure fraud-detection-config bucket exists
    - Create archive/ prefix for backups
    - Set up lifecycle policies for old backups
    - _Requirements: 8.3, 9.5_
  
  - [ ] 15.4 Write CDK deployment script
    - Create deploy.sh script
    - Synthesize and deploy CDK stack
    - Output SageMaker Studio URL
    - _Requirements: 1.5_
  
  - [ ] 15.5 Git commit and push
    - **Git commit and push**: "feat: create CDK infrastructure for SageMaker Studio environment
    
    - Created SageMaker Studio CDK stack with IAM authentication
    - Configured default user settings with execution role
    - Set up Jupyter server with ml.t3.medium instance and data science image
    - Created IAM roles with S3, SageMaker, Parameter Store, Step Functions, and CloudWatch permissions
    - Ensured fraud-detection-config bucket exists with archive/ prefix for backups
    - Set up lifecycle policies for old backup management
    - Created deploy.sh script for CDK synthesis and deployment with SageMaker Studio URL output"

- [ ] 16. Create documentation
  - [ ] 16.1 Create README.md
    - Document system architecture
    - Add setup instructions for SageMaker Studio
    - Include example workflows
    - Add troubleshooting guidance
    - Document integration with production pipeline
    - _Requirements: 12.1_
  
  - [ ] 16.2 Create notebook template
    - Create template.ipynb with pre-configured experiment tracking
    - Include production integration helper functions
    - Add code version and dependency logging
    - _Requirements: 12.3, 12.4_
  
  - [ ] 16.3 Create reproducibility checklist
    - Document data version tracking
    - Document code version tracking
    - Document hyperparameter tracking
    - Document random seed management
    - _Requirements: 12.5_
  
  - [ ] 16.4 Add inline code documentation
    - Document all class methods with docstrings
    - Add usage examples in docstrings
    - Document integration patterns
    - _Requirements: 12.1_
  
  - [ ] 16.5 Git commit and push
    - **Git commit and push**: "docs: create comprehensive documentation for ML experimentation workflow
    
    - Created README.md with system architecture, setup instructions, example workflows, and troubleshooting guidance
    - Documented integration with production fraud detection pipeline
    - Created template.ipynb with pre-configured experiment tracking and production integration helpers
    - Added code version and dependency logging to notebook template
    - Created reproducibility checklist documenting data, code, hyperparameter, and random seed tracking
    - Added comprehensive docstrings to all class methods with usage examples
    - Documented integration patterns for SageMaker, Parameter Store, and Step Functions"

- [ ]* 17. Write property test for notebook execution logging
  - **Property 22: Notebook Execution Logging**
  - **Validates: Requirements 12.4**

- [ ] 18. Create integration tests
  - [ ] 18.1 Create end-to-end experimentation test
    - Test complete workflow: data load → train → evaluate → promote
    - Use mocked AWS services
    - Verify all components work together
    - _Requirements: 15.1, 15.2, 15.3, 15.5_
  
  - [ ] 18.2 Create A/B testing integration test
    - Test complete A/B workflow: deploy challenger → compare → promote
    - Use mocked SageMaker
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [ ] 18.3 Git commit and push
    - **Git commit and push**: "test: create integration tests for end-to-end workflows
    
    - Created end-to-end experimentation integration test validating complete workflow (data load → train → evaluate → promote)
    - Used mocked AWS services (moto) for reliable test execution
    - Verified all components work together correctly
    - Created A/B testing integration test validating complete workflow (deploy challenger → compare → promote)
    - Tested with mocked SageMaker client
    - Added assertions for workflow state transitions and data flow"

- [ ] 19. Final checkpoint - Ensure all tests pass and documentation is complete
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: final checkpoint - ML experimentation workflow complete
    
    - Verified all unit tests, property tests, and integration tests pass
    - Confirmed all modules implement required functionality correctly
    - Validated CDK infrastructure stack synthesizes without errors
    - Ensured all example notebooks execute successfully
    - Verified documentation is complete and accurate
    - Project ready for deployment to AWS SageMaker Studio environment"

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- All Python code should follow PEP 8 style guidelines
- Use type hints for all function signatures
- Integration tests use moto for mocking AWS services
