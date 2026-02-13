# Implementation Plan: Fraud Detection ML Pipeline

## Overview

This implementation plan breaks down the fraud detection ML pipeline into discrete coding tasks. The system consists of two Step Functions workflows (training and inference) with Lambda handlers in Kotlin and a Glue job in Python for data preparation. All handlers extend the CEAP WorkflowLambdaHandler base class for S3 orchestration.

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
  - Create Gradle multi-module project in CEAPFraudDetection workspace
  - Add CEAP platform as Git submodule at `ceap-platform/`
  - Configure Gradle dependencies: AWS SDK v2, Kotlin, Jackson, Kotest
  - Create module structure: `fraud-detection-common`, `fraud-detection-training`, `fraud-detection-inference`
  - Set up build configuration for Lambda deployment packages
  - **Git commit and push**: "feat: set up fraud detection ML pipeline project structure
    
    - Created Gradle multi-module project with fraud-detection-common, fraud-detection-training, and fraud-detection-inference modules
    - Added CEAP platform as Git submodule at ceap-platform/
    - Configured AWS SDK v2, Kotlin, Jackson, and Kotest dependencies
    - Set up build configuration for Lambda deployment packages with fat JAR assembly"
  - _Requirements: 1.1, 1.2, 10.1_

- [ ] 2. Implement core data models
  - [x] 2.1 Create Transaction and ScoredTransaction data classes
    - Define Transaction with id, timestamp, amount, merchantCategory, features
    - Define ScoredTransaction extending Transaction with fraudScore and scoringTimestamp
    - Add Jackson annotations for JSON serialization
    - _Requirements: 6.1, 6.4, 7.3_
  
  - [x] 2.2 Write property test for Transaction model
    - **Property 10: Fraud Score Range**
    - **Validates: Requirements 6.4**
  
  - [x] 2.3 Create workflow models (ExecutionContext, StageResult)
    - Define ExecutionContext with executionId, currentStage, previousStage, workflowBucket, initialData
    - Define StageResult with status, stage, recordsProcessed, errorMessage
    - _Requirements: 2.1, 10.4_
  
  - [x] 2.4 Git commit and push
    - **Git commit and push**: "feat: implement core data models for fraud detection pipeline
    
    - Created Transaction data class with id, timestamp, amount, merchantCategory, and features map
    - Created ScoredTransaction extending Transaction with fraudScore and scoringTimestamp
    - Implemented ExecutionContext for workflow state management with executionId, stage tracking, and S3 bucket configuration
    - Implemented StageResult for stage output with status, recordsProcessed, and error handling
    - Added Jackson annotations for JSON serialization/deserialization
    - Included property tests validating fraud score range constraints"


- [ ] 3. Implement WorkflowLambdaHandler base class
  - [x] 3.1 Create abstract WorkflowLambdaHandler class
    - Implement RequestHandler interface for Lambda
    - Add S3Client and ObjectMapper initialization
    - Implement handleRequest method with error handling
    - _Requirements: 1.3, 10.2, 10.4_
  
  - [x] 3.2 Implement S3 input reading logic
    - Add readInput method with convention-based path resolution
    - Handle first stage (initialData) vs non-first stage (S3) logic
    - Implement S3 error handling (403, 404, 503, 500)
    - _Requirements: 2.3, 2.4, 3.1, 13.2, 15.1_
  
  - [x] 3.3 Implement S3 output writing logic
    - Add writeOutput method with convention-based path resolution
    - Implement S3 error handling for writes
    - Add comprehensive logging for S3 operations
    - _Requirements: 2.2, 2.5, 13.1, 13.3_
  
  - [x] 3.4 Write property tests for S3 orchestration
    - **Property 1: S3 Output Convention**
    - **Property 2: S3 Input Convention**
    - **Property 3: S3 Path Construction**
    - **Validates: Requirements 2.5, 3.1, 13.1, 13.2, 13.3**
  
  - [x] 3.5 Write property test for error handling
    - **Property 5: Error Handling**
    - **Validates: Requirements 10.4, 15.1**
  
  - [x] 3.6 Git commit and push
    - **Git commit and push**: "feat: implement WorkflowLambdaHandler base class for S3 orchestration
    
    - Created abstract WorkflowLambdaHandler implementing AWS Lambda RequestHandler interface
    - Implemented convention-based S3 input reading with support for first stage (initialData) and subsequent stages (S3 paths)
    - Implemented convention-based S3 output writing with standardized path structure (s3://bucket/execId/stage/output.json)
    - Added comprehensive error handling for S3 operations (403, 404, 503, 500 errors)
    - Integrated S3Client and ObjectMapper for AWS operations and JSON processing
    - Added detailed logging for all S3 operations and error conditions
    - Included property tests validating S3 path conventions and error handling behavior"

- [x] 4. Checkpoint - Ensure base handler tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: checkpoint - base handler tests passing
    
    - Verified all WorkflowLambdaHandler property tests pass
    - Confirmed S3 orchestration conventions work correctly
    - Validated error handling behavior meets requirements
    - Ensured foundation is solid before building pipeline handlers"


- [ ] 5. Implement Glue data preparation script
  - [x] 5.1 Create PySpark data preparation script
    - Write data-prep.py using AWS Glue PySpark
    - Implement dataset loading from S3
    - Implement train/validation/test split (70/15/15)
    - Write prepared datasets to S3 in Parquet format
    - Add data quality validation
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 11.1, 11.2, 11.4, 11.5_
  
  - [x] 5.2 Write property test for data split proportions
    - **Property 7: Data Split Proportions**
    - **Validates: Requirements 2.3**
  
  - [x] 5.3 Write property test for SageMaker output format
    - **Property 8: SageMaker Output Format**
    - **Validates: Requirements 2.4**
  
  - [x] 5.4 Git commit and push
    - **Git commit and push**: "feat: implement Glue data preparation script for ML training
    
    - Created PySpark data-prep.py script using AWS Glue framework
    - Implemented dataset loading from S3 with configurable input paths
    - Implemented 70/15/15 train/validation/test split with stratification
    - Added data quality validation checks (null values, feature completeness, label distribution)
    - Implemented Parquet output format compatible with SageMaker XGBoost
    - Added comprehensive logging for data statistics and quality metrics
    - Included property tests validating split proportions and SageMaker output format"

- [ ] 6. Implement TrainHandler
  - [x] 6.1 Create TrainHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Add SageMaker client initialization
    - Configure training job with XGBoost algorithm
    - Implement training job creation and waiting logic
    - Return training job metadata
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 6.2 Write unit tests for TrainHandler
    - Test training job configuration
    - Test training job creation with mocked SageMaker client
    - Test error handling for training failures
    - _Requirements: 3.2, 3.3_
  
  - [x] 6.3 Git commit and push
    - **Git commit and push**: "feat: implement TrainHandler for SageMaker model training
    
    - Created TrainHandler extending WorkflowLambdaHandler for training workflow stage
    - Integrated SageMaker client for training job management
    - Configured XGBoost algorithm with hyperparameters (max_depth=5, eta=0.2, objective=binary:logistic)
    - Implemented training job creation with S3 input/output paths from workflow context
    - Added training job status polling with configurable wait logic
    - Implemented error handling for training failures with detailed error messages
    - Returned training job metadata including model artifact location
    - Included unit tests with mocked SageMaker client validating job configuration and error handling"


- [x] 7. Implement EvaluateHandler
  - [x] 7.1 Create EvaluateHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Create temporary SageMaker endpoint for evaluation
    - Load test data and run predictions
    - Calculate accuracy, precision, recall, F1, AUC metrics
    - Validate accuracy >= 0.90 threshold
    - Clean up evaluation endpoint
    - _Requirements: 3.4, 3.5, 3.6_
  
  - [x] 7.2 Write unit tests for EvaluateHandler
    - Test model evaluation with mocked endpoint
    - Test accuracy threshold validation (should fail if < 0.90)
    - Test metrics calculation
    - _Requirements: 3.6_
  
  - [x] 7.3 Git commit and push
    - **Git commit and push**: "feat: implement EvaluateHandler for model evaluation
    
    - Created EvaluateHandler extending WorkflowLambdaHandler for evaluation workflow stage
    - Implemented temporary SageMaker endpoint creation for model evaluation
    - Loaded test dataset from S3 and executed batch predictions
    - Calculated comprehensive metrics: accuracy, precision, recall, F1 score, and AUC
    - Implemented accuracy threshold validation (>= 0.90) with failure handling
    - Added automatic cleanup of temporary evaluation endpoint
    - Implemented error handling for endpoint creation and prediction failures
    - Included unit tests validating metrics calculation and threshold enforcement"

- [-] 8. Implement DeployHandler
  - [x] 8.1 Create DeployHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Create SageMaker model from artifact
    - Create endpoint configuration
    - Create or update production endpoint
    - Implement endpoint health check with test transaction
    - Write endpoint metadata to config bucket
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 8.2 Write unit tests for DeployHandler
    - Test endpoint creation with mocked SageMaker client
    - Test endpoint update for existing endpoints
    - Test health check validation
    - _Requirements: 4.2, 4.3, 4.4_
  
  - [x] 8.3 Git commit and push
    - **Git commit and push**: "feat: implement DeployHandler for production model deployment
    
    - Created DeployHandler extending WorkflowLambdaHandler for deployment workflow stage
    - Implemented SageMaker model creation from training artifact
    - Created endpoint configuration with instance type and count specifications
    - Implemented endpoint creation for new deployments and updates for existing endpoints
    - Added endpoint health check with test transaction validation
    - Wrote endpoint metadata (name, ARN, timestamp) to config bucket for inference pipeline
    - Implemented error handling for deployment failures and rollback scenarios
    - Included unit tests validating endpoint creation, updates, and health checks"

- [x] 9. Checkpoint - Ensure training pipeline handlers pass tests
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: checkpoint - training pipeline handlers complete
    
    - Verified all training pipeline handler tests pass (TrainHandler, EvaluateHandler, DeployHandler)
    - Confirmed SageMaker integration works correctly for training, evaluation, and deployment
    - Validated end-to-end training workflow from data prep through model deployment
    - Ensured training pipeline is ready for infrastructure deployment"


- [ ] 10. Implement ScoreHandler
  - [ ] 10.1 Create ScoreHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Read current endpoint name from config bucket
    - Load transaction batch from S3
    - Implement endpoint invocation for each transaction
    - Create ScoredTransaction objects with fraud scores
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ] 10.2 Write property test for endpoint invocation
    - **Property 11: Endpoint Invocation**
    - **Validates: Requirements 6.3**
  
  - [ ] 10.3 Write unit tests for ScoreHandler
    - Test transaction batch loading
    - Test endpoint invocation with mocked SageMaker Runtime client
    - Test scored transaction creation
    - _Requirements: 6.1, 6.2, 6.3_
  
  - [ ] 10.4 Git commit and push
    - **Git commit and push**: "feat: implement ScoreHandler for real-time fraud scoring
    
    - Created ScoreHandler extending WorkflowLambdaHandler for scoring workflow stage
    - Implemented endpoint name retrieval from config bucket (written by DeployHandler)
    - Loaded transaction batches from S3 with configurable batch sizes
    - Integrated SageMaker Runtime client for real-time endpoint invocation
    - Implemented per-transaction scoring with feature vector transformation
    - Created ScoredTransaction objects with fraud scores and timestamps
    - Added error handling for endpoint invocation failures and retries
    - Included property tests validating endpoint invocation behavior and unit tests for batch processing"

- [ ] 11. Implement StoreHandler
  - [ ] 11.1 Create StoreHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Implement DynamoDB batch write logic (max 25 items)
    - Handle unprocessed items and track error count
    - Calculate summary statistics (risk distribution, avg score)
    - _Requirements: 7.1, 7.2, 7.3, 7.5_
  
  - [ ] 11.2 Write property test for DynamoDB write completeness
    - **Property 12: DynamoDB Write Completeness**
    - **Validates: Requirements 7.2, 7.3**
  
  - [ ] 11.3 Write unit tests for StoreHandler
    - Test batch write with mocked DynamoDB client
    - Test unprocessed items handling
    - Test summary statistics calculation
    - _Requirements: 7.2, 7.3, 7.5_
  
  - [ ] 11.4 Git commit and push
    - **Git commit and push**: "feat: implement StoreHandler for DynamoDB persistence
    
    - Created StoreHandler extending WorkflowLambdaHandler for storage workflow stage
    - Implemented DynamoDB batch write logic with 25-item batch limit
    - Added unprocessed items retry logic with exponential backoff
    - Tracked error count and failed items for monitoring
    - Calculated summary statistics: risk distribution (high/medium/low), average fraud score, total transactions
    - Implemented error handling for DynamoDB throttling and service errors
    - Added comprehensive logging for batch operations and retry attempts
    - Included property tests validating write completeness and unit tests for batch processing and statistics"


- [ ] 12. Implement AlertHandler
  - [ ] 12.1 Create AlertHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Filter high-risk transactions (fraud score >= 0.8)
    - Implement alert batching (max 100 per message)
    - Build alert message with transaction details
    - Publish alerts to SNS topic
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 12.2 Write property test for high-risk identification
    - **Property 13: High-Risk Transaction Identification**
    - **Validates: Requirements 8.2**
  
  - [ ] 12.3 Write property test for alert batching
    - **Property 15: Alert Batching**
    - **Validates: Requirements 8.5**
  
  - [ ] 12.4 Write property test for alert message completeness
    - **Property 14: Alert Message Completeness**
    - **Validates: Requirements 8.4**
  
  - [ ] 12.5 Write unit tests for AlertHandler
    - Test high-risk filtering
    - Test alert message formatting
    - Test SNS publish with mocked client
    - _Requirements: 8.2, 8.3, 8.4_
  
  - [ ] 12.6 Git commit and push
    - **Git commit and push**: "feat: implement AlertHandler for high-risk transaction alerts
    
    - Created AlertHandler extending WorkflowLambdaHandler for alerting workflow stage
    - Implemented high-risk transaction filtering (fraud score >= 0.8)
    - Added alert batching logic with 100 transactions per SNS message limit
    - Built structured alert messages with transaction details (id, amount, score, timestamp, merchant)
    - Integrated SNS client for alert publishing to configured topic
    - Implemented error handling for SNS publish failures with retry logic
    - Added alert statistics tracking (total high-risk, batches sent)
    - Included property tests validating high-risk identification, batching, and message completeness"

- [ ] 13. Implement MonitorHandler
  - [ ] 13.1 Create MonitorHandler extending WorkflowLambdaHandler
    - Implement processData method
    - Load historical baseline from S3 metrics
    - Calculate distribution metrics (high/medium/low risk percentages)
    - Implement drift detection logic (avg score drift > 0.1 OR high risk drift > 0.05)
    - Send monitoring alert if drift detected
    - Write metrics to S3 for historical tracking
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  
  - [ ] 13.2 Write property test for drift detection
    - **Property 16: Distribution Drift Detection**
    - **Validates: Requirements 14.2**
  
  - [ ] 13.3 Write property test for metrics persistence
    - **Property 17: Metrics Persistence**
    - **Validates: Requirements 14.3**
  
  - [ ] 13.4 Write unit tests for MonitorHandler
    - Test baseline calculation
    - Test drift detection with various distributions
    - Test metrics writing to S3
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  
  - [ ] 13.5 Git commit and push
    - **Git commit and push**: "feat: implement MonitorHandler for distribution drift detection
    
    - Created MonitorHandler extending WorkflowLambdaHandler for monitoring workflow stage
    - Implemented historical baseline loading from S3 metrics with fallback for first run
    - Calculated distribution metrics: high/medium/low risk percentages and average fraud score
    - Implemented drift detection logic (avg score drift > 0.1 OR high risk % drift > 0.05)
    - Integrated SNS client for monitoring alerts when drift detected
    - Wrote current metrics to S3 for historical tracking and baseline updates
    - Added comprehensive logging for drift analysis and alert triggers
    - Included property tests validating drift detection thresholds and metrics persistence"

- [ ] 14. Checkpoint - Ensure inference pipeline handlers pass tests
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: checkpoint - inference pipeline handlers complete
    
    - Verified all inference pipeline handler tests pass (ScoreHandler, StoreHandler, AlertHandler, MonitorHandler)
    - Confirmed SageMaker Runtime, DynamoDB, and SNS integrations work correctly
    - Validated end-to-end inference workflow from scoring through monitoring
    - Ensured inference pipeline is ready for infrastructure deployment"


- [ ] 15. Create CDK infrastructure stacks
  - [ ] 15.1 Create TrainingPipelineStack
    - Define Standard workflow with 4 stages: DataPrep (Glue), Train, Evaluate, Deploy
    - Configure Glue job with data-prep.py script
    - Configure Lambda functions for Train, Evaluate, Deploy handlers
    - Set up S3 buckets for workflow, data, models, config
    - Configure IAM roles with SageMaker permissions
    - Add EventBridge schedule for weekly execution
    - _Requirements: 1.5, 5.1, 5.2, 12.1, 12.2_
  
  - [ ] 15.2 Create InferencePipelineStack
    - Define Express workflow with 4 stages: Score, Store, Alert, Monitor
    - Configure Lambda functions for all handlers
    - Set up DynamoDB table with GSI
    - Configure SNS topics for alerts and monitoring
    - Configure IAM roles with SageMaker Runtime and DynamoDB permissions
    - Add EventBridge schedule for daily execution
    - _Requirements: 1.5, 9.1, 9.2, 12.1, 12.2, 12.3_
  
  - [ ] 15.3 Write unit tests for CDK stacks
    - Test stack synthesis
    - Verify resource creation
    - _Requirements: 12.1, 12.2_
  
  - [ ] 15.4 Git commit and push
    - **Git commit and push**: "feat: create CDK infrastructure stacks for both pipelines
    
    - Created TrainingPipelineStack with Standard workflow (DataPrep, Train, Evaluate, Deploy stages)
    - Configured Glue job with data-prep.py script and appropriate IAM permissions
    - Created Lambda functions for Train, Evaluate, Deploy handlers with SageMaker permissions
    - Set up S3 buckets for workflow orchestration, data storage, model artifacts, and config
    - Added EventBridge schedule for weekly training execution
    - Created InferencePipelineStack with Express workflow (Score, Store, Alert, Monitor stages)
    - Configured Lambda functions for all inference handlers with appropriate permissions
    - Set up DynamoDB table with GSI for transaction queries
    - Configured SNS topics for alerts and monitoring notifications
    - Added EventBridge schedule for daily inference execution
    - Included unit tests validating stack synthesis and resource creation"

- [ ] 16. Create deployment scripts
  - [ ] 16.1 Create deploy-training-pipeline.sh script
    - Build Gradle project
    - Package Lambda functions
    - Upload Glue script to S3
    - Deploy CDK stack
    - _Requirements: 1.5, 12.5_
  
  - [ ] 16.2 Create deploy-inference-pipeline.sh script
    - Build Gradle project
    - Package Lambda functions
    - Deploy CDK stack
    - _Requirements: 1.5, 12.5_
  
  - [ ] 16.3 Git commit and push
    - **Git commit and push**: "feat: create deployment scripts for pipeline automation
    
    - Created deploy-training-pipeline.sh script with Gradle build, Lambda packaging, Glue script upload, and CDK deployment
    - Created deploy-inference-pipeline.sh script with Gradle build, Lambda packaging, and CDK deployment
    - Added error handling and validation checks in deployment scripts
    - Included environment variable configuration for AWS region and account
    - Added deployment status reporting and rollback instructions"


- [ ] 17. Create integration tests
  - [ ] 17.1 Create TrainingPipelineIntegrationTest
    - Test end-to-end training workflow with mocked AWS services
    - Verify S3 orchestration between stages
    - Test error handling and retry logic
    - _Requirements: 16.2, 16.3_
  
  - [ ] 17.2 Create InferencePipelineIntegrationTest
    - Test end-to-end inference workflow with mocked AWS services
    - Verify S3 orchestration between stages
    - Test DynamoDB storage and SNS alerting
    - _Requirements: 16.2, 16.3_
  
  - [ ] 17.3 Git commit and push
    - **Git commit and push**: "test: create integration tests for end-to-end workflows
    
    - Created TrainingPipelineIntegrationTest validating complete training workflow
    - Tested S3 orchestration between DataPrep, Train, Evaluate, and Deploy stages
    - Validated error handling and retry logic for training pipeline
    - Created InferencePipelineIntegrationTest validating complete inference workflow
    - Tested S3 orchestration between Score, Store, Alert, and Monitor stages
    - Validated DynamoDB storage, SNS alerting, and drift detection
    - Used mocked AWS services for reliable test execution
    - Added assertions for workflow state transitions and data flow"

- [ ] 18. Create documentation
  - [ ] 18.1 Create README.md
    - Document system architecture
    - Add setup instructions
    - Include deployment guide
    - Add monitoring and troubleshooting sections
    - _Requirements: 17.1, 17.4, 17.5_
  
  - [ ] 18.2 Add code comments
    - Document CEAP integration patterns
    - Explain S3 orchestration conventions
    - Document SageMaker integration
    - _Requirements: 17.2_
  
  - [ ] 18.3 Create example configuration files
    - Add example workflow configurations
    - Add example Lambda handler configurations
    - Add example Glue job parameters
    - _Requirements: 17.3_
  
  - [ ] 18.4 Git commit and push
    - **Git commit and push**: "docs: create comprehensive project documentation
    
    - Created README.md with system architecture overview and component diagrams
    - Added detailed setup instructions including prerequisites and dependency installation
    - Included deployment guide for both training and inference pipelines
    - Documented monitoring and troubleshooting procedures
    - Added inline code comments explaining CEAP integration patterns
    - Documented S3 orchestration conventions and path structures
    - Explained SageMaker integration patterns for training and inference
    - Created example configuration files for workflows, Lambda handlers, and Glue jobs
    - Added operational runbooks for common scenarios"

- [ ] 19. Final checkpoint - Ensure all tests pass and documentation is complete
  - Ensure all tests pass, ask the user if questions arise.
  - **Git commit and push**: "chore: final checkpoint - fraud detection ML pipeline complete
    
    - Verified all unit tests, property tests, and integration tests pass
    - Confirmed all handlers implement required functionality correctly
    - Validated CDK infrastructure stacks synthesize without errors
    - Ensured documentation is complete and accurate
    - Project ready for deployment to AWS environment"

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflow execution

