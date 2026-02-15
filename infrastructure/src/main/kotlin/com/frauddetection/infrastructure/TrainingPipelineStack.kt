package com.frauddetection.infrastructure

import software.amazon.awscdk.CfnOutput
import software.amazon.awscdk.Duration
import software.amazon.awscdk.Stack
import software.amazon.awscdk.StackProps
import software.amazon.awscdk.services.events.Rule
import software.amazon.awscdk.services.events.Schedule
import software.amazon.awscdk.services.events.targets.SfnStateMachine
import software.amazon.awscdk.services.glue.CfnJob
import software.amazon.awscdk.services.iam.Effect
import software.amazon.awscdk.services.iam.PolicyStatement
import software.amazon.awscdk.services.iam.Role
import software.amazon.awscdk.services.iam.ServicePrincipal
import software.amazon.awscdk.services.iam.ManagedPolicy
import software.amazon.awscdk.services.lambda.Code
import software.amazon.awscdk.services.lambda.Function
import software.amazon.awscdk.services.lambda.Runtime
import software.amazon.awscdk.services.logs.RetentionDays
import software.amazon.awscdk.services.s3.BlockPublicAccess
import software.amazon.awscdk.services.s3.Bucket
import software.amazon.awscdk.services.s3.BucketEncryption
import software.amazon.awscdk.services.s3.LifecycleRule
import software.amazon.awscdk.services.stepfunctions.*
import software.amazon.awscdk.services.stepfunctions.tasks.GlueStartJobRun
import software.amazon.awscdk.services.stepfunctions.tasks.LambdaInvoke
import software.constructs.Construct

/**
 * Training Pipeline Stack - Weekly ML model training workflow
 * 
 * This stack implements the fraud detection training pipeline using AWS Step Functions (Standard workflow).
 * The pipeline trains fraud detection models on historical transaction data and deploys them to SageMaker endpoints.
 * 
 * Workflow Stages:
 * 1. DataPrep (Glue): Prepare historical transaction data for model training
 * 2. Train (Lambda): Configure and launch SageMaker training job
 * 3. Evaluate (Lambda): Evaluate trained model on test dataset
 * 4. Deploy (Lambda): Deploy trained model to production SageMaker endpoint
 * 
 * Resources:
 * - 1 Glue Job (data-prep.py script)
 * - 3 Lambda Functions (Train, Evaluate, Deploy handlers)
 * - 1 Step Functions State Machine (Standard workflow)
 * - 4 S3 Buckets (workflow, data, models, config)
 * - 1 EventBridge Rule (weekly execution schedule)
 * 
 * Requirements Validated:
 * - Requirement 1.5: Training pipeline orchestrated via Step Functions
 * - Requirement 5.1: Weekly scheduled execution
 * - Requirement 5.2: Standard workflow type for long-duration operations
 * - Requirement 12.1: Automated infrastructure deployment
 * - Requirement 12.2: S3 buckets with appropriate lifecycle policies
 */
class TrainingPipelineStack(
    scope: Construct,
    id: String,
    props: StackProps,
    val envName: String,
    val bucketSuffix: String = System.getenv("BUCKET_SUFFIX") ?: "default"
) : Stack(scope, id, props) {
    
    // S3 Buckets
    val workflowBucket: Bucket
    val dataBucket: software.amazon.awscdk.services.s3.IBucket
    val modelsBucket: Bucket
    val configBucket: Bucket
    
    // Glue Job (deprecated - replaced by Lambda)
    val dataPrepJob: CfnJob
    val glueRole: Role
    
    // IAM Roles
    val sageMakerExecutionRole: Role
    
    // Lambda Functions
    val dataPrepHandler: Function  // New: replaces Glue job
    val trainHandler: Function
    val checkTrainingStatusHandler: Function  // New: polls training job status
    val evaluateHandler: Function
    val deployHandler: Function
    
    // Step Functions Workflow
    val trainingWorkflow: StateMachine
    
    init {
        // Set stack description
        this.templateOptions.description = 
            "Fraud Detection Training Pipeline Stack - Weekly ML model training workflow " +
            "that trains fraud detection models on historical transaction data and deploys " +
            "them to SageMaker endpoints. Uses Standard workflow for long-duration operations."
        
        // ========================================
        // S3 Buckets
        // ========================================
        
        // Workflow bucket for Step Functions intermediate storage
        workflowBucket = Bucket.Builder.create(this, "WorkflowBucket")
            .bucketName("fraud-detection-workflow-$bucketSuffix")
            .versioned(false)
            .encryption(BucketEncryption.S3_MANAGED)
            .blockPublicAccess(BlockPublicAccess.BLOCK_ALL)
            .lifecycleRules(listOf(
                LifecycleRule.builder()
                    .id("DeleteOldExecutions")
                    .prefix("executions/")
                    .expiration(Duration.days(7))
                    .enabled(true)
                    .build()
            ))
            .build()
        
        // Data bucket for raw and prepared datasets
        // Import existing bucket instead of creating new one
        dataBucket = Bucket.fromBucketName(
            this,
            "DataBucket",
            "fraud-detection-data-$bucketSuffix"
        )
        
        // Models bucket for trained model artifacts
        modelsBucket = Bucket.Builder.create(this, "ModelsBucket")
            .bucketName("fraud-detection-models-$bucketSuffix")
            .versioned(true)
            .encryption(BucketEncryption.S3_MANAGED)
            .blockPublicAccess(BlockPublicAccess.BLOCK_ALL)
            .lifecycleRules(listOf(
                LifecycleRule.builder()
                    .id("DeleteOldModels")
                    .noncurrentVersionExpiration(Duration.days(90))
                    .enabled(true)
                    .build()
            ))
            .build()
        
        // Config bucket for endpoint metadata
        configBucket = Bucket.Builder.create(this, "ConfigBucket")
            .bucketName("fraud-detection-config-$bucketSuffix")
            .versioned(true)
            .encryption(BucketEncryption.S3_MANAGED)
            .blockPublicAccess(BlockPublicAccess.BLOCK_ALL)
            .build()
        
        // ========================================
        // Glue Job for Data Preparation
        // ========================================
        
        // Create IAM role for Glue job
        glueRole = Role.Builder.create(this, "GlueRole")
            .roleName("fraud-detection-glue-role-$envName")
            .assumedBy(ServicePrincipal("glue.amazonaws.com"))
            .managedPolicies(listOf(
                ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSGlueServiceRole")
            ))
            .build()
        
        // Grant S3 permissions to Glue role
        dataBucket.grantReadWrite(glueRole)
        workflowBucket.grantReadWrite(glueRole)
        
        // Create Glue job for data preparation
        dataPrepJob = CfnJob.Builder.create(this, "DataPrepJob")
            .name("fraud-detection-data-prep-$envName")
            .role(glueRole.roleArn)
            .command(CfnJob.JobCommandProperty.builder()
                .name("glueetl")
                .scriptLocation("s3://fraud-detection-glue-scripts-$envName-${this.account}/data-prep.py")
                .pythonVersion("3")
                .build())
            .glueVersion("4.0")
            .maxRetries(2)
            .timeout(30) // 30 minutes
            .numberOfWorkers(5)
            .workerType("G.1X")
            .defaultArguments(mapOf(
                "--job-language" to "python",
                "--enable-metrics" to "true",
                "--enable-continuous-cloudwatch-log" to "true",
                "--enable-spark-ui" to "true"
            ))
            .build()
        
        // ========================================
        // IAM Roles
        // ========================================
        
        // SageMaker Execution Role
        sageMakerExecutionRole = Role.Builder.create(this, "SageMakerExecutionRole")
            .roleName("fraud-detection-sagemaker-role-$envName")
            .assumedBy(ServicePrincipal("sagemaker.amazonaws.com"))
            .managedPolicies(listOf(
                ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess")
            ))
            .build()
        
        // Grant S3 permissions to SageMaker role
        dataBucket.grantReadWrite(sageMakerExecutionRole)
        modelsBucket.grantReadWrite(sageMakerExecutionRole)
        
        // ========================================
        // Lambda Functions
        // ========================================
        
        // DataPrep Handler Lambda (replaces Glue job)
        dataPrepHandler = Function.Builder.create(this, "DataPrepHandler")
            .functionName("fraud-detection-dataprep-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.training.DataPrepHandler::handleRequest")
            .code(Code.fromAsset("../fraud-training-pipeline/build/libs/fraud-training-pipeline.jar"))
            .memorySize(3008) // Max Lambda memory (3 GB)
            .timeout(Duration.minutes(15)) // Max Lambda timeout
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucket.bucketName,
                "DATA_BUCKET" to dataBucket.bucketName,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to DataPrep Handler
        dataBucket.grantRead(dataPrepHandler)
        dataBucket.grantWrite(dataPrepHandler)
        workflowBucket.grantReadWrite(dataPrepHandler)
        
        // Train Handler Lambda
        trainHandler = Function.Builder.create(this, "TrainHandler")
            .functionName("fraud-detection-train-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.training.TrainHandler::handleRequest")
            .code(Code.fromAsset("../fraud-training-pipeline/build/libs/fraud-training-pipeline.jar"))
            .memorySize(1024)
            .timeout(Duration.minutes(15))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucket.bucketName,
                "DATA_BUCKET" to dataBucket.bucketName,
                "MODELS_BUCKET" to modelsBucket.bucketName,
                "SAGEMAKER_EXECUTION_ROLE_ARN" to sageMakerExecutionRole.roleArn,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Train Handler
        workflowBucket.grantReadWrite(trainHandler)
        dataBucket.grantRead(trainHandler)
        modelsBucket.grantReadWrite(trainHandler)
        
        trainHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob"
                ))
                .resources(listOf("*"))
                .build()
        )
        
        trainHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("iam:PassRole"))
                .resources(listOf(sageMakerExecutionRole.roleArn))
                .build()
        )
        
        // CheckTrainingStatus Handler Lambda (polls training job status)
        checkTrainingStatusHandler = Function.Builder.create(this, "CheckTrainingStatusHandler")
            .functionName("fraud-detection-check-training-status-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.training.CheckTrainingStatusHandler::handleRequest")
            .code(Code.fromAsset("../fraud-training-pipeline/build/libs/fraud-training-pipeline.jar"))
            .memorySize(256)
            .timeout(Duration.seconds(30))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucket.bucketName,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to CheckTrainingStatus Handler
        workflowBucket.grantReadWrite(checkTrainingStatusHandler)
        
        checkTrainingStatusHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "sagemaker:DescribeTrainingJob"
                ))
                .resources(listOf("*"))
                .build()
        )
        
        // Evaluate Handler Lambda
        evaluateHandler = Function.Builder.create(this, "EvaluateHandler")
            .functionName("fraud-detection-evaluate-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.training.EvaluateHandler::handleRequest")
            .code(Code.fromAsset("../fraud-training-pipeline/build/libs/fraud-training-pipeline.jar"))
            .memorySize(1024)
            .timeout(Duration.minutes(15))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucket.bucketName,
                "DATA_BUCKET" to dataBucket.bucketName,
                "SAGEMAKER_EXECUTION_ROLE_ARN" to sageMakerExecutionRole.roleArn,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Evaluate Handler
        workflowBucket.grantReadWrite(evaluateHandler)
        dataBucket.grantRead(evaluateHandler)
        modelsBucket.grantRead(evaluateHandler)
        
        evaluateHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteModel",
                    "sagemaker:InvokeEndpoint"
                ))
                .resources(listOf("*"))
                .build()
        )
        
        evaluateHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("iam:PassRole"))
                .resources(listOf(sageMakerExecutionRole.roleArn))
                .build()
        )
        
        // Deploy Handler Lambda
        deployHandler = Function.Builder.create(this, "DeployHandler")
            .functionName("fraud-detection-deploy-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.training.DeployHandler::handleRequest")
            .code(Code.fromAsset("../fraud-training-pipeline/build/libs/fraud-training-pipeline.jar"))
            .memorySize(512)
            .timeout(Duration.minutes(15))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucket.bucketName,
                "CONFIG_BUCKET" to configBucket.bucketName,
                "SAGEMAKER_EXECUTION_ROLE_ARN" to sageMakerExecutionRole.roleArn,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Deploy Handler
        workflowBucket.grantReadWrite(deployHandler)
        configBucket.grantReadWrite(deployHandler)
        modelsBucket.grantRead(deployHandler)
        
        deployHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:InvokeEndpoint"
                ))
                .resources(listOf("*"))
                .build()
        )
        
        deployHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("iam:PassRole"))
                .resources(listOf(sageMakerExecutionRole.roleArn))
                .build()
        )
        
        // ========================================
        // Step Functions Workflow
        // ========================================
        
        // DataPrep Task (Lambda - replaces Glue job)
        val dataPrepTask = LambdaInvoke.Builder.create(this, "DataPrepTask")
            .lambdaFunction(dataPrepHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "DataPrepStage",
                "workflowBucket" to workflowBucket.bucketName,
                "initialData.$" to "$"
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout"))
                    .interval(Duration.seconds(30))
                    .maxAttempts(2)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "DataPrepFailed")
                    .cause("Data preparation stage failed after retries")
                    .error("DataPrepError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Train Task (Lambda)
        val trainTask = LambdaInvoke.Builder.create(this, "TrainTask")
            .lambdaFunction(trainHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "TrainStage",
                "previousStage" to "DataPrepStage",
                "workflowBucket" to workflowBucket.bucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(30))
                    .maxAttempts(2)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "TrainFailed")
                    .cause("Training stage failed after retries")
                    .error("TrainError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Wait 5 minutes before checking training status
        val waitForTraining = software.amazon.awscdk.services.stepfunctions.Wait.Builder.create(this, "WaitForTraining")
            .time(software.amazon.awscdk.services.stepfunctions.WaitTime.duration(Duration.minutes(5)))
            .comment("Wait 5 minutes before checking training job status")
            .build()
        
        // Check Training Status Task
        val checkTrainingStatusTask = LambdaInvoke.Builder.create(this, "CheckTrainingStatusTask")
            .lambdaFunction(checkTrainingStatusHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "CheckTrainingStatusStage",
                "previousStage" to "TrainStage",
                "workflowBucket" to workflowBucket.bucketName,
                "initialData.$" to "$"
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
        
        // Evaluate Task (Lambda)
        val evaluateTask = LambdaInvoke.Builder.create(this, "EvaluateTask")
            .lambdaFunction(evaluateHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "EvaluateStage",
                "previousStage" to "TrainStage",
                "workflowBucket" to workflowBucket.bucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(30))
                    .maxAttempts(2)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "EvaluateFailed")
                    .cause("Evaluation stage failed after retries")
                    .error("EvaluateError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Deploy Task (Lambda)
        val deployTask = LambdaInvoke.Builder.create(this, "DeployTask")
            .lambdaFunction(deployHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "DeployStage",
                "previousStage" to "EvaluateStage",
                "workflowBucket" to workflowBucket.bucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(30))
                    .maxAttempts(2)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "DeployFailed")
                    .cause("Deployment stage failed after retries")
                    .error("DeployError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Success state
        val workflowSuccess = Succeed.Builder.create(this, "TrainingSuccess")
            .comment("Training workflow completed successfully")
            .build()
        
        // Training failed state
        val trainingJobFailed = Fail.Builder.create(this, "TrainingJobFailed")
            .cause("SageMaker training job failed")
            .error("TrainingJobError")
            .build()
        
        // Connect wait loop: Wait → CheckStatus → Choice
        waitForTraining.next(checkTrainingStatusTask)
        
        // Choice state to check if training is complete
        val isTrainingComplete = software.amazon.awscdk.services.stepfunctions.Choice.Builder.create(this, "IsTrainingComplete")
            .comment("Check if training job is complete or failed")
            .build()
        
        checkTrainingStatusTask.next(isTrainingComplete)
        
        isTrainingComplete
            .when(
                software.amazon.awscdk.services.stepfunctions.Condition.booleanEquals("$.isComplete", true),
                evaluateTask
            )
            .when(
                software.amazon.awscdk.services.stepfunctions.Condition.booleanEquals("$.isFailed", true),
                trainingJobFailed
            )
            .otherwise(waitForTraining)
        
        // Chain tasks: DataPrep → Train → Wait → CheckStatus → Choice → [Evaluate → Deploy → Success | Fail | Wait]
        val definition = dataPrepTask
            .next(trainTask)
            .next(waitForTraining)
        
        evaluateTask
            .next(deployTask)
            .next(workflowSuccess)
        
        // Create State Machine (Standard workflow)
        trainingWorkflow = StateMachine.Builder.create(this, "TrainingWorkflow")
            .stateMachineName("FraudDetectionTraining-$envName")
            .stateMachineType(StateMachineType.STANDARD)
            .definitionBody(DefinitionBody.fromChainable(definition))
            .timeout(Duration.hours(4))
            .build()
        
        // Grant S3 permissions to workflow execution role
        workflowBucket.grantReadWrite(trainingWorkflow)
        
        // ========================================
        // EventBridge Schedule
        // ========================================
        
        // Schedule training workflow (weekly on Sunday at 2 AM UTC)
        val trainingSchedule = Rule.Builder.create(this, "TrainingSchedule")
            .ruleName("FraudDetectionTraining-$envName")
            .schedule(Schedule.cron(
                software.amazon.awscdk.services.events.CronOptions.builder()
                    .minute("0")
                    .hour("2")
                    .weekDay("SUN")
                    .build()
            ))
            .targets(listOf(SfnStateMachine(trainingWorkflow)))
            .build()
        
        // ========================================
        // Stack Outputs
        // ========================================
        
        CfnOutput.Builder.create(this, "TrainingWorkflowArnOutput")
            .value(trainingWorkflow.stateMachineArn)
            .exportName("$stackName-TrainingWorkflowArn")
            .description("ARN of the training Step Functions workflow")
            .build()
        
        CfnOutput.Builder.create(this, "WorkflowBucketNameOutput")
            .value(workflowBucket.bucketName)
            .exportName("$stackName-WorkflowBucketName")
            .description("Name of the S3 bucket for workflow intermediate storage")
            .build()
        
        CfnOutput.Builder.create(this, "DataBucketNameOutput")
            .value(dataBucket.bucketName)
            .exportName("$stackName-DataBucketName")
            .description("Name of the S3 bucket for data storage")
            .build()
        
        CfnOutput.Builder.create(this, "ModelsBucketNameOutput")
            .value(modelsBucket.bucketName)
            .exportName("$stackName-ModelsBucketName")
            .description("Name of the S3 bucket for model artifacts")
            .build()
        
        CfnOutput.Builder.create(this, "ConfigBucketNameOutput")
            .value(configBucket.bucketName)
            .exportName("$stackName-ConfigBucketName")
            .description("Name of the S3 bucket for endpoint configuration")
            .build()
    }
}
