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
import software.amazon.awscdk.services.stepfunctions.tasks.SageMakerCreateTrainingJob
import software.amazon.awscdk.services.stepfunctions.tasks.SageMakerCreateTrainingJobProps
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
    
    // Glue Job
    val dataPrepJob: CfnJob
    val glueRole: Role
    
    // IAM Roles
    val sageMakerExecutionRole: Role
    
    // Lambda Functions
    val trainHandler: Function
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
        
        // Grant permission to read Glue script from S3
        glueRole.addToPrincipalPolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("s3:GetObject", "s3:ListBucket"))
                .resources(listOf(
                    "arn:aws:s3:::fraud-detection-glue-scripts-$envName-${this.account}/*",
                    "arn:aws:s3:::fraud-detection-glue-scripts-$envName-${this.account}"
                ))
                .build()
        )
        
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
        
        // Grant Parameter Store read access for hyperparameters
        trainHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "ssm:GetParameter",
                    "ssm:GetParameters"
                ))
                .resources(listOf(
                    "arn:aws:ssm:*:*:parameter/fraud-detection/hyperparameters/*"
                ))
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
        
        // DataPrep Task (Glue Job)
        val dataPrepTask = GlueStartJobRun.Builder.create(this, "DataPrepTask")
            .glueJobName(dataPrepJob.name!!)
            .integrationPattern(software.amazon.awscdk.services.stepfunctions.IntegrationPattern.RUN_JOB)
            .arguments(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "--execution_id.$" to "$$.Execution.Name",
                "--workflow_bucket" to workflowBucket.bucketName,
                "--dataset_s3_path.$" to "$.datasetS3Path",
                "--output_prefix.$" to "$.outputPrefix",
                "--train_split.$" to "States.Format('{}', $.trainSplit)",
                "--validation_split.$" to "States.Format('{}', $.validationSplit)",
                "--test_split.$" to "States.Format('{}', $.testSplit)"
            )))
            .resultPath("$.dataPrepResult")
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
        
        // Train Task (Native SageMaker Integration - replaces Lambda)
        // Step Functions directly creates and waits for SageMaker training job
        val trainTask = software.amazon.awscdk.services.stepfunctions.tasks.SageMakerCreateTrainingJob.Builder.create(this, "TrainTask")
            .trainingJobName(software.amazon.awscdk.services.stepfunctions.JsonPath.stringAt("States.Format('fraud-detection-{}', $$.Execution.Name)"))
            .role(sageMakerExecutionRole)
            .algorithmSpecification(software.amazon.awscdk.services.stepfunctions.tasks.AlgorithmSpecification.builder()
                .trainingImage(software.amazon.awscdk.services.stepfunctions.tasks.DockerImage.fromRegistry("683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"))
                .trainingInputMode(software.amazon.awscdk.services.stepfunctions.tasks.InputMode.FILE)
                .build())
            .inputDataConfig(listOf(
                software.amazon.awscdk.services.stepfunctions.tasks.Channel.builder()
                    .channelName("train")
                    .dataSource(software.amazon.awscdk.services.stepfunctions.tasks.DataSource.builder()
                        .s3DataSource(software.amazon.awscdk.services.stepfunctions.tasks.S3DataSource.builder()
                            .s3Location(software.amazon.awscdk.services.stepfunctions.tasks.S3Location.fromJsonExpression("States.Format('{}train.parquet', $.outputPrefix)"))
                            .build())
                        .build())
                    .contentType("application/x-parquet")
                    .build(),
                software.amazon.awscdk.services.stepfunctions.tasks.Channel.builder()
                    .channelName("validation")
                    .dataSource(software.amazon.awscdk.services.stepfunctions.tasks.DataSource.builder()
                        .s3DataSource(software.amazon.awscdk.services.stepfunctions.tasks.S3DataSource.builder()
                            .s3Location(software.amazon.awscdk.services.stepfunctions.tasks.S3Location.fromJsonExpression("States.Format('{}validation.parquet', $.outputPrefix)"))
                            .build())
                        .build())
                    .contentType("application/x-parquet")
                    .build()
            ))
            .outputDataConfig(software.amazon.awscdk.services.stepfunctions.tasks.OutputDataConfig.builder()
                .s3OutputLocation(software.amazon.awscdk.services.stepfunctions.tasks.S3Location.fromBucket(modelsBucket, ""))
                .build())
            .resourceConfig(software.amazon.awscdk.services.stepfunctions.tasks.ResourceConfig.builder()
                .instanceCount(1)
                .instanceType(software.amazon.awscdk.services.ec2.InstanceType.of(
                    software.amazon.awscdk.services.ec2.InstanceClass.MEMORY5,
                    software.amazon.awscdk.services.ec2.InstanceSize.XLARGE
                ))
                .volumeSize(software.amazon.awscdk.Size.gibibytes(30))
                .build())
            .stoppingCondition(software.amazon.awscdk.services.stepfunctions.tasks.StoppingCondition.builder()
                .maxRuntime(Duration.hours(1))
                .build())
            .hyperparameters(mapOf(
                "objective" to "binary:logistic",
                "num_round" to "150",
                "max_depth" to "7",
                "eta" to "0.2",
                "subsample" to "0.8",
                "colsample_bytree" to "0.8"
            ))
            .integrationPattern(software.amazon.awscdk.services.stepfunctions.IntegrationPattern.RUN_JOB)
            .resultPath("$.trainingResult")
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
                Fail.Builder.create(this, "TrainFailed")
                    .cause("Training stage failed after retries")
                    .error("TrainError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Evaluate Task (Lambda)
        val evaluateTask = LambdaInvoke.Builder.create(this, "EvaluateTask")
            .lambdaFunction(evaluateHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "EvaluateStage",
                "workflowBucket" to workflowBucket.bucketName,
                "initialData" to mapOf(
                    "modelArtifactPath.$" to "$.trainingResult.ModelArtifacts.S3ModelArtifacts",
                    "testDataPath.$" to "States.Format('{}test.csv', $.outputPrefix)"
                )
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
        
        // Chain tasks: DataPrep → Train → Evaluate → Deploy → Success
        val definition = dataPrepTask
            .next(trainTask)
            .next(evaluateTask)
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
        
        // Grant SageMaker permissions to workflow execution role
        trainingWorkflow.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:AddTags",
                    "sagemaker:ListTags"
                ))
                .resources(listOf("*"))
                .build()
        )
        
        // Grant PassRole permission for SageMaker execution role
        trainingWorkflow.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("iam:PassRole"))
                .resources(listOf(sageMakerExecutionRole.roleArn))
                .build()
        )
        
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
