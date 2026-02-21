package com.frauddetection.infrastructure

import software.amazon.awscdk.CfnOutput
import software.amazon.awscdk.Duration
import software.amazon.awscdk.Fn
import software.amazon.awscdk.Stack
import software.amazon.awscdk.StackProps
import software.amazon.awscdk.services.dynamodb.Attribute
import software.amazon.awscdk.services.dynamodb.AttributeType
import software.amazon.awscdk.services.dynamodb.BillingMode
import software.amazon.awscdk.services.dynamodb.GlobalSecondaryIndexProps
import software.amazon.awscdk.services.dynamodb.Table
import software.amazon.awscdk.services.events.Rule
import software.amazon.awscdk.services.events.Schedule
import software.amazon.awscdk.services.events.targets.SfnStateMachine
import software.amazon.awscdk.services.iam.Effect
import software.amazon.awscdk.services.iam.PolicyStatement
import software.amazon.awscdk.services.lambda.Code
import software.amazon.awscdk.services.lambda.Function
import software.amazon.awscdk.services.lambda.Runtime
import software.amazon.awscdk.services.logs.RetentionDays
import software.amazon.awscdk.services.s3.Bucket
import software.amazon.awscdk.services.sns.Topic
import software.amazon.awscdk.services.stepfunctions.*
import software.amazon.awscdk.services.stepfunctions.tasks.LambdaInvoke
import software.constructs.Construct

/**
 * Inference Pipeline Stack - Daily transaction scoring workflow
 * 
 * This stack implements the fraud detection inference pipeline using AWS Step Functions (Express workflow).
 * The pipeline scores daily transactions for fraud risk, stores results in DynamoDB, and alerts on high-risk cases.
 * 
 * Workflow Stages:
 * 1. Score (Lambda): Score daily transaction batch using deployed SageMaker endpoint
 * 2. Store (Lambda): Store scored transactions in DynamoDB
 * 3. Alert (Lambda): Identify high-risk transactions and send alerts via SNS
 * 4. Monitor (Lambda): Monitor model performance and detect distribution drift
 * 
 * Resources:
 * - 4 Lambda Functions (Score, Store, Alert, Monitor handlers)
 * - 1 Step Functions State Machine (Express workflow)
 * - 1 DynamoDB Table (FraudScores with BatchDateIndex GSI)
 * - 2 SNS Topics (alerts and monitoring)
 * - 1 EventBridge Rule (daily execution schedule)
 * 
 * Dependencies:
 * - TrainingPipelineStack (for S3 buckets and endpoint configuration)
 * 
 * Requirements Validated:
 * - Requirement 1.5: Inference pipeline orchestrated via Step Functions
 * - Requirement 9.1: Daily scheduled execution
 * - Requirement 9.2: Express workflow type for fast, high-throughput operations
 * - Requirement 12.1: Automated infrastructure deployment
 * - Requirement 12.2: DynamoDB table with GSI
 * - Requirement 12.3: SNS topics for alerts and monitoring
 */
class InferencePipelineStack(
    scope: Construct,
    id: String,
    props: StackProps,
    val envName: String,
    val trainingStackName: String,
    val bucketSuffix: String = System.getenv("BUCKET_SUFFIX") ?: "default"
) : Stack(scope, id, props) {
    
    // DynamoDB Table
    val fraudScoresTable: Table
    
    // SNS Topics
    val alertTopic: Topic
    val monitoringTopic: Topic
    
    // Lambda Functions
    val scoreHandler: Function
    val storeHandler: Function
    val alertHandler: Function
    val monitorHandler: Function
    
    // Step Functions Workflow
    val inferenceWorkflow: StateMachine
    
    init {
        // Set stack description
        this.templateOptions.description = 
            "Fraud Detection Inference Pipeline Stack - Daily transaction scoring workflow " +
            "that scores transactions for fraud risk, stores results in DynamoDB, and alerts " +
            "on high-risk cases. Uses Express workflow for fast, high-throughput operations."
        
        // Import S3 bucket names from TrainingPipelineStack
        val workflowBucketName = Fn.importValue("$trainingStackName-WorkflowBucketName")
        val configBucketName = Fn.importValue("$trainingStackName-ConfigBucketName")
        val dataBucketName = Fn.importValue("$trainingStackName-DataBucketName")
        
        // ========================================
        // DynamoDB Table
        // ========================================
        
        // Create FraudScores table with GSI
        fraudScoresTable = Table.Builder.create(this, "FraudScoresTable")
            .tableName("FraudScores-$envName")
            .partitionKey(Attribute.builder()
                .name("transactionId")
                .type(AttributeType.STRING)
                .build())
            .sortKey(Attribute.builder()
                .name("timestamp")
                .type(AttributeType.NUMBER)
                .build())
            .billingMode(BillingMode.PAY_PER_REQUEST)
            .build()
        
        // Add Global Secondary Index for querying by batch date
        fraudScoresTable.addGlobalSecondaryIndex(
            GlobalSecondaryIndexProps.builder()
                .indexName("BatchDateIndex")
                .partitionKey(Attribute.builder()
                    .name("batchDate")
                    .type(AttributeType.STRING)
                    .build())
                .sortKey(Attribute.builder()
                    .name("fraudScore")
                    .type(AttributeType.NUMBER)
                    .build())
                .build()
        )
        
        // ========================================
        // SNS Topics
        // ========================================
        
        // Alert topic for high-risk transactions
        alertTopic = Topic.Builder.create(this, "AlertTopic")
            .topicName("fraud-detection-alerts-$envName")
            .displayName("Fraud Detection Alerts")
            .build()
        
        // Monitoring topic for drift detection
        monitoringTopic = Topic.Builder.create(this, "MonitoringTopic")
            .topicName("fraud-detection-monitoring-$envName")
            .displayName("Fraud Detection Monitoring")
            .build()
        
        // ========================================
        // Lambda Functions
        // ========================================
        
        // Score Handler Lambda
        scoreHandler = Function.Builder.create(this, "ScoreHandler")
            .functionName("fraud-detection-score-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.inference.ScoreHandler::handleRequest")
            .code(Code.fromAsset("../fraud-inference-pipeline/build/libs/fraud-inference-pipeline.jar"))
            .memorySize(1024)
            .timeout(Duration.minutes(5))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucketName,
                "CONFIG_BUCKET" to configBucketName,
                "DATA_BUCKET" to dataBucketName,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Score Handler
        scoreHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject"
                ))
                .resources(listOf(
                    "arn:aws:s3:::$workflowBucketName/*",
                    "arn:aws:s3:::$configBucketName/*",
                    "arn:aws:s3:::$dataBucketName/*"
                ))
                .build()
        )
        
        // Grant s3:ListBucket on bucket-level ARNs (required by S3 SDK for object existence checks)
        scoreHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("s3:ListBucket"))
                .resources(listOf(
                    "arn:aws:s3:::$workflowBucketName",
                    "arn:aws:s3:::$configBucketName",
                    "arn:aws:s3:::$dataBucketName"
                ))
                .build()
        )
        
        scoreHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf("sagemaker:InvokeEndpoint"))
                .resources(listOf("*"))
                .build()
        )
        
        // Store Handler Lambda
        storeHandler = Function.Builder.create(this, "StoreHandler")
            .functionName("fraud-detection-store-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.inference.StoreHandler::handleRequest")
            .code(Code.fromAsset("../fraud-inference-pipeline/build/libs/fraud-inference-pipeline.jar"))
            .memorySize(512)
            .timeout(Duration.minutes(5))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucketName,
                "DYNAMODB_TABLE" to fraudScoresTable.tableName,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Store Handler
        storeHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject"
                ))
                .resources(listOf("arn:aws:s3:::$workflowBucketName/*"))
                .build()
        )
        
        fraudScoresTable.grantWriteData(storeHandler)
        
        // Alert Handler Lambda
        alertHandler = Function.Builder.create(this, "AlertHandler")
            .functionName("fraud-detection-alert-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.inference.AlertHandler::handleRequest")
            .code(Code.fromAsset("../fraud-inference-pipeline/build/libs/fraud-inference-pipeline.jar"))
            .memorySize(512)
            .timeout(Duration.minutes(2))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucketName,
                "FRAUD_ALERT_TOPIC_ARN" to alertTopic.topicArn,
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Alert Handler
        alertHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject"
                ))
                .resources(listOf("arn:aws:s3:::$workflowBucketName/*"))
                .build()
        )
        
        alertTopic.grantPublish(alertHandler)
        
        // Monitor Handler Lambda
        monitorHandler = Function.Builder.create(this, "MonitorHandler")
            .functionName("fraud-detection-monitor-$envName")
            .runtime(Runtime.JAVA_17)
            .handler("com.fraud.inference.MonitorHandler::handleRequest")
            .code(Code.fromAsset("../fraud-inference-pipeline/build/libs/fraud-inference-pipeline.jar"))
            .memorySize(512)
            .timeout(Duration.minutes(2))
            .environment(mapOf(
                "ENVIRONMENT" to envName,
                "WORKFLOW_BUCKET" to workflowBucketName,
                "MONITORING_ALERT_TOPIC_ARN" to monitoringTopic.topicArn,
                "METRICS_BUCKET" to "fraud-detection-metrics",
                "LOG_LEVEL" to "INFO"
            ))
            .logRetention(RetentionDays.ONE_MONTH)
            .build()
        
        // Grant permissions to Monitor Handler
        monitorHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject"
                ))
                .resources(listOf("arn:aws:s3:::$workflowBucketName/*"))
                .build()
        )
        
        // Grant permissions to Monitor Handler for metrics bucket
        monitorHandler.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket"
                ))
                .resources(listOf(
                    "arn:aws:s3:::fraud-detection-metrics",
                    "arn:aws:s3:::fraud-detection-metrics/*"
                ))
                .build()
        )
        
        monitoringTopic.grantPublish(monitorHandler)
        
        // ========================================
        // Step Functions Workflow
        // ========================================
        
        // Score Task (Lambda)
        val scoreTask = LambdaInvoke.Builder.create(this, "ScoreTask")
            .lambdaFunction(scoreHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "ScoreStage",
                "workflowBucket" to workflowBucketName,
                "initialData.$" to "$"
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(10))
                    .maxAttempts(3)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "ScoreFailed")
                    .cause("Scoring stage failed after retries")
                    .error("ScoreError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Store Task (Lambda)
        val storeTask = LambdaInvoke.Builder.create(this, "StoreTask")
            .lambdaFunction(storeHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to workflowBucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(10))
                    .maxAttempts(3)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "StoreFailed")
                    .cause("Storage stage failed after retries")
                    .error("StoreError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Alert Task (Lambda)
        val alertTask = LambdaInvoke.Builder.create(this, "AlertTask")
            .lambdaFunction(alertHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to workflowBucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(10))
                    .maxAttempts(3)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "AlertFailed")
                    .cause("Alert stage failed after retries")
                    .error("AlertError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Monitor Task (Lambda)
        val monitorTask = LambdaInvoke.Builder.create(this, "MonitorTask")
            .lambdaFunction(monitorHandler)
            .payload(software.amazon.awscdk.services.stepfunctions.TaskInput.fromObject(mapOf(
                "executionId.$" to "$$.Execution.Name",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to workflowBucketName
            )))
            .outputPath("$.Payload")
            .retryOnServiceExceptions(true)
            .build()
            .addRetry(
                RetryProps.builder()
                    .errors(listOf("States.TaskFailed", "States.Timeout", "Lambda.ServiceException"))
                    .interval(Duration.seconds(10))
                    .maxAttempts(3)
                    .backoffRate(2.0)
                    .build()
            )
            .addCatch(
                Fail.Builder.create(this, "MonitorFailed")
                    .cause("Monitoring stage failed after retries")
                    .error("MonitorError")
                    .build(),
                CatchProps.builder()
                    .errors(listOf("States.ALL"))
                    .resultPath("$.error")
                    .build()
            )
        
        // Success state
        val workflowSuccess = Succeed.Builder.create(this, "InferenceSuccess")
            .comment("Inference workflow completed successfully")
            .build()
        
        // Chain tasks: Score → Store → Alert → Monitor → Success
        val definition = scoreTask
            .next(storeTask)
            .next(alertTask)
            .next(monitorTask)
            .next(workflowSuccess)
        
        // Create State Machine (Express workflow)
        inferenceWorkflow = StateMachine.Builder.create(this, "InferenceWorkflow")
            .stateMachineName("FraudDetectionInference-$envName")
            .stateMachineType(StateMachineType.EXPRESS)
            .definitionBody(DefinitionBody.fromChainable(definition))
            .timeout(Duration.minutes(30))
            .logs(software.amazon.awscdk.services.stepfunctions.LogOptions.builder()
                .destination(software.amazon.awscdk.services.logs.LogGroup.Builder.create(this, "InferenceWorkflowLogGroup")
                    .logGroupName("/aws/stepfunctions/FraudDetectionInference-$envName")
                    .retention(RetentionDays.ONE_WEEK)
                    .build())
                .level(software.amazon.awscdk.services.stepfunctions.LogLevel.ALL)
                .includeExecutionData(true)
                .build())
            .build()
        
        // Grant S3 permissions to workflow execution role
        inferenceWorkflow.addToRolePolicy(
            PolicyStatement.Builder.create()
                .effect(Effect.ALLOW)
                .actions(listOf(
                    "s3:GetObject",
                    "s3:PutObject"
                ))
                .resources(listOf("arn:aws:s3:::$workflowBucketName/*"))
                .build()
        )
        
        // ========================================
        // EventBridge Schedule
        // ========================================
        
        // Schedule inference workflow (daily at 1 AM UTC)
        val inferenceSchedule = Rule.Builder.create(this, "InferenceSchedule")
            .ruleName("FraudDetectionInference-$envName")
            .schedule(Schedule.cron(
                software.amazon.awscdk.services.events.CronOptions.builder()
                    .minute("0")
                    .hour("1")
                    .build()
            ))
            .targets(listOf(SfnStateMachine(inferenceWorkflow)))
            .build()
        
        // ========================================
        // Stack Outputs
        // ========================================
        
        CfnOutput.Builder.create(this, "InferenceWorkflowArnOutput")
            .value(inferenceWorkflow.stateMachineArn)
            .exportName("$stackName-InferenceWorkflowArn")
            .description("ARN of the inference Step Functions workflow")
            .build()
        
        CfnOutput.Builder.create(this, "FraudScoresTableNameOutput")
            .value(fraudScoresTable.tableName)
            .exportName("$stackName-FraudScoresTableName")
            .description("Name of the DynamoDB table for fraud scores")
            .build()
        
        CfnOutput.Builder.create(this, "AlertTopicArnOutput")
            .value(alertTopic.topicArn)
            .exportName("$stackName-AlertTopicArn")
            .description("ARN of the SNS topic for fraud alerts")
            .build()
        
        CfnOutput.Builder.create(this, "MonitoringTopicArnOutput")
            .value(monitoringTopic.topicArn)
            .exportName("$stackName-MonitoringTopicArn")
            .description("ARN of the SNS topic for monitoring alerts")
            .build()
    }
}
