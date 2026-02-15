package com.frauddetection.infrastructure

import software.amazon.awscdk.App
import software.amazon.awscdk.Environment
import software.amazon.awscdk.StackProps

/**
 * CDK Application for Fraud Detection ML Pipeline
 * 
 * This application creates two stacks:
 * 1. TrainingPipelineStack - Weekly model training workflow
 * 2. InferencePipelineStack - Daily fraud scoring workflow
 * 
 * Configuration via CDK context parameters:
 * - envName: Environment name (dev, staging, prod)
 * - bucketSuffix: Unique suffix for S3 bucket names (e.g., "quannh0308-20260214")
 * - awsAccountId: AWS account ID
 * - awsRegion: AWS region
 * - trainingStackName: Name of the training stack (for inference stack to import outputs)
 */
fun main() {
    val app = App()
    
    // Read context parameters
    val envName = app.node.tryGetContext("envName") as? String ?: "dev"
    val bucketSuffix = app.node.tryGetContext("bucketSuffix") as? String 
        ?: System.getenv("BUCKET_SUFFIX") 
        ?: "quannh0308-20260214"
    val awsAccountId = app.node.tryGetContext("awsAccountId") as? String 
        ?: System.getenv("AWS_ACCOUNT_ID")
        ?: throw IllegalArgumentException("AWS_ACCOUNT_ID must be provided via context or environment variable")
    val awsRegion = app.node.tryGetContext("awsRegion") as? String 
        ?: System.getenv("AWS_REGION")
        ?: "us-east-1"
    
    val env = Environment.builder()
        .account(awsAccountId)
        .region(awsRegion)
        .build()
    
    // Create Training Pipeline Stack
    val trainingStackName = "FraudDetectionTrainingPipeline-$envName"
    val trainingStack = TrainingPipelineStack(
        app,
        trainingStackName,
        StackProps.builder()
            .env(env)
            .description("Fraud Detection Training Pipeline - Weekly ML model training workflow")
            .build(),
        envName = envName,
        bucketSuffix = bucketSuffix
    )
    
    // Create Inference Pipeline Stack
    val inferenceStackName = "FraudDetectionInferencePipeline-$envName"
    val inferenceStack = InferencePipelineStack(
        app,
        inferenceStackName,
        StackProps.builder()
            .env(env)
            .description("Fraud Detection Inference Pipeline - Daily fraud scoring workflow")
            .build(),
        envName = envName,
        trainingStackName = trainingStackName,
        bucketSuffix = bucketSuffix
    )
    
    // Inference stack depends on training stack (needs to import bucket names)
    inferenceStack.node.addDependency(trainingStack)
    
    app.synth()
}
