package com.fraud.training

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemaker.waiters.SageMakerWaiter
import software.amazon.awssdk.services.ssm.SsmClient
import software.amazon.awssdk.services.ssm.model.GetParameterRequest
import software.amazon.awssdk.services.ssm.model.GetParameterResponse
import software.amazon.awssdk.services.ssm.model.Parameter

/**
 * Test wrapper for TrainHandler that exposes processData for testing.
 */
class TestableTrainHandler(
    sageMakerClient: SageMakerClient,
    ssmClient: SsmClient,
    sageMakerExecutionRoleArn: String
) : TrainHandler(sageMakerClient, ssmClient, sageMakerExecutionRoleArn) {
    public override fun processData(input: JsonNode): JsonNode {
        return super.processData(input)
    }
}

/**
 * Unit tests for TrainHandler.
 * 
 * Tests cover:
 * - Training job configuration validation
 * - Training job creation with mocked SageMaker client
 * - Error handling for training failures
 * 
 * Requirements tested:
 * - Requirement 3.2: Configure SageMaker training job with appropriate instance type and hyperparameters
 * - Requirement 3.3: Train the Fraud_Detector using training and validation datasets
 */
class TrainHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // Mock SSM client for all tests
    val mockSsmClient = mockk<SsmClient>()
    
    beforeTest {
        every { mockSsmClient.getParameter(any<GetParameterRequest>()) } answers {
            val request = firstArg<GetParameterRequest>()
            val paramName = request.name().substringAfterLast("/")
            val value = when (paramName) {
                "objective" -> "binary:logistic"
                "num_round" -> "100"
                "max_depth" -> "5"
                "eta" -> "0.2"
                "subsample" -> "0.8"
                "colsample_bytree" -> "0.8"
                else -> "unknown"
            }
            GetParameterResponse.builder()
                .parameter(Parameter.builder().name(request.name()).value(value).build())
                .build()
        }
    }
    
    test("processData should configure training job with correct parameters") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        // Capture the training job request
        val capturedRequest = slot<CreateTrainingJobRequest>()
        
        every { mockSageMakerClient.createTrainingJob(capture(capturedRequest)) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } returns mockk()
        every { 
            mockSageMakerClient.describeTrainingJob(any<DescribeTrainingJobRequest>()) 
        } returns DescribeTrainingJobResponse.builder()
            .trainingJobStatus(TrainingJobStatus.COMPLETED)
            .modelArtifacts(ModelArtifacts.builder()
                .s3ModelArtifacts("s3://fraud-detection-models/fraud-detection-123/output/model.tar.gz")
                .build())
            .build()
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify training job configuration
        val request = capturedRequest.captured
        
        // Verify algorithm specification
        request.algorithmSpecification().trainingImage() shouldContain "xgboost"
        request.algorithmSpecification().trainingInputMode() shouldBe TrainingInputMode.FILE
        
        // Verify role ARN
        request.roleArn() shouldBe "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        // Verify input data configuration
        request.inputDataConfig().size shouldBe 2
        val trainChannel = request.inputDataConfig().find { it.channelName() == "train" }
        trainChannel shouldNotBe null
        trainChannel!!.dataSource().s3DataSource().s3Uri() shouldBe "s3://fraud-detection-data/prepared/train.parquet"
        trainChannel.contentType() shouldBe "application/x-parquet"
        
        val validationChannel = request.inputDataConfig().find { it.channelName() == "validation" }
        validationChannel shouldNotBe null
        validationChannel!!.dataSource().s3DataSource().s3Uri() shouldBe "s3://fraud-detection-data/prepared/validation.parquet"
        validationChannel.contentType() shouldBe "application/x-parquet"
        
        // Verify output configuration
        request.outputDataConfig().s3OutputPath() shouldBe "s3://fraud-detection-models/"
        
        // Verify resource configuration
        request.resourceConfig().instanceType() shouldBe TrainingInstanceType.ML_M5_XLARGE
        request.resourceConfig().instanceCount() shouldBe 1
        request.resourceConfig().volumeSizeInGB() shouldBe 30
        
        // Verify stopping condition
        request.stoppingCondition().maxRuntimeInSeconds() shouldBe 3600
        
        // Verify hyperparameters
        val hyperParams = request.hyperParameters()
        hyperParams["objective"] shouldBe "binary:logistic"
        hyperParams["num_round"] shouldBe "100"
        hyperParams["max_depth"] shouldBe "5"
        hyperParams["eta"] shouldBe "0.2"
        hyperParams["subsample"] shouldBe "0.8"
        hyperParams["colsample_bytree"] shouldBe "0.8"
        
        // Verify result
        result.get("trainingJobName").asText() shouldNotBe null
        result.get("modelArtifactPath").asText() shouldContain "s3://fraud-detection-models"
        result.get("trainingJobStatus").asText() shouldBe "Completed"
    }
    
    test("processData should create training job and wait for completion") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        var createJobCalled = false
        var waiterCalled = false
        var describeJobCalled = false
        
        every { mockSageMakerClient.createTrainingJob(any<CreateTrainingJobRequest>()) } answers {
            createJobCalled = true
            mockk()
        }
        
        every { mockSageMakerClient.waiter() } returns mockWaiter
        
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } answers {
            waiterCalled = true
            mockk()
        }
        
        every { 
            mockSageMakerClient.describeTrainingJob(any<DescribeTrainingJobRequest>()) 
        } answers {
            describeJobCalled = true
            DescribeTrainingJobResponse.builder()
                .trainingJobStatus(TrainingJobStatus.COMPLETED)
                .modelArtifacts(ModelArtifacts.builder()
                    .s3ModelArtifacts("s3://fraud-detection-models/fraud-detection-123/output/model.tar.gz")
                    .build())
                .build()
        }
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify all steps were called
        createJobCalled shouldBe true
        waiterCalled shouldBe true
        describeJobCalled shouldBe true
        
        // Verify result
        result.get("trainingJobStatus").asText() shouldBe "Completed"
        result.get("modelArtifactPath").asText() shouldNotBe null
    }
    
    test("processData should throw exception when training job fails") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        every { mockSageMakerClient.createTrainingJob(any<CreateTrainingJobRequest>()) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } returns mockk()
        
        // Simulate failed training job
        every { 
            mockSageMakerClient.describeTrainingJob(any<DescribeTrainingJobRequest>()) 
        } returns DescribeTrainingJobResponse.builder()
            .trainingJobStatus(TrainingJobStatus.FAILED)
            .failureReason("Insufficient training data")
            .build()
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalStateException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "Training job failed with status"
        exception.message shouldContain "Failed"
        exception.message shouldContain "Insufficient training data"
    }
    
    test("processData should throw exception when waiter times out") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        every { mockSageMakerClient.createTrainingJob(any<CreateTrainingJobRequest>()) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        
        // Simulate waiter timeout
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } throws RuntimeException("Waiter timeout exceeded")
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalStateException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "Training job did not complete within timeout"
        exception.message shouldContain "Waiter timeout exceeded"
    }
    
    test("processData should throw exception when trainDataPath is missing") {
        // Create handler with mock client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input without trainDataPath
        val input = objectMapper.createObjectNode().apply {
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalArgumentException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "trainDataPath is required"
    }
    
    test("processData should throw exception when validationDataPath is missing") {
        // Create handler with mock client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input without validationDataPath
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalArgumentException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "validationDataPath is required"
    }
    
    test("processData should handle training job with stopped status") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        every { mockSageMakerClient.createTrainingJob(any<CreateTrainingJobRequest>()) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } returns mockk()
        
        // Simulate stopped training job
        every { 
            mockSageMakerClient.describeTrainingJob(any<DescribeTrainingJobRequest>()) 
        } returns DescribeTrainingJobResponse.builder()
            .trainingJobStatus(TrainingJobStatus.STOPPED)
            .failureReason("Training job was manually stopped")
            .build()
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalStateException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "Training job failed with status"
        exception.message shouldContain "Stopped"
        exception.message shouldContain "Training job was manually stopped"
    }
    
    test("processData should generate unique training job names") {
        // Mock SageMaker client
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        val capturedJobNames = mutableListOf<String>()
        
        every { mockSageMakerClient.createTrainingJob(any<CreateTrainingJobRequest>()) } answers {
            val request = firstArg<CreateTrainingJobRequest>()
            capturedJobNames.add(request.trainingJobName())
            mockk()
        }
        
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilTrainingJobCompletedOrStopped(any<DescribeTrainingJobRequest>()) 
        } returns mockk()
        every { 
            mockSageMakerClient.describeTrainingJob(any<DescribeTrainingJobRequest>()) 
        } returns DescribeTrainingJobResponse.builder()
            .trainingJobStatus(TrainingJobStatus.COMPLETED)
            .modelArtifacts(ModelArtifacts.builder()
                .s3ModelArtifacts("s3://fraud-detection-models/fraud-detection-123/output/model.tar.gz")
                .build())
            .build()
        
        // Create handler with mocked client using constructor injection
        val handler = TestableTrainHandler(
            sageMakerClient = mockSageMakerClient,
            ssmClient = mockSsmClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainDataPath", "s3://fraud-detection-data/prepared/train.parquet")
            put("validationDataPath", "s3://fraud-detection-data/prepared/validation.parquet")
        }
        
        // Execute multiple times
        handler.processData(input)
        Thread.sleep(10) // Small delay to ensure different timestamps
        handler.processData(input)
        
        // Verify job names are unique
        capturedJobNames.size shouldBe 2
        capturedJobNames[0] shouldNotBe capturedJobNames[1]
        capturedJobNames.forEach { jobName ->
            jobName shouldContain "fraud-detection-"
        }
    }
})
