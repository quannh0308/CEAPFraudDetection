package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.doubles.shouldBeGreaterThanOrEqual
import io.kotest.matchers.doubles.shouldBeLessThanOrEqual
import io.kotest.matchers.ints.shouldBeGreaterThan as intShouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeGreaterThan as doubleShouldBeGreaterThan
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemaker.waiters.SageMakerWaiter
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse

/**
 * Test wrapper for EvaluateHandler that exposes processData for testing.
 */
class TestableEvaluateHandler(
    sageMakerClient: SageMakerClient,
    sageMakerRuntimeClient: SageMakerRuntimeClient,
    sageMakerExecutionRoleArn: String
) : EvaluateHandler(sageMakerClient, sageMakerRuntimeClient, sageMakerExecutionRoleArn) {
    public override fun processData(input: JsonNode): JsonNode {
        return super.processData(input)
    }
}

/**
 * Unit tests for EvaluateHandler.
 * 
 * Tests cover:
 * - Model evaluation with mocked endpoint
 * - Accuracy threshold validation (should fail if < 0.90)
 * - Metrics calculation
 * 
 * Requirements tested:
 * - Requirement 3.6: Fail if model accuracy is below 0.90
 */
class EvaluateHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("processData should create temporary endpoint and evaluate model") {
        // Mock SageMaker clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        var createModelCalled = false
        var createEndpointConfigCalled = false
        var createEndpointCalled = false
        var deleteEndpointCalled = false
        var deleteEndpointConfigCalled = false
        var deleteModelCalled = false
        
        // Mock model creation
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } answers {
            createModelCalled = true
            mockk()
        }
        
        // Mock endpoint config creation
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } answers {
            createEndpointConfigCalled = true
            mockk()
        }
        
        // Mock endpoint creation
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } answers {
            createEndpointCalled = true
            mockk()
        }
        
        // Mock waiter
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) 
        } returns mockk()
        
        // Mock endpoint invocation (return high accuracy predictions)
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            val payload = request.body().asUtf8String()
            
            // Parse features to determine if it's fraud (simple mock logic)
            val features = objectMapper.readTree(payload)
            val v1 = features.get("V1")?.asDouble() ?: 0.0
            
            // Mock prediction: high V1 values indicate fraud
            val prediction = if (v1 > 1.0) 0.95 else 0.05
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(prediction.toString()))
                .build()
        }
        
        // Mock cleanup
        every { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) } answers {
            deleteEndpointCalled = true
            mockk()
        }
        every { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) } answers {
            deleteEndpointConfigCalled = true
            mockk()
        }
        every { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) } answers {
            deleteModelCalled = true
            mockk()
        }
        
        // Create handler with mocked clients
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify all steps were called
        createModelCalled shouldBe true
        createEndpointConfigCalled shouldBe true
        createEndpointCalled shouldBe true
        deleteEndpointCalled shouldBe true
        deleteEndpointConfigCalled shouldBe true
        deleteModelCalled shouldBe true
        
        // Verify result contains metrics
        result.get("modelArtifactPath").asText() shouldBe "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz"
        result.get("accuracy").asDouble() shouldBeGreaterThanOrEqual 0.0
        result.get("accuracy").asDouble() shouldBeLessThanOrEqual 1.0
        result.get("precision").asDouble() shouldBeGreaterThanOrEqual 0.0
        result.get("recall").asDouble() shouldBeGreaterThanOrEqual 0.0
        result.get("f1Score").asDouble() shouldBeGreaterThanOrEqual 0.0
        result.get("auc").asDouble() shouldBeGreaterThanOrEqual 0.0
        result.get("testRecordCount").asInt() intShouldBeGreaterThan 0
    }
    
    test("processData should fail when accuracy is below 0.90 threshold") {
        // Mock SageMaker clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        // Mock model creation
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        
        // Mock endpoint config creation
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        
        // Mock endpoint creation
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } returns mockk()
        
        // Mock waiter
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) 
        } returns mockk()
        
        // Mock endpoint invocation (return low accuracy predictions - all wrong)
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            val payload = request.body().asUtf8String()
            
            // Parse features to determine if it's fraud
            val features = objectMapper.readTree(payload)
            val v1 = features.get("V1")?.asDouble() ?: 0.0
            
            // Mock prediction: WRONG predictions (inverted logic for low accuracy)
            val prediction = if (v1 > 1.0) 0.05 else 0.95
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(prediction.toString()))
                .build()
        }
        
        // Mock cleanup (should still be called even on failure)
        every { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) } returns mockk()
        
        // Create handler with mocked clients
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalStateException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "Model accuracy"
        exception.message shouldContain "below minimum threshold 0.90"
        
        // Verify cleanup was still called
        verify { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) }
        verify { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) }
        verify { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) }
    }
    
    test("processData should calculate metrics correctly") {
        // Mock SageMaker clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        // Mock model creation
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        
        // Mock endpoint config creation
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        
        // Mock endpoint creation
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } returns mockk()
        
        // Mock waiter
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) 
        } returns mockk()
        
        // Mock endpoint invocation with perfect predictions
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            val payload = request.body().asUtf8String()
            
            // Parse features to determine if it's fraud
            val features = objectMapper.readTree(payload)
            val v1 = features.get("V1")?.asDouble() ?: 0.0
            
            // Mock prediction: perfect predictions
            val prediction = if (v1 > 1.0) 0.95 else 0.05
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(prediction.toString()))
                .build()
        }
        
        // Mock cleanup
        every { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) } returns mockk()
        
        // Create handler with mocked clients
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify metrics are calculated
        val accuracy = result.get("accuracy").asDouble()
        val precision = result.get("precision").asDouble()
        val recall = result.get("recall").asDouble()
        val f1Score = result.get("f1Score").asDouble()
        val auc = result.get("auc").asDouble()
        
        // With perfect predictions, accuracy should be 1.0
        accuracy shouldBe 1.0
        
        // Precision and recall should be 1.0 for perfect predictions
        precision shouldBe 1.0
        recall shouldBe 1.0
        
        // F1 score should be 1.0 for perfect predictions
        f1Score shouldBe 1.0
        
        // AUC should be high for good predictions
        auc doubleShouldBeGreaterThan 0.5
    }
    
    test("processData should cleanup resources even when evaluation fails") {
        // Mock SageMaker clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        var deleteEndpointCalled = false
        var deleteEndpointConfigCalled = false
        var deleteModelCalled = false
        
        // Mock model creation
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        
        // Mock endpoint config creation
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        
        // Mock endpoint creation
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } returns mockk()
        
        // Mock waiter
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) 
        } returns mockk()
        
        // Mock endpoint invocation to throw exception
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } throws RuntimeException("Endpoint invocation failed")
        
        // Mock cleanup
        every { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) } answers {
            deleteEndpointCalled = true
            mockk()
        }
        every { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) } answers {
            deleteEndpointConfigCalled = true
            mockk()
        }
        every { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) } answers {
            deleteModelCalled = true
            mockk()
        }
        
        // Create handler with mocked clients
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute and verify exception
        shouldThrow<RuntimeException> {
            handler.processData(input)
        }
        
        // Verify cleanup was still called despite the exception
        deleteEndpointCalled shouldBe true
        deleteEndpointConfigCalled shouldBe true
        deleteModelCalled shouldBe true
    }
    
    test("processData should throw exception when modelArtifactPath is missing") {
        // Create handler with mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input without modelArtifactPath
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalArgumentException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "modelArtifactPath is required"
    }
    
    test("processData should configure endpoint with correct parameters") {
        // Mock SageMaker clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        // Capture requests
        val capturedModelRequest = slot<CreateModelRequest>()
        val capturedEndpointConfigRequest = slot<CreateEndpointConfigRequest>()
        val capturedEndpointRequest = slot<CreateEndpointRequest>()
        
        // Mock model creation
        every { mockSageMakerClient.createModel(capture(capturedModelRequest)) } returns mockk()
        
        // Mock endpoint config creation
        every { mockSageMakerClient.createEndpointConfig(capture(capturedEndpointConfigRequest)) } returns mockk()
        
        // Mock endpoint creation
        every { mockSageMakerClient.createEndpoint(capture(capturedEndpointRequest)) } returns mockk()
        
        // Mock waiter
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { 
            mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) 
        } returns mockk()
        
        // Mock endpoint invocation (return correct predictions based on features)
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            val payload = request.body().asUtf8String()
            
            // Parse features to determine if it's fraud
            val features = objectMapper.readTree(payload)
            val v1 = features.get("V1")?.asDouble() ?: 0.0
            
            // Mock prediction: high V1 values indicate fraud
            val prediction = if (v1 > 1.0) 0.95 else 0.05
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(prediction.toString()))
                .build()
        }
        
        // Mock cleanup
        every { mockSageMakerClient.deleteEndpoint(any<DeleteEndpointRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteEndpointConfig(any<DeleteEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.deleteModel(any<DeleteModelRequest>()) } returns mockk()
        
        // Create handler with mocked clients
        val handler = TestableEvaluateHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        )
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("trainingJobName", "fraud-detection-1234567890")
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz")
            put("trainingJobStatus", "Completed")
            put("testDataPath", "s3://fraud-detection-data/prepared/test.parquet")
        }
        
        // Execute
        handler.processData(input)
        
        // Verify model configuration
        val modelRequest = capturedModelRequest.captured
        modelRequest.modelName() shouldContain "fraud-detection-eval-"
        modelRequest.primaryContainer().image() shouldContain "xgboost"
        modelRequest.primaryContainer().modelDataUrl() shouldBe "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz"
        modelRequest.executionRoleArn() shouldBe "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
        
        // Verify endpoint config
        val endpointConfigRequest = capturedEndpointConfigRequest.captured
        endpointConfigRequest.endpointConfigName() shouldContain "fraud-detection-eval-"
        endpointConfigRequest.productionVariants().size shouldBe 1
        endpointConfigRequest.productionVariants()[0].instanceType() shouldBe ProductionVariantInstanceType.ML_M5_LARGE
        endpointConfigRequest.productionVariants()[0].initialInstanceCount() shouldBe 1
        
        // Verify endpoint
        val endpointRequest = capturedEndpointRequest.captured
        endpointRequest.endpointName() shouldContain "fraud-detection-eval-"
    }
})
