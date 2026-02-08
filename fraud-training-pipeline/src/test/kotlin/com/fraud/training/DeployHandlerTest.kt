package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemaker.waiters.SageMakerWaiter
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse

/**
 * Test wrapper for DeployHandler that exposes processData for testing and allows S3 client injection.
 */
class TestableDeployHandler(
    sageMakerClient: SageMakerClient,
    sageMakerRuntimeClient: SageMakerRuntimeClient,
    sageMakerExecutionRoleArn: String,
    configBucket: String
) : DeployHandler(sageMakerClient, sageMakerRuntimeClient, sageMakerExecutionRoleArn, configBucket) {
    
    // Allow setting a mock S3 client for testing
    fun setS3Client(mockS3Client: S3Client) {
        val baseClass = this.javaClass.superclass.superclass // DeployHandler -> WorkflowLambdaHandler
        val s3ClientField = baseClass.getDeclaredField("s3Client")
        s3ClientField.isAccessible = true
        s3ClientField.set(this, mockS3Client)
    }
    
    public override fun processData(input: JsonNode): JsonNode {
        return super.processData(input)
    }
}

/**
 * Unit tests for DeployHandler.
 * 
 * Tests cover:
 * - Endpoint creation with mocked SageMaker client
 * - Endpoint update for existing endpoints
 * - Health check validation
 * 
 * Requirements tested:
 * - Requirement 4.2: Create or update a SageMaker_Endpoint with the trained model
 * - Requirement 4.3: Configure the endpoint with appropriate instance type for real-time inference
 * - Requirement 4.4: Validate endpoint health by sending a test transaction
 */
class DeployHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("processData should create new endpoint when endpoint does not exist") {
        // Mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockS3Client = mockk<S3Client>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        // Capture requests
        val capturedModelRequest = slot<CreateModelRequest>()
        val capturedEndpointConfigRequest = slot<CreateEndpointConfigRequest>()
        val capturedEndpointRequest = slot<CreateEndpointRequest>()
        
        // Mock endpoint does not exist
        every { 
            mockSageMakerClient.describeEndpoint(any<DescribeEndpointRequest>()) 
        } throws ResourceNotFoundException.builder().message("Endpoint not found").build()
        
        every { mockSageMakerClient.createModel(capture(capturedModelRequest)) } returns mockk()
        every { mockSageMakerClient.createEndpointConfig(capture(capturedEndpointConfigRequest)) } returns mockk()
        every { mockSageMakerClient.createEndpoint(capture(capturedEndpointRequest)) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) } returns mockk()
        
        // Mock health check
        val mockInvokeResponse = mockk<InvokeEndpointResponse>()
        every { mockInvokeResponse.body() } returns SdkBytes.fromUtf8String("0.0234")
        every { mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) } returns mockInvokeResponse
        
        // Mock S3 write
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns mockk()
        
        // Create handler
        val handler = TestableDeployHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            configBucket = "fraud-detection-config"
        )
        handler.setS3Client(mockS3Client)
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-123/output/model.tar.gz")
            put("accuracy", 0.9523)
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify model creation
        val modelRequest = capturedModelRequest.captured
        modelRequest.modelName() shouldContain "fraud-detection-prod-"
        modelRequest.primaryContainer().image() shouldContain "xgboost"
        modelRequest.primaryContainer().modelDataUrl() shouldBe "s3://fraud-detection-models/fraud-detection-123/output/model.tar.gz"
        
        // Verify endpoint config
        val endpointConfigRequest = capturedEndpointConfigRequest.captured
        endpointConfigRequest.productionVariants().size shouldBe 1
        val variant = endpointConfigRequest.productionVariants()[0]
        variant.instanceType() shouldBe ProductionVariantInstanceType.ML_M5_LARGE
        variant.initialInstanceCount() shouldBe 1
        
        // Verify endpoint creation
        val endpointRequest = capturedEndpointRequest.captured
        endpointRequest.endpointName() shouldBe "fraud-detection-prod"
        
        // Verify result
        result.get("endpointName").asText() shouldBe "fraud-detection-prod"
        result.get("modelAccuracy").asDouble() shouldBe 0.9523
        result.get("healthCheckPrediction").asDouble() shouldBe 0.0234
    }
    
    test("processData should update existing endpoint when endpoint already exists") {
        // Mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockS3Client = mockk<S3Client>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        val capturedUpdateRequest = slot<UpdateEndpointRequest>()
        
        // Mock endpoint exists
        every { 
            mockSageMakerClient.describeEndpoint(any<DescribeEndpointRequest>()) 
        } returns DescribeEndpointResponse.builder()
            .endpointName("fraud-detection-prod")
            .endpointStatus(EndpointStatus.IN_SERVICE)
            .build()
        
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.updateEndpoint(capture(capturedUpdateRequest)) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) } returns mockk()
        
        // Mock health check
        val mockInvokeResponse = mockk<InvokeEndpointResponse>()
        every { mockInvokeResponse.body() } returns SdkBytes.fromUtf8String("0.0456")
        every { mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) } returns mockInvokeResponse
        
        // Mock S3 write
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns mockk()
        
        // Create handler
        val handler = TestableDeployHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            configBucket = "fraud-detection-config"
        )
        handler.setS3Client(mockS3Client)
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-456/output/model.tar.gz")
            put("accuracy", 0.9612)
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify endpoint was updated
        val updateRequest = capturedUpdateRequest.captured
        updateRequest.endpointName() shouldBe "fraud-detection-prod"
        
        // Verify result
        result.get("endpointName").asText() shouldBe "fraud-detection-prod"
        result.get("modelAccuracy").asDouble() shouldBe 0.9612
    }
    
    test("processData should perform health check and validate endpoint") {
        // Mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockS3Client = mockk<S3Client>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        val capturedInvokeRequest = slot<InvokeEndpointRequest>()
        
        every { 
            mockSageMakerClient.describeEndpoint(any<DescribeEndpointRequest>()) 
        } throws ResourceNotFoundException.builder().message("Endpoint not found").build()
        
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) } returns mockk()
        
        // Mock health check
        val mockInvokeResponse = mockk<InvokeEndpointResponse>()
        every { mockInvokeResponse.body() } returns SdkBytes.fromUtf8String("0.0789")
        every { mockSageMakerRuntimeClient.invokeEndpoint(capture(capturedInvokeRequest)) } returns mockInvokeResponse
        
        // Mock S3 write
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns mockk()
        
        // Create handler
        val handler = TestableDeployHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            configBucket = "fraud-detection-config"
        )
        handler.setS3Client(mockS3Client)
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-789/output/model.tar.gz")
            put("accuracy", 0.9345)
        }
        
        // Execute
        val result = handler.processData(input)
        
        // Verify health check was performed
        val invokeRequest = capturedInvokeRequest.captured
        invokeRequest.endpointName() shouldBe "fraud-detection-prod"
        invokeRequest.contentType() shouldBe "application/json"
        
        // Verify result includes health check prediction
        result.get("healthCheckPrediction").asDouble() shouldBe 0.0789
    }
    
    test("processData should throw exception when health check fails") {
        // Mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockS3Client = mockk<S3Client>()
        val mockWaiter = mockk<SageMakerWaiter>()
        
        every { 
            mockSageMakerClient.describeEndpoint(any<DescribeEndpointRequest>()) 
        } throws ResourceNotFoundException.builder().message("Endpoint not found").build()
        
        every { mockSageMakerClient.createModel(any<CreateModelRequest>()) } returns mockk()
        every { mockSageMakerClient.createEndpointConfig(any<CreateEndpointConfigRequest>()) } returns mockk()
        every { mockSageMakerClient.createEndpoint(any<CreateEndpointRequest>()) } returns mockk()
        every { mockSageMakerClient.waiter() } returns mockWaiter
        every { mockWaiter.waitUntilEndpointInService(any<DescribeEndpointRequest>()) } returns mockk()
        
        // Mock health check failure
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } throws RuntimeException("Endpoint invocation failed")
        
        // Create handler
        val handler = TestableDeployHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            configBucket = "fraud-detection-config"
        )
        handler.setS3Client(mockS3Client)
        
        // Prepare input
        val input = objectMapper.createObjectNode().apply {
            put("modelArtifactPath", "s3://fraud-detection-models/fraud-detection-999/output/model.tar.gz")
            put("accuracy", 0.9234)
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalStateException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "Endpoint health check failed"
    }
    
    test("processData should throw exception when modelArtifactPath is missing") {
        // Mock clients
        val mockSageMakerClient = mockk<SageMakerClient>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        val mockS3Client = mockk<S3Client>()
        
        val handler = TestableDeployHandler(
            sageMakerClient = mockSageMakerClient,
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            sageMakerExecutionRoleArn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            configBucket = "fraud-detection-config"
        )
        handler.setS3Client(mockS3Client)
        
        // Prepare input without modelArtifactPath
        val input = objectMapper.createObjectNode().apply {
            put("accuracy", 0.9234)
        }
        
        // Execute and verify exception
        val exception = shouldThrow<IllegalArgumentException> {
            handler.processData(input)
        }
        
        exception.message shouldContain "modelArtifactPath is required"
    }
})
