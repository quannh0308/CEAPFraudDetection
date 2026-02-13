package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import com.fraud.common.models.Transaction
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectResponse
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse
import java.io.ByteArrayInputStream

/**
 * Unit tests for ScoreHandler.
 * 
 * This test suite validates:
 * - Transaction batch loading from S3
 * - Endpoint invocation with mocked SageMaker Runtime client
 * - Scored transaction creation
 * 
 * **Requirements:**
 * - Requirement 6.1: Load daily transaction batch from S3
 * - Requirement 6.2: Read the current SageMaker_Endpoint name from S3
 * - Requirement 6.3: Invoke the SageMaker_Endpoint for each transaction
 */
class ScoreHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Transaction Batch Loading Tests
    // ========================================
    
    test("should load transaction batch from S3 successfully") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transactions
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            ),
            Transaction(
                id = "txn-002",
                timestamp = 1705334500000L,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock S3 output write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } returns InvokeEndpointResponse.builder()
            .body(SdkBytes.fromUtf8String("0.5"))
            .build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then the transaction batch should be loaded successfully
        result shouldNotBe null
        result.status shouldBe "SUCCESS"
        
        // Verify S3 was called to load the transaction batch
        verify(exactly = 1) {
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            )
        }
    }
    
    test("should fail when transaction batch path is missing") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        // Create input without transactionBatchPath
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "batchDate" to "2024-01-15"
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then it should fail with appropriate error
        result.status shouldBe "FAILED"
        result.errorMessage shouldContain "transactionBatchPath is required"
    }
    
    test("should fail when batch date is missing") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        // Create input without batchDate
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to "s3://fraud-detection-data/daily-batches/2024-01-15.json"
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then it should fail with appropriate error
        result.status shouldBe "FAILED"
        result.errorMessage shouldContain "batchDate is required"
    }
    
    test("should handle empty transaction batch") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock empty transaction batch
        val emptyBatch = emptyList<Transaction>()
        val transactionBatchJson = objectMapper.writeValueAsString(emptyBatch)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock S3 output write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes an empty batch
        val result = handler.handleRequest(input, mockContext)
        
        // Then it should succeed with zero transactions
        result.status shouldBe "SUCCESS"
        
        // Verify no endpoint invocations were made
        verify(exactly = 0) {
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>())
        }
    }
    
    // ========================================
    // Endpoint Invocation Tests
    // ========================================
    
    test("should invoke SageMaker endpoint for each transaction") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transactions
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            ),
            Transaction(
                id = "txn-002",
                timestamp = 1705334500000L,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
            ),
            Transaction(
                id = "txn-003",
                timestamp = 1705334600000L,
                amount = 75.00,
                merchantCategory = "restaurant",
                features = mapOf("Time" to 200.0, "V1" to 0.5, "Amount" to 75.00)
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock S3 output write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } returns InvokeEndpointResponse.builder()
            .body(SdkBytes.fromUtf8String("0.5"))
            .build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then the endpoint should be invoked exactly once per transaction
        verify(exactly = transactions.size) {
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>())
        }
        
        result.status shouldBe "SUCCESS"
    }
    
    test("should use correct endpoint name from config") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod-custom"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transaction
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            )
        )
        
        // Mock endpoint metadata read with custom endpoint name
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock S3 output write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            // Verify the correct endpoint name is used
            request.endpointName() shouldBe endpointName
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String("0.5"))
                .build()
        }
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then the correct endpoint name should be used
        result.status shouldBe "SUCCESS"
        
        verify(exactly = 1) {
            mockSageMakerRuntimeClient.invokeEndpoint(
                match<InvokeEndpointRequest> { it.endpointName() == endpointName }
            )
        }
    }
    
    test("should send transaction features in endpoint invocation") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transaction with specific features
        val expectedFeatures = mapOf(
            "Time" to 12345.0,
            "V1" to -1.36,
            "V2" to 0.72,
            "Amount" to 149.62
        )
        
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = expectedFeatures
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Mock S3 output write
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } returns PutObjectResponse.builder().build()
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val request = firstArg<InvokeEndpointRequest>()
            
            // Verify the features are sent correctly
            val payload = request.body().asUtf8String()
            val sentFeatures = objectMapper.readValue(payload, Map::class.java) as Map<String, Double>
            sentFeatures shouldBe expectedFeatures
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String("0.5"))
                .build()
        }
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then the transaction features should be sent correctly
        result.status shouldBe "SUCCESS"
    }
    
    // ========================================
    // Scored Transaction Creation Tests
    // ========================================
    
    test("should create scored transactions with correct attributes") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transactions
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            ),
            Transaction(
                id = "txn-002",
                timestamp = 1705334500000L,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Capture the output written to S3
        var capturedOutput: String? = null
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            capturedOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Mock SageMaker endpoint invocations with different scores
        val fraudScores = listOf(0.0234, 0.8912)
        var invocationIndex = 0
        
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } answers {
            val score = fraudScores[invocationIndex]
            invocationIndex++
            
            InvokeEndpointResponse.builder()
                .body(SdkBytes.fromUtf8String(score.toString()))
                .build()
        }
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then scored transactions should be created with correct attributes
        result.status shouldBe "SUCCESS"
        
        // Parse the captured output
        capturedOutput shouldNotBe null
        val outputJson = objectMapper.readTree(capturedOutput)
        val scoredTransactions = outputJson.get("scoredTransactions")
        
        scoredTransactions.size() shouldBe 2
        
        // Verify first scored transaction
        val scoredTxn1 = scoredTransactions.get(0)
        scoredTxn1.get("transactionId").asText() shouldBe "txn-001"
        scoredTxn1.get("timestamp").asLong() shouldBe 1705334400000L
        scoredTxn1.get("amount").asDouble() shouldBe 149.62
        scoredTxn1.get("merchantCategory").asText() shouldBe "retail"
        scoredTxn1.get("fraudScore").asDouble() shouldBe 0.0234
        scoredTxn1.get("scoringTimestamp").asLong() shouldNotBe 0L
        
        // Verify second scored transaction
        val scoredTxn2 = scoredTransactions.get(1)
        scoredTxn2.get("transactionId").asText() shouldBe "txn-002"
        scoredTxn2.get("timestamp").asLong() shouldBe 1705334500000L
        scoredTxn2.get("amount").asDouble() shouldBe 2500.00
        scoredTxn2.get("merchantCategory").asText() shouldBe "online"
        scoredTxn2.get("fraudScore").asDouble() shouldBe 0.8912
        scoredTxn2.get("scoringTimestamp").asLong() shouldNotBe 0L
    }
    
    test("should include transaction features in scored transactions") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transaction with specific features
        val expectedFeatures = mapOf(
            "Time" to 12345.0,
            "V1" to -1.36,
            "V2" to 0.72,
            "V3" to 1.23,
            "Amount" to 149.62
        )
        
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = expectedFeatures
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Capture the output written to S3
        var capturedOutput: String? = null
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            capturedOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } returns InvokeEndpointResponse.builder()
            .body(SdkBytes.fromUtf8String("0.5"))
            .build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then scored transactions should include the original features
        result.status shouldBe "SUCCESS"
        
        // Parse the captured output
        capturedOutput shouldNotBe null
        val outputJson = objectMapper.readTree(capturedOutput)
        val scoredTransactions = outputJson.get("scoredTransactions")
        
        scoredTransactions.size() shouldBe 1
        
        // Verify features are preserved
        val scoredTxn = scoredTransactions.get(0)
        val features = scoredTxn.get("features")
        
        features.get("Time").asDouble() shouldBe 12345.0
        features.get("V1").asDouble() shouldBe -1.36
        features.get("V2").asDouble() shouldBe 0.72
        features.get("V3").asDouble() shouldBe 1.23
        features.get("Amount").asDouble() shouldBe 149.62
    }
    
    test("should include batch metadata in output") {
        val mockS3Client = mockk<S3Client>()
        val mockSageMakerRuntimeClient = mockk<SageMakerRuntimeClient>()
        
        val handler = ScoreHandler(
            sageMakerRuntimeClient = mockSageMakerRuntimeClient,
            configBucket = "test-config-bucket"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val endpointName = "fraud-detection-prod"
        val batchDate = "2024-01-15"
        val transactionBatchPath = "s3://fraud-detection-data/daily-batches/$batchDate.json"
        
        // Create test transactions
        val transactions = listOf(
            Transaction(
                id = "txn-001",
                timestamp = 1705334400000L,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62)
            ),
            Transaction(
                id = "txn-002",
                timestamp = 1705334500000L,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00)
            )
        )
        
        // Mock endpoint metadata read
        val endpointMetadata = objectMapper.createObjectNode().apply {
            put("endpointName", endpointName)
            put("modelName", "fraud-detection-prod-123")
            put("deploymentTimestamp", System.currentTimeMillis())
        }
        
        val endpointMetadataStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(endpointMetadata).toByteArray())
            )
        )
        
        // Mock transaction batch read
        val transactionBatchJson = objectMapper.writeValueAsString(transactions)
        val transactionBatchStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(ByteArrayInputStream(transactionBatchJson.toByteArray()))
        )
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { it.key() == "current-endpoint.json" }
            ) 
        } returns endpointMetadataStream
        
        every { 
            mockS3Client.getObject(
                match<GetObjectRequest> { 
                    it.bucket() == "fraud-detection-data" && it.key() == "daily-batches/$batchDate.json" 
                }
            ) 
        } returns transactionBatchStream
        
        // Capture the output written to S3
        var capturedOutput: String? = null
        every { 
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
        } answers {
            val requestBody = secondArg<RequestBody>()
            capturedOutput = requestBody.contentStreamProvider().newStream().readAllBytes().toString(Charsets.UTF_8)
            PutObjectResponse.builder().build()
        }
        
        // Mock SageMaker endpoint invocations
        every { 
            mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
        } returns InvokeEndpointResponse.builder()
            .body(SdkBytes.fromUtf8String("0.5"))
            .build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket",
            "initialData" to mapOf(
                "transactionBatchPath" to transactionBatchPath,
                "batchDate" to batchDate
            )
        )
        
        val mockContext = mockk<Context>()
        
        // When the handler processes the request
        val result = handler.handleRequest(input, mockContext)
        
        // Then output should include batch metadata
        result.status shouldBe "SUCCESS"
        
        // Parse the captured output
        capturedOutput shouldNotBe null
        val outputJson = objectMapper.readTree(capturedOutput)
        
        // Verify metadata fields
        outputJson.get("batchDate").asText() shouldBe batchDate
        outputJson.get("transactionCount").asInt() shouldBe 2
        outputJson.get("endpointName").asText() shouldBe endpointName
        outputJson.has("scoredTransactions") shouldBe true
    }
})
