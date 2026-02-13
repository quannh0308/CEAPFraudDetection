package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.Transaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectResponse
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse
import java.io.ByteArrayInputStream

/**
 * Property-based tests for endpoint invocation in ScoreHandler.
 * 
 * This test suite validates:
 * - Property 11: Endpoint Invocation
 * 
 * **Requirements:**
 * - Requirement 6.3: Invoke the SageMaker_Endpoint for each transaction
 */
class ScoreHandlerEndpointInvocationPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 11: Endpoint Invocation
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 11: Endpoint Invocation
    test("Property 11: Endpoint Invocation - scoring stage SHALL invoke endpoint exactly once per transaction") {
        checkAll(100, Arb.transactionBatch(minSize = 1, maxSize = 50)) { transactions ->
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
                    match<GetObjectRequest> { it.key() == "daily-batches/$batchDate.json" }
                ) 
            } returns transactionBatchStream
            
            // Mock S3 output write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            // Track endpoint invocations
            var invocationCount = 0
            
            // Mock SageMaker endpoint invocations
            every { 
                mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
            } answers {
                invocationCount++
                val request = firstArg<InvokeEndpointRequest>()
                
                // Verify endpoint name is correct
                request.endpointName() shouldBe endpointName
                
                // Return a valid fraud score
                val fraudScore = 0.5 // Fixed score for testing
                InvokeEndpointResponse.builder()
                    .body(SdkBytes.fromUtf8String(fraudScore.toString()))
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
            
            // When the scoring stage processes the batch
            handler.handleRequest(input, mockContext)
            
            // Then the endpoint SHALL be invoked exactly once per transaction
            invocationCount shouldBe transactions.size
            
            // Verify each invocation was made
            verify(exactly = transactions.size) {
                mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>())
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 11: Endpoint Invocation
    test("Property 11: Endpoint Invocation - each invocation SHALL use correct endpoint name") {
        checkAll(100, Arb.transactionBatch(minSize = 1, maxSize = 20), Arb.endpointName()) { transactions, endpointName ->
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
                    match<GetObjectRequest> { it.key() == "daily-batches/$batchDate.json" }
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
                
                // Verify endpoint name matches the one from config
                request.endpointName() shouldBe endpointName
                
                // Return a valid fraud score
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
            
            // When the scoring stage processes the batch
            handler.handleRequest(input, mockContext)
            
            // Then all invocations SHALL use the correct endpoint name
            verify(exactly = transactions.size) {
                mockSageMakerRuntimeClient.invokeEndpoint(
                    match<InvokeEndpointRequest> { it.endpointName() == endpointName }
                )
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 11: Endpoint Invocation
    test("Property 11: Endpoint Invocation - each invocation SHALL include transaction features") {
        checkAll(100, Arb.transactionBatch(minSize = 1, maxSize = 10)) { transactions ->
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
                    match<GetObjectRequest> { it.key() == "daily-batches/$batchDate.json" }
                ) 
            } returns transactionBatchStream
            
            // Mock S3 output write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            val invokedFeatures = mutableListOf<Map<String, Double>>()
            
            // Mock SageMaker endpoint invocations
            every { 
                mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
            } answers {
                val request = firstArg<InvokeEndpointRequest>()
                
                // Capture the features sent in the request
                val payload = request.body().asUtf8String()
                val features = objectMapper.readValue(payload, Map::class.java) as Map<String, Double>
                invokedFeatures.add(features)
                
                // Return a valid fraud score
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
            
            // When the scoring stage processes the batch
            handler.handleRequest(input, mockContext)
            
            // Then each invocation SHALL include the correct transaction features
            invokedFeatures.size shouldBe transactions.size
            
            transactions.forEachIndexed { index, transaction ->
                invokedFeatures[index] shouldBe transaction.features
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 11: Endpoint Invocation
    test("Property 11: Endpoint Invocation - SHALL invoke endpoint for empty batch (zero invocations)") {
        checkAll(100, Arb.endpointName()) { endpointName ->
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
                    match<GetObjectRequest> { it.key() == "daily-batches/$batchDate.json" }
                ) 
            } returns transactionBatchStream
            
            // Mock S3 output write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            // Mock SageMaker endpoint invocations (should not be called)
            every { 
                mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>()) 
            } answers {
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
            
            // When the scoring stage processes an empty batch
            handler.handleRequest(input, mockContext)
            
            // Then the endpoint SHALL NOT be invoked (zero invocations)
            verify(exactly = 0) {
                mockSageMakerRuntimeClient.invokeEndpoint(any<InvokeEndpointRequest>())
            }
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary transaction batches with configurable size range.
 */
private fun Arb.Companion.transactionBatch(minSize: Int = 0, maxSize: Int = 100): Arb<List<Transaction>> {
    return arbitrary { rs ->
        val size = Arb.int(minSize..maxSize).bind()
        List(size) { Arb.transaction().bind() }
    }
}

/**
 * Generates arbitrary Transaction instances.
 */
private fun Arb.Companion.transaction(): Arb<Transaction> {
    return arbitrary { rs ->
        Transaction(
            id = Arb.transactionId().bind(),
            timestamp = Arb.long(1700000000000L..1800000000000L).bind(),
            amount = Arb.double(0.01, 10000.0).bind(),
            merchantCategory = Arb.merchantCategory().bind(),
            features = Arb.transactionFeatures().bind()
        )
    }
}

/**
 * Generates arbitrary transaction IDs.
 */
private fun Arb.Companion.transactionId(): Arb<String> {
    return Arb.string(10..20, Codepoint.alphanumeric()).map { "txn-$it" }
}

/**
 * Generates arbitrary merchant categories.
 */
private fun Arb.Companion.merchantCategory(): Arb<String> {
    return Arb.of(
        "retail",
        "online",
        "restaurant",
        "gas_station",
        "grocery",
        "entertainment",
        "travel",
        "healthcare"
    )
}

/**
 * Generates arbitrary transaction features (V1-V28 from Kaggle dataset).
 */
private fun Arb.Companion.transactionFeatures(): Arb<Map<String, Double>> {
    return arbitrary { rs ->
        val features = mutableMapOf<String, Double>()
        
        // Add Time feature
        features["Time"] = Arb.double(0.0, 172800.0).bind() // 48 hours in seconds
        
        // Add V1-V28 features (PCA components)
        for (i in 1..28) {
            features["V$i"] = Arb.double(-10.0, 10.0).bind()
        }
        
        // Add Amount feature
        features["Amount"] = Arb.double(0.01, 10000.0).bind()
        
        features
    }
}

/**
 * Generates arbitrary SageMaker endpoint names.
 */
private fun Arb.Companion.endpointName(): Arb<String> {
    return Arb.choice(
        Arb.string(10..20, Codepoint.alphanumeric()).map { "fraud-detection-$it" },
        Arb.constant("fraud-detection-prod"),
        Arb.string(10..15, Codepoint.alphanumeric()).map { "fraud-detection-prod-$it" }
    )
}
