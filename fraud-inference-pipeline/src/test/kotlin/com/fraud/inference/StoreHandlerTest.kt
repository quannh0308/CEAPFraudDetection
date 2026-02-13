package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.doubles.shouldBeGreaterThanOrEqual
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.shouldBe
import io.kotest.matchers.string.shouldContain
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.dynamodb.DynamoDbClient
import software.amazon.awssdk.services.dynamodb.model.AttributeValue
import software.amazon.awssdk.services.dynamodb.model.BatchWriteItemRequest
import software.amazon.awssdk.services.dynamodb.model.BatchWriteItemResponse
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectResponse
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import java.io.ByteArrayInputStream

/**
 * Unit tests for StoreHandler.
 * 
 * This test suite validates:
 * - Batch write with mocked DynamoDB client
 * - Unprocessed items handling
 * - Summary statistics calculation
 * 
 * **Requirements:**
 * - Requirement 7.2: Write each transaction record to DynamoDB with transaction ID as primary key
 * - Requirement 7.3: Store fraud score, timestamp, transaction amount, and relevant features
 * - Requirement 7.5: Write summary statistics to S3
 */
class StoreHandlerTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    test("should successfully write scored transactions to DynamoDB") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        // Replace S3 client with mock
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val batchDate = "2024-01-15"
        
        // Create test scored transactions
        val scoredTransactions = listOf(
            ScoredTransaction(
                transactionId = "txn-001",
                timestamp = 1705334400000,
                amount = 149.62,
                merchantCategory = "retail",
                features = mapOf("Time" to 0.0, "V1" to -1.36, "Amount" to 149.62),
                fraudScore = 0.0234,
                scoringTimestamp = 1705334401000
            ),
            ScoredTransaction(
                transactionId = "txn-002",
                timestamp = 1705334500000,
                amount = 2500.00,
                merchantCategory = "online",
                features = mapOf("Time" to 100.0, "V1" to 2.45, "Amount" to 2500.00),
                fraudScore = 0.8912,
                scoringTimestamp = 1705334501000
            )
        )
        
        // Mock S3 input read (from ScoreStage)
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        
        // Mock S3 output write
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        // Mock DynamoDB batch write (success)
        every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } returns BatchWriteItemResponse.builder()
            .unprocessedItems(emptyMap())
            .build()
        
        // Create input for handler
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes the scored transactions
        val result = handler.handleRequest(input, mockContext)
        
        // Then the handler should return success
        result.status shouldBe "SUCCESS"
        
        // And DynamoDB batch write should be called
        verify(exactly = 1) {
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>())
        }
        
        // And output should be written to S3
        verify(exactly = 1) {
            mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>())
        }
    }
    
    test("should handle unprocessed items with retry logic") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val batchDate = "2024-01-15"
        
        // Create test scored transactions (30 items to trigger multiple batches)
        val scoredTransactions = (1..30).map { i ->
            ScoredTransaction(
                transactionId = "txn-${String.format("%03d", i)}",
                timestamp = 1705334400000 + i * 1000,
                amount = 100.0 + i,
                merchantCategory = "retail",
                features = mapOf("Time" to i.toDouble(), "V1" to -1.0, "Amount" to (100.0 + i)),
                fraudScore = 0.1 * (i % 10),
                scoringTimestamp = 1705334401000 + i * 1000
            )
        }
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        var callCount = 0
        
        // Mock DynamoDB batch write with unprocessed items on first call, success on retry
        every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } answers {
            callCount++
            val request = firstArg<BatchWriteItemRequest>()
            val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
            
            if (callCount == 1 && writeRequests.size == 25) {
                // First batch: return 5 unprocessed items
                val unprocessedItems = writeRequests.take(5)
                BatchWriteItemResponse.builder()
                    .unprocessedItems(mapOf("FraudScores" to unprocessedItems))
                    .build()
            } else {
                // Subsequent calls: success
                BatchWriteItemResponse.builder()
                    .unprocessedItems(emptyMap())
                    .build()
            }
        }
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes the scored transactions
        val result = handler.handleRequest(input, mockContext)
        
        // Then the handler should return success
        result.status shouldBe "SUCCESS"
        
        // And DynamoDB batch write should be called multiple times (initial + retry + second batch)
        verify(atLeast = 3) {
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>())
        }
    }
    
    test("should calculate correct summary statistics") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val batchDate = "2024-01-15"
        
        // Create test scored transactions with known risk distribution
        val scoredTransactions = listOf(
            // High risk (>= 0.8): 2 transactions
            ScoredTransaction("txn-001", 1705334400000, 100.0, "retail", mapOf(), 0.85, 1705334401000),
            ScoredTransaction("txn-002", 1705334400000, 100.0, "retail", mapOf(), 0.92, 1705334401000),
            // Medium risk (>= 0.5, < 0.8): 3 transactions
            ScoredTransaction("txn-003", 1705334400000, 100.0, "retail", mapOf(), 0.55, 1705334401000),
            ScoredTransaction("txn-004", 1705334400000, 100.0, "retail", mapOf(), 0.65, 1705334401000),
            ScoredTransaction("txn-005", 1705334400000, 100.0, "retail", mapOf(), 0.75, 1705334401000),
            // Low risk (< 0.5): 5 transactions
            ScoredTransaction("txn-006", 1705334400000, 100.0, "retail", mapOf(), 0.10, 1705334401000),
            ScoredTransaction("txn-007", 1705334400000, 100.0, "retail", mapOf(), 0.20, 1705334401000),
            ScoredTransaction("txn-008", 1705334400000, 100.0, "retail", mapOf(), 0.30, 1705334401000),
            ScoredTransaction("txn-009", 1705334400000, 100.0, "retail", mapOf(), 0.40, 1705334401000),
            ScoredTransaction("txn-010", 1705334400000, 100.0, "retail", mapOf(), 0.45, 1705334401000)
        )
        
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        
        val capturedOutput = slot<RequestBody>()
        every { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) } returns PutObjectResponse.builder().build()
        
        every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } returns BatchWriteItemResponse.builder()
            .unprocessedItems(emptyMap())
            .build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes the scored transactions
        handler.handleRequest(input, mockContext)
        
        // Then the output should contain correct statistics
        val outputJson = String(capturedOutput.captured.contentStreamProvider().newStream().readAllBytes(), Charsets.UTF_8)
        val output = objectMapper.readTree(outputJson)
        
        output.get("totalTransactions").asInt() shouldBe 10
        output.get("successCount").asInt() shouldBe 10
        output.get("errorCount").asInt() shouldBe 0
        
        val riskDistribution = output.get("riskDistribution")
        riskDistribution.get("highRisk").asInt() shouldBe 2
        riskDistribution.get("mediumRisk").asInt() shouldBe 3
        riskDistribution.get("lowRisk").asInt() shouldBe 5
        
        // Average fraud score: (0.85 + 0.92 + 0.55 + 0.65 + 0.75 + 0.10 + 0.20 + 0.30 + 0.40 + 0.45) / 10 = 0.517
        val avgFraudScore = output.get("avgFraudScore").asDouble()
        avgFraudScore shouldBeGreaterThanOrEqual 0.51
        avgFraudScore shouldBeLessThan 0.53
    }
    
    test("should fail when scoredTransactions is missing from input") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        // Mock S3 input read with missing scoredTransactions
        val invalidOutput = objectMapper.createObjectNode().apply {
            put("batchDate", "2024-01-15")
            put("transactionCount", 0)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(invalidOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes invalid input
        val result = handler.handleRequest(input, mockContext)
        
        // Then the handler should return failure
        result.status shouldBe "FAILED"
        result.errorMessage shouldContain "scoredTransactions is required"
    }
    
    test("should fail when batchDate is missing from input") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val scoredTransactions = listOf(
            ScoredTransaction("txn-001", 1705334400000, 100.0, "retail", mapOf(), 0.5, 1705334401000)
        )
        
        // Mock S3 input read with missing batchDate
        val invalidOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("transactionCount", 1)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(invalidOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes invalid input
        val result = handler.handleRequest(input, mockContext)
        
        // Then the handler should return failure
        result.status shouldBe "FAILED"
        result.errorMessage shouldContain "batchDate is required"
    }
    
    test("should handle empty transaction batch") {
        val mockS3Client = mockk<S3Client>()
        val mockDynamoDbClient = mockk<DynamoDbClient>()
        
        val handler = StoreHandler(
            dynamoDbClient = mockDynamoDbClient,
            tableName = "FraudScores"
        )
        
        val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
        s3Field.isAccessible = true
        s3Field.set(handler, mockS3Client)
        
        val batchDate = "2024-01-15"
        
        // Mock S3 input read with empty transaction batch
        val scoreStageOutput = objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.createArrayNode()
            )
            put("batchDate", batchDate)
            put("transactionCount", 0)
            put("endpointName", "fraud-detection-prod")
        }
        
        val scoreStageStream = ResponseInputStream(
            GetObjectResponse.builder().build(),
            AbortableInputStream.create(
                ByteArrayInputStream(objectMapper.writeValueAsString(scoreStageOutput).toByteArray())
            )
        )
        
        every { mockS3Client.getObject(any<GetObjectRequest>()) } returns scoreStageStream
        every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
        
        val input = mapOf(
            "executionId" to "exec-test",
            "currentStage" to "StoreStage",
            "previousStage" to "ScoreStage",
            "workflowBucket" to "test-workflow-bucket"
        )
        
        val mockContext = mockk<Context>()
        
        // When the storage stage processes empty batch
        val result = handler.handleRequest(input, mockContext)
        
        // Then the handler should return success with zero counts
        result.status shouldBe "SUCCESS"
        
        // And DynamoDB batch write should not be called
        verify(exactly = 0) {
            mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>())
        }
    }
})
