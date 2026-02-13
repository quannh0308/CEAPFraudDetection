package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.collections.shouldContainAll
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
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
 * Property-based tests for DynamoDB write completeness in StoreHandler.
 * 
 * This test suite validates:
 * - Property 12: DynamoDB Write Completeness
 * 
 * **Requirements:**
 * - Requirement 7.2: Write each transaction record to DynamoDB with transaction ID as primary key
 * - Requirement 7.3: Store fraud score, timestamp, transaction amount, and relevant features
 */
class StoreHandlerDynamoDBWriteCompletenessPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 12: DynamoDB Write Completeness
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 12: DynamoDB Write Completeness
    test("Property 12: DynamoDB Write Completeness - each item SHALL contain all required attributes") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
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
            
            every { 
                mockS3Client.getObject(any<GetObjectRequest>()) 
            } returns scoreStageStream
            
            // Mock S3 output write
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            val writtenItems = mutableListOf<Map<String, AttributeValue>>()
            
            // Mock DynamoDB batch write
            every { 
                mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) 
            } answers {
                val request = firstArg<BatchWriteItemRequest>()
                val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
                
                // Capture all written items
                writeRequests.forEach { writeRequest ->
                    val item = writeRequest.putRequest().item()
                    writtenItems.add(item)
                }
                
                // Return success (no unprocessed items)
                BatchWriteItemResponse.builder()
                    .unprocessedItems(emptyMap())
                    .build()
            }
            
            // Create input for handler
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            
            // When the storage stage processes the scored transactions
            handler.handleRequest(input, mockContext)
            
            // Then each item SHALL contain all required attributes
            writtenItems.size shouldBe scoredTransactions.size
            
            val requiredAttributes = setOf(
                "transactionId",
                "timestamp",
                "batchDate",
                "amount",
                "merchantCategory",
                "fraudScore",
                "scoringTimestamp",
                "features"
            )
            
            writtenItems.forEach { item ->
                // Verify all required attributes are present
                item.keys shouldContainAll requiredAttributes
                
                // Verify attribute types
                item["transactionId"]?.s() shouldBe item["transactionId"]?.s() // String
                item["timestamp"]?.n() shouldBe item["timestamp"]?.n() // Number
                item["batchDate"]?.s() shouldBe item["batchDate"]?.s() // String
                item["amount"]?.n() shouldBe item["amount"]?.n() // Number
                item["merchantCategory"]?.s() shouldBe item["merchantCategory"]?.s() // String
                item["fraudScore"]?.n() shouldBe item["fraudScore"]?.n() // Number
                item["scoringTimestamp"]?.n() shouldBe item["scoringTimestamp"]?.n() // Number
                item["features"]?.s() shouldBe item["features"]?.s() // String (JSON)
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 12: DynamoDB Write Completeness
    test("Property 12: DynamoDB Write Completeness - transactionId SHALL match source transaction") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 30), Arb.batchDate()) { scoredTransactions, batchDate ->
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
            
            // Mock S3 input read
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
            
            val writtenItems = mutableListOf<Map<String, AttributeValue>>()
            
            every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } answers {
                val request = firstArg<BatchWriteItemRequest>()
                val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
                writeRequests.forEach { writtenItems.add(it.putRequest().item()) }
                BatchWriteItemResponse.builder().unprocessedItems(emptyMap()).build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then transactionId SHALL match the source transaction
            val writtenTransactionIds = writtenItems.map { it["transactionId"]?.s() }
            val sourceTransactionIds = scoredTransactions.map { it.transactionId }
            
            writtenTransactionIds shouldBe sourceTransactionIds
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 12: DynamoDB Write Completeness
    test("Property 12: DynamoDB Write Completeness - fraudScore SHALL match source transaction") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 30), Arb.batchDate()) { scoredTransactions, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockDynamoDbClient = mockk<DynamoDbClient>()
            
            val handler = StoreHandler(
                dynamoDbClient = mockDynamoDbClient,
                tableName = "FraudScores"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
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
            
            val writtenItems = mutableListOf<Map<String, AttributeValue>>()
            
            every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } answers {
                val request = firstArg<BatchWriteItemRequest>()
                val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
                writeRequests.forEach { writtenItems.add(it.putRequest().item()) }
                BatchWriteItemResponse.builder().unprocessedItems(emptyMap()).build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then fraudScore SHALL match the source transaction
            writtenItems.forEachIndexed { index, item ->
                val writtenScore = item["fraudScore"]?.n()?.toDouble()
                val sourceScore = scoredTransactions[index].fraudScore
                writtenScore shouldBe sourceScore
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 12: DynamoDB Write Completeness
    test("Property 12: DynamoDB Write Completeness - batchDate SHALL be included in all items") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 30), Arb.batchDate()) { scoredTransactions, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockDynamoDbClient = mockk<DynamoDbClient>()
            
            val handler = StoreHandler(
                dynamoDbClient = mockDynamoDbClient,
                tableName = "FraudScores"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
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
            
            val writtenItems = mutableListOf<Map<String, AttributeValue>>()
            
            every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } answers {
                val request = firstArg<BatchWriteItemRequest>()
                val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
                writeRequests.forEach { writtenItems.add(it.putRequest().item()) }
                BatchWriteItemResponse.builder().unprocessedItems(emptyMap()).build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then batchDate SHALL be included in all items
            writtenItems.forEach { item ->
                item["batchDate"]?.s() shouldBe batchDate
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 12: DynamoDB Write Completeness
    test("Property 12: DynamoDB Write Completeness - features SHALL be stored as JSON string") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 20), Arb.batchDate()) { scoredTransactions, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockDynamoDbClient = mockk<DynamoDbClient>()
            
            val handler = StoreHandler(
                dynamoDbClient = mockDynamoDbClient,
                tableName = "FraudScores"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
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
            
            val writtenItems = mutableListOf<Map<String, AttributeValue>>()
            
            every { mockDynamoDbClient.batchWriteItem(any<BatchWriteItemRequest>()) } answers {
                val request = firstArg<BatchWriteItemRequest>()
                val writeRequests = request.requestItems()["FraudScores"] ?: emptyList()
                writeRequests.forEach { writtenItems.add(it.putRequest().item()) }
                BatchWriteItemResponse.builder().unprocessedItems(emptyMap()).build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "StoreStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Then features SHALL be stored as valid JSON string
            writtenItems.forEachIndexed { index, item ->
                val featuresJson = item["features"]?.s()
                featuresJson shouldBe objectMapper.writeValueAsString(scoredTransactions[index].features)
                
                // Verify it's valid JSON that can be parsed back
                val parsedFeatures = objectMapper.readValue(featuresJson, Map::class.java)
                parsedFeatures shouldBe scoredTransactions[index].features
            }
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary scored transaction batches with configurable size range.
 */
private fun Arb.Companion.scoredTransactionBatch(minSize: Int = 0, maxSize: Int = 100): Arb<List<ScoredTransaction>> {
    return arbitrary { rs ->
        val size = Arb.int(minSize..maxSize).bind()
        List(size) { Arb.scoredTransaction().bind() }
    }
}

/**
 * Generates arbitrary ScoredTransaction instances.
 */
private fun Arb.Companion.scoredTransaction(): Arb<ScoredTransaction> {
    return arbitrary { rs ->
        ScoredTransaction(
            transactionId = Arb.transactionId().bind(),
            timestamp = Arb.long(1700000000000L..1800000000000L).bind(),
            amount = Arb.double(0.01, 10000.0).bind(),
            merchantCategory = Arb.merchantCategory().bind(),
            features = Arb.transactionFeatures().bind(),
            fraudScore = Arb.double(0.0, 1.0).bind(),
            scoringTimestamp = Arb.long(1700000000000L..1800000000000L).bind()
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
        features["Time"] = Arb.double(0.0, 172800.0).bind()
        
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
 * Generates arbitrary batch dates in YYYY-MM-DD format.
 */
private fun Arb.Companion.batchDate(): Arb<String> {
    return arbitrary { rs ->
        val year = Arb.int(2024..2025).bind()
        val month = Arb.int(1..12).bind()
        val day = Arb.int(1..28).bind() // Use 28 to avoid month-specific logic
        String.format("%04d-%02d-%02d", year, month, day)
    }
}
