package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fraud.common.models.ScoredTransaction
import io.kotest.core.spec.style.FunSpec
import io.kotest.matchers.shouldBe
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import io.mockk.*
import software.amazon.awssdk.core.ResponseInputStream
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.http.AbortableInputStream
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectResponse
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.PublishRequest
import software.amazon.awssdk.services.sns.model.PublishResponse
import java.io.ByteArrayInputStream

/**
 * Property-based tests for high-risk transaction identification in AlertHandler.
 * 
 * This test suite validates:
 * - Property 13: High-Risk Transaction Identification
 * 
 * **Requirements:**
 * - Requirement 8.2: Identify all high-risk transactions (fraud score >= 0.8)
 */
class AlertHandlerHighRiskIdentificationPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 13: High-Risk Transaction Identification
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 13: High-Risk Transaction Identification
    test("Property 13: High-Risk Transaction Identification - transactions with fraudScore >= 0.8 SHALL be identified as high-risk") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
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
            
            val publishedMessages = mutableListOf<String>()
            
            // Mock SNS publish
            every { 
                mockSnsClient.publish(any<PublishRequest>()) 
            } answers {
                val request = firstArg<PublishRequest>()
                publishedMessages.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
            }
            
            // Create input for handler
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            
            // When the alerting stage processes the scored transactions
            val result = handler.handleRequest(input, mockContext)
            
            // Then the high-risk count SHALL match transactions with fraudScore >= 0.8
            val expectedHighRiskCount = scoredTransactions.count { it.fraudScore >= 0.8 }
            
            // Read the output from S3 (written by the handler)
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            val actualHighRiskCount = resultNode.get("highRiskCount").asInt()
            
            actualHighRiskCount shouldBe expectedHighRiskCount
            
            // And if there are high-risk transactions, alerts SHALL be sent
            if (expectedHighRiskCount > 0) {
                publishedMessages.size shouldBe ((expectedHighRiskCount - 1) / 100 + 1) // Number of batches
                
                // Verify each high-risk transaction appears in the alert messages
                val allAlertText = publishedMessages.joinToString("\n")
                scoredTransactions.filter { it.fraudScore >= 0.8 }.forEach { txn ->
                    allAlertText.contains(txn.transactionId) shouldBe true
                }
            } else {
                publishedMessages.size shouldBe 0
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 13: High-Risk Transaction Identification
    test("Property 13: High-Risk Transaction Identification - transactions with fraudScore < 0.8 SHALL NOT be identified as high-risk") {
        checkAll(100, Arb.scoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
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
            
            val publishedMessages = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedMessages.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            handler.handleRequest(input, mockContext)
            
            // Capture the output written to S3
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            
            // Then transactions with fraudScore < 0.8 SHALL NOT appear in alert messages
            val lowRiskTransactions = scoredTransactions.filter { it.fraudScore < 0.8 }
            
            // Only check if there are alert messages (i.e., there were high-risk transactions)
            if (publishedMessages.isNotEmpty()) {
                val allAlertText = publishedMessages.joinToString("\n")
                // Low-risk transaction IDs should NOT appear in any alert message
                lowRiskTransactions.forEach { txn ->
                    allAlertText.contains(txn.transactionId) shouldBe false
                }
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 13: High-Risk Transaction Identification
    test("Property 13: High-Risk Transaction Identification - boundary case fraudScore = 0.8 SHALL be identified as high-risk") {
        checkAll(100, Arb.int(1..50), Arb.batchDate()) { transactionCount, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            // Create transactions with exactly 0.8 fraud score
            val scoredTransactions = List(transactionCount) {
                ScoredTransaction(
                    transactionId = "txn-boundary-$it",
                    timestamp = System.currentTimeMillis(),
                    amount = 100.0,
                    merchantCategory = "retail",
                    features = mapOf("V1" to 1.0),
                    fraudScore = 0.8, // Exactly at threshold
                    scoringTimestamp = System.currentTimeMillis()
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
            
            val publishedMessages = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedMessages.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Capture the output written to S3
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            // Then all transactions SHALL be identified as high-risk
            val actualHighRiskCount = resultNode.get("highRiskCount").asInt()
            
            actualHighRiskCount shouldBe transactionCount
            
            // And alerts SHALL be sent
            publishedMessages.size shouldBe ((transactionCount - 1) / 100 + 1)
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 13: High-Risk Transaction Identification
    test("Property 13: High-Risk Transaction Identification - boundary case fraudScore = 0.7999 SHALL NOT be identified as high-risk") {
        checkAll(100, Arb.int(1..50), Arb.batchDate()) { transactionCount, batchDate ->
            val mockS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = AlertHandler(
                snsClient = mockSnsClient,
                fraudAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:fraud-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            // Create transactions with 0.7999 fraud score (just below threshold)
            val scoredTransactions = List(transactionCount) {
                ScoredTransaction(
                    transactionId = "txn-below-$it",
                    timestamp = System.currentTimeMillis(),
                    amount = 100.0,
                    merchantCategory = "retail",
                    features = mapOf("V1" to 1.0),
                    fraudScore = 0.7999, // Just below threshold
                    scoringTimestamp = System.currentTimeMillis()
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
            
            val publishedMessages = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedMessages.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedMessages.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "AlertStage",
                "previousStage" to "ScoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Capture the output written to S3
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            // Then NO transactions SHALL be identified as high-risk
            val actualHighRiskCount = resultNode.get("highRiskCount").asInt()
            
            actualHighRiskCount shouldBe 0
            
            // And NO alerts SHALL be sent
            publishedMessages.size shouldBe 0
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
        List(size) { index ->
            ScoredTransaction(
                transactionId = "txn-${java.util.UUID.randomUUID()}",
                timestamp = Arb.long(1700000000000L..1800000000000L).bind(),
                amount = Arb.double(0.01, 10000.0).bind(),
                merchantCategory = Arb.merchantCategory().bind(),
                features = Arb.transactionFeatures().bind(),
                fraudScore = Arb.double(0.0, 1.0).bind(),
                scoringTimestamp = Arb.long(1700000000000L..1800000000000L).bind()
            )
        }
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
 * Generates arbitrary transaction IDs using UUIDs to ensure uniqueness.
 */
private fun Arb.Companion.transactionId(): Arb<String> {
    return Arb.uuid().map { "txn-$it" }
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
