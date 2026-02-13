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
 * Property-based tests for alert message completeness in AlertHandler.
 * 
 * This test suite validates:
 * - Property 14: Alert Message Completeness
 * 
 * **Requirements:**
 * - Requirement 8.4: Include transaction ID, fraud score, amount, and timestamp in each alert
 */
class AlertHandlerAlertMessageCompletenessPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 14: Alert Message Completeness
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 14: Alert Message Completeness
    test("Property 14: Alert Message Completeness - for any high-risk transaction, the alert message SHALL contain transaction ID, fraud score, amount, and timestamp") {
        checkAll(100, Arb.highRiskScoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
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
            
            // Then for each high-risk transaction, the alert message SHALL contain all required fields
            val allAlertText = publishedMessages.joinToString("\n")
            
            scoredTransactions.forEach { txn ->
                // Transaction ID SHALL be present
                allAlertText.contains(txn.transactionId) shouldBe true
                
                // Fraud score SHALL be present (formatted to 4 decimal places)
                val formattedScore = String.format("%.4f", txn.fraudScore)
                allAlertText.contains(formattedScore) shouldBe true
                
                // Amount SHALL be present (formatted to 2 decimal places)
                val formattedAmount = String.format("%.2f", txn.amount)
                allAlertText.contains(formattedAmount) shouldBe true
                
                // Timestamp SHALL be present
                allAlertText.contains(txn.timestamp.toString()) shouldBe true
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 14: Alert Message Completeness
    test("Property 14: Alert Message Completeness - alert message SHALL contain merchant category for each transaction") {
        checkAll(100, Arb.highRiskScoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
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
            
            // Then for each high-risk transaction, the merchant category SHALL be present
            val allAlertText = publishedMessages.joinToString("\n")
            
            scoredTransactions.forEach { txn ->
                allAlertText.contains(txn.merchantCategory) shouldBe true
            }
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 14: Alert Message Completeness
    test("Property 14: Alert Message Completeness - alert message SHALL contain batch date") {
        checkAll(100, Arb.highRiskScoredTransactionBatch(minSize = 1, maxSize = 50), Arb.batchDate()) { scoredTransactions, batchDate ->
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
            
            // Then the batch date SHALL be present in the alert message
            val allAlertText = publishedMessages.joinToString("\n")
            allAlertText.contains(batchDate) shouldBe true
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary high-risk scored transaction batches (fraud score >= 0.8).
 */
private fun Arb.Companion.highRiskScoredTransactionBatch(minSize: Int = 0, maxSize: Int = 100): Arb<List<ScoredTransaction>> {
    return arbitrary { rs ->
        val size = Arb.int(minSize..maxSize).bind()
        List(size) { index ->
            ScoredTransaction(
                transactionId = "txn-${java.util.UUID.randomUUID()}",
                timestamp = Arb.long(1700000000000L..1800000000000L).bind(),
                amount = Arb.double(0.01, 10000.0).bind(),
                merchantCategory = Arb.merchantCategory().bind(),
                features = Arb.transactionFeatures().bind(),
                fraudScore = Arb.double(0.8, 1.0).bind(), // High-risk only
                scoringTimestamp = Arb.long(1700000000000L..1800000000000L).bind()
            )
        }
    }
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
