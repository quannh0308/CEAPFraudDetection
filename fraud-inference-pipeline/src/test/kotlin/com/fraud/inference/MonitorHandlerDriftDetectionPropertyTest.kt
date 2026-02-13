package com.fraud.inference

import com.amazonaws.services.lambda.runtime.Context
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
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
import software.amazon.awssdk.services.s3.model.NoSuchKeyException
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectResponse
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.PublishRequest
import software.amazon.awssdk.services.sns.model.PublishResponse
import java.io.ByteArrayInputStream
import kotlin.math.abs

/**
 * Property-based tests for distribution drift detection in MonitorHandler.
 * 
 * This test suite validates:
 * - Property 16: Distribution Drift Detection
 * 
 * **Requirements:**
 * - Requirement 14.2: Compare fraud score distribution to historical baselines
 */
class MonitorHandlerDriftDetectionPropertyTest : FunSpec({
    val objectMapper: ObjectMapper = jacksonObjectMapper()
    
    // ========================================
    // Property 16: Distribution Drift Detection
    // ========================================
    
    // Feature: fraud-detection-ml-pipeline, Property 16: Distribution Drift Detection
    test("Property 16: Distribution Drift Detection - drift SHALL be detected when avgScoreDrift > 0.1") {
        checkAll(100, Arb.performanceMetrics(), Arb.batchDate()) { currentMetrics, batchDate ->
            // Create baseline with avgFraudScore that differs by more than 0.1
            val baselineAvgScore = if (currentMetrics.avgFraudScore > 0.5) {
                currentMetrics.avgFraudScore - 0.11 // Drift > 0.1
            } else {
                currentMetrics.avgFraudScore + 0.11 // Drift > 0.1
            }
            
            val baseline = PerformanceMetrics(
                avgFraudScore = baselineAvgScore,
                highRiskPct = currentMetrics.highRiskPct // Same high risk %
            )
            
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            // Replace S3 client with mock
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            // Mock S3 input read (from StoreStage)
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { 
                mockS3Client.getObject(any<GetObjectRequest>()) 
            } returns storeStageStream
            
            // Mock historical metrics loading (return baseline)
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            // Mock S3 output write (for both workflow output and metrics)
            every { 
                mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) 
            } returns PutObjectResponse.builder().build()
            
            every {
                mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>())
            } returns PutObjectResponse.builder().build()
            
            val publishedAlerts = mutableListOf<String>()
            
            // Mock SNS publish
            every { 
                mockSnsClient.publish(any<PublishRequest>()) 
            } answers {
                val request = firstArg<PublishRequest>()
                publishedAlerts.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
            }
            
            // Create input for handler
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            
            // When the monitoring stage processes the metrics
            val result = handler.handleRequest(input, mockContext)
            
            // Then drift SHALL be detected
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            val driftDetected = resultNode.get("driftDetected").asBoolean()
            val avgScoreDrift = resultNode.get("avgScoreDrift").asDouble()
            
            driftDetected shouldBe true
            avgScoreDrift shouldBe abs(currentMetrics.avgFraudScore - baseline.avgFraudScore)
            
            // And a drift alert SHALL be sent
            publishedAlerts.size shouldBe 1
            publishedAlerts[0].contains("Model Distribution Drift Detected") shouldBe true
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 16: Distribution Drift Detection
    test("Property 16: Distribution Drift Detection - drift SHALL be detected when highRiskDrift > 0.05") {
        checkAll(100, Arb.performanceMetrics(), Arb.batchDate()) { currentMetrics, batchDate ->
            // Create baseline with highRiskPct that differs by more than 0.05
            val baselineHighRiskPct = if (currentMetrics.highRiskPct > 0.1) {
                currentMetrics.highRiskPct - 0.06 // Drift > 0.05
            } else {
                currentMetrics.highRiskPct + 0.06 // Drift > 0.05
            }
            
            val baseline = PerformanceMetrics(
                avgFraudScore = currentMetrics.avgFraudScore, // Same avg score
                highRiskPct = baselineHighRiskPct
            )
            
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            val publishedAlerts = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedAlerts.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Then drift SHALL be detected
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            val driftDetected = resultNode.get("driftDetected").asBoolean()
            val highRiskDrift = resultNode.get("highRiskDrift").asDouble()
            
            driftDetected shouldBe true
            highRiskDrift shouldBe abs(currentMetrics.highRiskPct - baseline.highRiskPct)
            
            // And a drift alert SHALL be sent
            publishedAlerts.size shouldBe 1
            publishedAlerts[0].contains("Model Distribution Drift Detected") shouldBe true
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 16: Distribution Drift Detection
    test("Property 16: Distribution Drift Detection - drift SHALL NOT be detected when avgScoreDrift <= 0.1 AND highRiskDrift <= 0.05") {
        checkAll(100, Arb.performanceMetrics(), Arb.batchDate()) { currentMetrics, batchDate ->
            // Create baseline with small differences (within thresholds)
            val baseline = PerformanceMetrics(
                avgFraudScore = currentMetrics.avgFraudScore + 0.05, // Drift = 0.05 <= 0.1
                highRiskPct = currentMetrics.highRiskPct + 0.02 // Drift = 0.02 <= 0.05
            )
            
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            val publishedAlerts = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedAlerts.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Then drift SHALL NOT be detected
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            val driftDetected = resultNode.get("driftDetected").asBoolean()
            
            driftDetected shouldBe false
            
            // And NO drift alert SHALL be sent
            publishedAlerts.size shouldBe 0
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 16: Distribution Drift Detection
    test("Property 16: Distribution Drift Detection - boundary case avgScoreDrift = 0.1 SHALL NOT trigger drift") {
        checkAll(100, Arb.performanceMetrics(), Arb.batchDate()) { currentMetrics, batchDate ->
            // Create baseline with exactly 0.1 difference
            val baseline = PerformanceMetrics(
                avgFraudScore = currentMetrics.avgFraudScore + 0.1, // Drift = 0.1 (not > 0.1)
                highRiskPct = currentMetrics.highRiskPct // Same high risk %
            )
            
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            val publishedAlerts = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedAlerts.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Then drift SHALL NOT be detected (boundary is not inclusive)
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            val driftDetected = resultNode.get("driftDetected").asBoolean()
            
            driftDetected shouldBe false
            publishedAlerts.size shouldBe 0
        }
    }
    
    // Feature: fraud-detection-ml-pipeline, Property 16: Distribution Drift Detection
    test("Property 16: Distribution Drift Detection - boundary case highRiskDrift = 0.05 SHALL NOT trigger drift") {
        checkAll(100, Arb.performanceMetrics(), Arb.batchDate()) { currentMetrics, batchDate ->
            // Create baseline with exactly 0.05 difference
            val baseline = PerformanceMetrics(
                avgFraudScore = currentMetrics.avgFraudScore, // Same avg score
                highRiskPct = currentMetrics.highRiskPct + 0.05 // Drift = 0.05 (not > 0.05)
            )
            
            val mockS3Client = mockk<S3Client>()
            val mockMetricsS3Client = mockk<S3Client>()
            val mockSnsClient = mockk<SnsClient>()
            
            val handler = MonitorHandler(
                metricsS3Client = mockMetricsS3Client,
                snsClient = mockSnsClient,
                metricsBucket = "test-metrics-bucket",
                monitoringAlertTopicArn = "arn:aws:sns:us-east-1:123456789012:monitoring-alerts"
            )
            
            val s3Field = handler.javaClass.superclass.getDeclaredField("s3Client")
            s3Field.isAccessible = true
            s3Field.set(handler, mockS3Client)
            
            val storeStageOutput = objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("totalTransactions", 100)
                put("successCount", 100)
                put("errorCount", 0)
                set<com.fasterxml.jackson.databind.node.ObjectNode>(
                    "riskDistribution",
                    objectMapper.createObjectNode().apply {
                        put("highRisk", (currentMetrics.highRiskPct * 100).toInt())
                        put("mediumRisk", 20)
                        put("lowRisk", 80 - (currentMetrics.highRiskPct * 100).toInt())
                    }
                )
                put("avgFraudScore", currentMetrics.avgFraudScore)
            }
            
            val storeStageStream = ResponseInputStream(
                software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                AbortableInputStream.create(
                    ByteArrayInputStream(objectMapper.writeValueAsString(storeStageOutput).toByteArray())
                )
            )
            
            every { mockS3Client.getObject(any<GetObjectRequest>()) } returns storeStageStream
            
            every {
                mockMetricsS3Client.getObject(any<GetObjectRequest>())
            } answers {
                val historicalMetrics = objectMapper.createObjectNode().apply {
                    put("avgFraudScore", baseline.avgFraudScore)
                    put("highRiskPct", baseline.highRiskPct)
                }
                ResponseInputStream(
                    software.amazon.awssdk.services.s3.model.GetObjectResponse.builder().build(),
                    AbortableInputStream.create(
                        ByteArrayInputStream(objectMapper.writeValueAsString(historicalMetrics).toByteArray())
                    )
                )
            }
            
            every { mockS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            every { mockMetricsS3Client.putObject(any<PutObjectRequest>(), any<RequestBody>()) } returns PutObjectResponse.builder().build()
            
            val publishedAlerts = mutableListOf<String>()
            
            every { mockSnsClient.publish(any<PublishRequest>()) } answers {
                val request = firstArg<PublishRequest>()
                publishedAlerts.add(request.message())
                PublishResponse.builder().messageId("msg-${publishedAlerts.size}").build()
            }
            
            val input = mapOf(
                "executionId" to "exec-test",
                "currentStage" to "MonitorStage",
                "previousStage" to "StoreStage",
                "workflowBucket" to "test-workflow-bucket"
            )
            
            val mockContext = mockk<Context>()
            val result = handler.handleRequest(input, mockContext)
            
            // Then drift SHALL NOT be detected (boundary is not inclusive)
            val capturedOutput = slot<RequestBody>()
            verify { mockS3Client.putObject(any<PutObjectRequest>(), capture(capturedOutput)) }
            
            val outputJson = capturedOutput.captured.contentStreamProvider().newStream().readAllBytes().decodeToString()
            val resultNode = objectMapper.readTree(outputJson)
            
            val driftDetected = resultNode.get("driftDetected").asBoolean()
            
            driftDetected shouldBe false
            publishedAlerts.size shouldBe 0
        }
    }
})

// ========================================
// Arbitrary Generators for Property Tests
// ========================================

/**
 * Generates arbitrary performance metrics.
 */
private fun Arb.Companion.performanceMetrics(): Arb<PerformanceMetrics> {
    return arbitrary { rs ->
        PerformanceMetrics(
            avgFraudScore = Arb.double(0.0, 1.0).bind(),
            highRiskPct = Arb.double(0.0, 0.5).bind() // 0-50% high risk
        )
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

/**
 * Performance metrics data class for testing.
 */
private data class PerformanceMetrics(
    val avgFraudScore: Double,
    val highRiskPct: Double
)
