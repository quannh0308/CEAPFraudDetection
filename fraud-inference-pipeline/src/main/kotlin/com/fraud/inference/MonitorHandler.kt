package com.fraud.inference

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.NoSuchKeyException
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.MessageAttributeValue
import software.amazon.awssdk.services.sns.model.PublishRequest
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import kotlin.math.abs

/**
 * Lambda handler for the monitoring stage of the fraud detection inference pipeline.
 * 
 * This handler monitors model performance by detecting distribution drift in fraud scores,
 * comparing current batch metrics against historical baselines.
 * 
 * **Workflow:**
 * 1. Reads storage summary from S3 (output of StoreStage)
 * 2. Loads historical baseline from S3 metrics (last 30 days average)
 * 3. Calculates distribution metrics (high/medium/low risk percentages)
 * 4. Implements drift detection logic (avg score drift > 0.1 OR high risk drift > 0.05)
 * 5. Sends monitoring alert via SNS if drift detected
 * 6. Writes current metrics to S3 for historical tracking
 * 
 * **Requirements:**
 * - Requirement 14.1: Calculate daily performance metrics
 * - Requirement 14.2: Compare fraud score distribution to historical baselines
 * - Requirement 14.3: Write performance metrics to S3 for analysis
 * - Requirement 14.4: Send monitoring alert if drift detected
 * 
 * **Input Format** (from StoreStage S3 output):
 * ```json
 * {
 *   "batchDate": "2024-01-15",
 *   "totalTransactions": 2,
 *   "successCount": 2,
 *   "errorCount": 0,
 *   "riskDistribution": {
 *     "highRisk": 1,
 *     "mediumRisk": 0,
 *     "lowRisk": 1
 *   },
 *   "avgFraudScore": 0.4573
 * }
 * ```
 * 
 * **Output Format** (to S3):
 * ```json
 * {
 *   "batchDate": "2024-01-15",
 *   "avgFraudScore": 0.4573,
 *   "avgScoreDrift": 0.0234,
 *   "highRiskDrift": 0.0123,
 *   "driftDetected": false,
 *   "metricsPath": "metrics/2024-01-15.json"
 * }
 * ```
 */
open class MonitorHandler(
    private val metricsS3Client: S3Client = S3Client.builder().build(),
    private val snsClient: SnsClient = SnsClient.builder().build(),
    private val metricsBucket: String = System.getenv("METRICS_BUCKET")
        ?: throw IllegalStateException("METRICS_BUCKET environment variable is required"),
    private val monitoringAlertTopicArn: String = System.getenv("MONITORING_ALERT_TOPIC_ARN")
        ?: throw IllegalStateException("MONITORING_ALERT_TOPIC_ARN environment variable is required")
) : WorkflowLambdaHandler() {
    
    companion object {
        const val AVG_SCORE_DRIFT_THRESHOLD = 0.1
        const val HIGH_RISK_DRIFT_THRESHOLD = 0.05
        const val BASELINE_DAYS = 30
        private val DATE_FORMATTER = DateTimeFormatter.ISO_LOCAL_DATE
    }
    
    /**
     * Processes the monitoring stage by detecting distribution drift.
     * 
     * This method:
     * 1. Extracts metrics from storage summary
     * 2. Loads historical baseline (last 30 days)
     * 3. Calculates distribution metrics
     * 4. Detects drift (avg score drift > 0.1 OR high risk drift > 0.05)
     * 5. Sends alert if drift detected
     * 6. Writes metrics to S3
     * 
     * @param input Input data containing storage summary
     * @return Output data containing monitoring summary
     * @throws IllegalArgumentException if required input fields are missing
     */
    override fun processData(input: JsonNode): JsonNode {
        val batchDate = input.get("batchDate")?.asText()
            ?: throw IllegalArgumentException("batchDate is required in input")
        
        val avgFraudScore = input.get("avgFraudScore")?.asDouble()
            ?: throw IllegalArgumentException("avgFraudScore is required in input")
        
        val riskDistribution = input.get("riskDistribution")
            ?: throw IllegalArgumentException("riskDistribution is required in input")
        
        val totalTransactions = input.get("totalTransactions")?.asInt()
            ?: throw IllegalArgumentException("totalTransactions is required in input")
        
        logger.info("Monitoring batch $batchDate: avgFraudScore=$avgFraudScore, totalTransactions=$totalTransactions")
        
        // 1. Calculate distribution metrics
        val highRiskCount = riskDistribution.get("highRisk")?.asInt() ?: 0
        val mediumRiskCount = riskDistribution.get("mediumRisk")?.asInt() ?: 0
        val lowRiskCount = riskDistribution.get("lowRisk")?.asInt() ?: 0
        
        val highRiskPct = if (totalTransactions > 0) highRiskCount.toDouble() / totalTransactions else 0.0
        val mediumRiskPct = if (totalTransactions > 0) mediumRiskCount.toDouble() / totalTransactions else 0.0
        val lowRiskPct = if (totalTransactions > 0) lowRiskCount.toDouble() / totalTransactions else 0.0
        
        logger.info(
            "Distribution metrics - " +
            "High: ${String.format("%.2f%%", highRiskPct * 100)}, " +
            "Medium: ${String.format("%.2f%%", mediumRiskPct * 100)}, " +
            "Low: ${String.format("%.2f%%", lowRiskPct * 100)}"
        )
        
        // 2. Load historical baseline from S3 metrics
        val baseline = loadHistoricalBaseline(batchDate)
        
        // 3. Compare to baseline (detect drift)
        val avgScoreDrift = abs(avgFraudScore - baseline.avgFraudScore)
        val highRiskDrift = abs(highRiskPct - baseline.highRiskPct)
        
        val driftDetected = avgScoreDrift > AVG_SCORE_DRIFT_THRESHOLD || highRiskDrift > HIGH_RISK_DRIFT_THRESHOLD
        
        logger.info(
            "Drift analysis - " +
            "Avg score drift: ${String.format("%.4f", avgScoreDrift)} (threshold: $AVG_SCORE_DRIFT_THRESHOLD), " +
            "High risk drift: ${String.format("%.4f", highRiskDrift)} (threshold: $HIGH_RISK_DRIFT_THRESHOLD), " +
            "Drift detected: $driftDetected"
        )
        
        // 4. Send monitoring alert if drift detected
        if (driftDetected) {
            sendDriftAlert(batchDate, avgFraudScore, baseline.avgFraudScore, highRiskPct, baseline.highRiskPct, avgScoreDrift, highRiskDrift)
            logger.warn("Model drift detected for batch $batchDate")
        } else {
            logger.info("No drift detected for batch $batchDate")
        }
        
        // 5. Write metrics to S3 for historical tracking
        val metricsPath = "metrics/$batchDate.json"
        writeMetricsToS3(batchDate, avgFraudScore, highRiskPct, mediumRiskPct, lowRiskPct, avgScoreDrift, highRiskDrift, driftDetected, metricsPath)
        
        logger.info("Monitoring complete for batch $batchDate. Metrics written to $metricsPath")
        
        // 6. Return monitoring summary
        return objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("avgFraudScore", avgFraudScore)
            put("avgScoreDrift", avgScoreDrift)
            put("highRiskDrift", highRiskDrift)
            put("driftDetected", driftDetected)
            put("metricsPath", metricsPath)
        }
    }
    
    /**
     * Loads historical baseline from S3 metrics (last 30 days average).
     * 
     * If fewer than 30 days of metrics are available, uses available data.
     * If no historical data exists (first run), returns default baseline.
     * 
     * @param currentBatchDate Current batch date to exclude from baseline
     * @return Performance baseline with average fraud score and high risk percentage
     */
    private fun loadHistoricalBaseline(currentBatchDate: String): PerformanceBaseline {
        val currentDate = LocalDate.parse(currentBatchDate, DATE_FORMATTER)
        val historicalMetrics = mutableListOf<DailyMetrics>()
        
        // Load last 30 days of metrics (excluding current date)
        for (daysAgo in 1..BASELINE_DAYS) {
            val date = currentDate.minusDays(daysAgo.toLong())
            val dateStr = date.format(DATE_FORMATTER)
            
            try {
                val metrics = loadMetricsForDate(dateStr)
                if (metrics != null) {
                    historicalMetrics.add(metrics)
                }
            } catch (e: NoSuchKeyException) {
                // Metrics don't exist for this date, skip
                logger.debug("No metrics found for date $dateStr")
            } catch (e: Exception) {
                logger.warn("Failed to load metrics for date $dateStr: ${e.message}")
            }
        }
        
        if (historicalMetrics.isEmpty()) {
            logger.info("No historical metrics found. Using default baseline for first run.")
            return PerformanceBaseline(
                avgFraudScore = 0.0,
                highRiskPct = 0.0
            )
        }
        
        val avgFraudScore = historicalMetrics.map { it.avgFraudScore }.average()
        val highRiskPct = historicalMetrics.map { it.highRiskPct }.average()
        
        logger.info(
            "Loaded baseline from ${historicalMetrics.size} days of historical data. " +
            "Baseline avg fraud score: ${String.format("%.4f", avgFraudScore)}, " +
            "Baseline high risk %: ${String.format("%.2f%%", highRiskPct * 100)}"
        )
        
        return PerformanceBaseline(avgFraudScore, highRiskPct)
    }
    
    /**
     * Loads metrics for a specific date from S3.
     * 
     * @param date Date string in ISO format (YYYY-MM-DD)
     * @return Daily metrics or null if not found
     */
    private fun loadMetricsForDate(date: String): DailyMetrics? {
        return try {
            val response = metricsS3Client.getObject(
                GetObjectRequest.builder()
                    .bucket(metricsBucket)
                    .key("metrics/$date.json")
                    .build()
            )
            
            val metricsJson = objectMapper.readTree(response.readAllBytes())
            DailyMetrics(
                avgFraudScore = metricsJson.get("avgFraudScore").asDouble(),
                highRiskPct = metricsJson.get("highRiskPct").asDouble()
            )
        } catch (e: NoSuchKeyException) {
            null
        }
    }
    
    /**
     * Sends a drift alert via SNS when model drift is detected.
     * 
     * @param batchDate Batch date
     * @param currentAvgScore Current average fraud score
     * @param baselineAvgScore Baseline average fraud score
     * @param currentHighRiskPct Current high risk percentage
     * @param baselineHighRiskPct Baseline high risk percentage
     * @param avgScoreDrift Average score drift magnitude
     * @param highRiskDrift High risk percentage drift magnitude
     */
    private fun sendDriftAlert(
        batchDate: String,
        currentAvgScore: Double,
        baselineAvgScore: Double,
        currentHighRiskPct: Double,
        baselineHighRiskPct: Double,
        avgScoreDrift: Double,
        highRiskDrift: Double
    ) {
        val message = buildDriftAlertMessage(
            batchDate,
            currentAvgScore,
            baselineAvgScore,
            currentHighRiskPct,
            baselineHighRiskPct,
            avgScoreDrift,
            highRiskDrift
        )
        
        try {
            snsClient.publish(
                PublishRequest.builder()
                    .topicArn(monitoringAlertTopicArn)
                    .subject("Model Drift Detected: $batchDate")
                    .message(message)
                    .messageAttributes(mapOf(
                        "batchDate" to MessageAttributeValue.builder()
                            .dataType("String")
                            .stringValue(batchDate)
                            .build(),
                        "avgScoreDrift" to MessageAttributeValue.builder()
                            .dataType("Number")
                            .stringValue(avgScoreDrift.toString())
                            .build(),
                        "highRiskDrift" to MessageAttributeValue.builder()
                            .dataType("Number")
                            .stringValue(highRiskDrift.toString())
                            .build()
                    ))
                    .build()
            )
            
            logger.info("Drift alert sent successfully for batch $batchDate")
        } catch (e: Exception) {
            logger.error("Failed to send drift alert for batch $batchDate: ${e.message}", e)
            // Don't fail the stage if alert fails
        }
    }
    
    /**
     * Builds a structured drift alert message.
     * 
     * @param batchDate Batch date
     * @param currentAvgScore Current average fraud score
     * @param baselineAvgScore Baseline average fraud score
     * @param currentHighRiskPct Current high risk percentage
     * @param baselineHighRiskPct Baseline high risk percentage
     * @param avgScoreDrift Average score drift magnitude
     * @param highRiskDrift High risk percentage drift magnitude
     * @return Formatted alert message string
     */
    private fun buildDriftAlertMessage(
        batchDate: String,
        currentAvgScore: Double,
        baselineAvgScore: Double,
        currentHighRiskPct: Double,
        baselineHighRiskPct: Double,
        avgScoreDrift: Double,
        highRiskDrift: Double
    ): String {
        val sb = StringBuilder()
        sb.appendLine("Model Distribution Drift Detected")
        sb.appendLine("Batch Date: $batchDate")
        sb.appendLine()
        sb.appendLine("Average Fraud Score:")
        sb.appendLine("  Current: ${String.format("%.4f", currentAvgScore)}")
        sb.appendLine("  Baseline: ${String.format("%.4f", baselineAvgScore)}")
        sb.appendLine("  Drift: ${String.format("%.4f", avgScoreDrift)} (threshold: $AVG_SCORE_DRIFT_THRESHOLD)")
        sb.appendLine()
        sb.appendLine("High Risk Percentage:")
        sb.appendLine("  Current: ${String.format("%.2f%%", currentHighRiskPct * 100)}")
        sb.appendLine("  Baseline: ${String.format("%.2f%%", baselineHighRiskPct * 100)}")
        sb.appendLine("  Drift: ${String.format("%.4f", highRiskDrift)} (threshold: $HIGH_RISK_DRIFT_THRESHOLD)")
        sb.appendLine()
        sb.appendLine("Action Required:")
        sb.appendLine("  - Review model performance")
        sb.appendLine("  - Consider retraining if drift persists")
        sb.appendLine("  - Investigate data quality issues")
        
        return sb.toString()
    }
    
    /**
     * Writes current metrics to S3 for historical tracking.
     * 
     * @param batchDate Batch date
     * @param avgFraudScore Average fraud score
     * @param highRiskPct High risk percentage
     * @param mediumRiskPct Medium risk percentage
     * @param lowRiskPct Low risk percentage
     * @param avgScoreDrift Average score drift
     * @param highRiskDrift High risk drift
     * @param driftDetected Whether drift was detected
     * @param metricsPath S3 path for metrics
     */
    private fun writeMetricsToS3(
        batchDate: String,
        avgFraudScore: Double,
        highRiskPct: Double,
        mediumRiskPct: Double,
        lowRiskPct: Double,
        avgScoreDrift: Double,
        highRiskDrift: Double,
        driftDetected: Boolean,
        metricsPath: String
    ) {
        val metrics = objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("avgFraudScore", avgFraudScore)
            put("highRiskPct", highRiskPct)
            put("mediumRiskPct", mediumRiskPct)
            put("lowRiskPct", lowRiskPct)
            put("avgScoreDrift", avgScoreDrift)
            put("highRiskDrift", highRiskDrift)
            put("driftDetected", driftDetected)
            put("timestamp", System.currentTimeMillis())
        }
        
        try {
            metricsS3Client.putObject(
                PutObjectRequest.builder()
                    .bucket(metricsBucket)
                    .key(metricsPath)
                    .build(),
                RequestBody.fromString(objectMapper.writeValueAsString(metrics))
            )
            
            logger.info("Metrics written to S3: s3://$metricsBucket/$metricsPath")
        } catch (e: Exception) {
            logger.error("Failed to write metrics to S3: ${e.message}", e)
            throw e
        }
    }
    
    /**
     * Performance baseline calculated from historical metrics.
     * 
     * @property avgFraudScore Average fraud score from historical data
     * @property highRiskPct High risk percentage from historical data
     */
    data class PerformanceBaseline(
        val avgFraudScore: Double,
        val highRiskPct: Double
    )
    
    /**
     * Daily metrics loaded from S3.
     * 
     * @property avgFraudScore Average fraud score for the day
     * @property highRiskPct High risk percentage for the day
     */
    data class DailyMetrics(
        val avgFraudScore: Double,
        val highRiskPct: Double
    )
}
