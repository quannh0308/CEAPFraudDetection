package com.fraud.inference

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import com.fraud.common.models.ScoredTransaction
import software.amazon.awssdk.services.sns.SnsClient
import software.amazon.awssdk.services.sns.model.MessageAttributeValue
import software.amazon.awssdk.services.sns.model.PublishRequest

/**
 * Lambda handler for the alerting stage of the fraud detection inference pipeline.
 * 
 * This handler identifies high-risk transactions (fraud score >= 0.8) and sends
 * alerts via SNS, implementing batching logic to avoid rate limits.
 * 
 * **Workflow:**
 * 1. Reads scored transactions from S3 (output of ScoreStage)
 * 2. Filters high-risk transactions (fraud score >= 0.8)
 * 3. Batches alerts (max 100 transactions per SNS message)
 * 4. Builds structured alert messages with transaction details
 * 5. Publishes alerts to SNS topic
 * 6. Returns alert summary for monitoring
 * 
 * **Requirements:**
 * - Requirement 8.1: Read scored transactions from S3
 * - Requirement 8.2: Identify all high-risk transactions (fraud score >= 0.8)
 * - Requirement 8.3: Publish alert messages to SNS topic
 * - Requirement 8.4: Include transaction ID, fraud score, amount, and timestamp in each alert
 * - Requirement 8.5: Batch alerts to avoid SNS rate limits (maximum 100 alerts per message)
 * 
 * **Input Format** (from ScoreStage S3 output):
 * ```json
 * {
 *   "scoredTransactions": [
 *     {
 *       "transactionId": "txn-001",
 *       "timestamp": 1705334400000,
 *       "amount": 149.62,
 *       "merchantCategory": "retail",
 *       "features": { "Time": 0.0, "V1": -1.36, "Amount": 149.62 },
 *       "fraudScore": 0.0234,
 *       "scoringTimestamp": 1705334401000
 *     },
 *     {
 *       "transactionId": "txn-002",
 *       "timestamp": 1705334500000,
 *       "amount": 2500.00,
 *       "merchantCategory": "online",
 *       "features": { "Time": 100.0, "V1": 2.45, "Amount": 2500.00 },
 *       "fraudScore": 0.8912,
 *       "scoringTimestamp": 1705334501000
 *     }
 *   ],
 *   "batchDate": "2024-01-15",
 *   "transactionCount": 2,
 *   "endpointName": "fraud-detection-prod"
 * }
 * ```
 * 
 * **Output Format** (to S3 for MonitorStage):
 * ```json
 * {
 *   "batchDate": "2024-01-15",
 *   "highRiskCount": 1,
 *   "alertsSent": 1,
 *   "alertBatches": 1
 * }
 * ```
 */
open class AlertHandler(
    private val snsClient: SnsClient = SnsClient.builder().build(),
    private val fraudAlertTopicArn: String = System.getenv("FRAUD_ALERT_TOPIC_ARN") 
        ?: throw IllegalStateException("FRAUD_ALERT_TOPIC_ARN environment variable is required")
) : WorkflowLambdaHandler() {
    
    companion object {
        const val HIGH_RISK_THRESHOLD = 0.8
        const val MAX_ALERTS_PER_MESSAGE = 100
    }
    
    /**
     * Processes the alerting stage by identifying high-risk transactions and sending alerts.
     * 
     * This method:
     * 1. Extracts scored transactions from input
     * 2. Filters high-risk transactions (fraud score >= 0.8)
     * 3. Batches alerts (max 100 per message)
     * 4. Builds alert messages with transaction details
     * 5. Publishes alerts to SNS topic
     * 6. Returns alert summary
     * 
     * @param input Input data containing scored transactions
     * @return Output data containing alert summary
     * @throws IllegalArgumentException if required input fields are missing
     */
    override fun processData(input: JsonNode): JsonNode {
        val scoredTransactions = input.get("scoredTransactions")
            ?: throw IllegalArgumentException("scoredTransactions is required in input")
        
        val batchDate = input.get("batchDate")?.asText()
            ?: throw IllegalArgumentException("batchDate is required in input")
        
        logger.info("Processing ${scoredTransactions.size()} scored transactions for alerts")
        
        // 1. Filter high-risk transactions (fraud score >= 0.8)
        val highRiskTransactions = scoredTransactions.filter { transactionNode ->
            val transaction = objectMapper.treeToValue(transactionNode, ScoredTransaction::class.java)
            transaction.fraudScore >= HIGH_RISK_THRESHOLD
        }
        
        if (highRiskTransactions.isEmpty()) {
            logger.info("No high-risk transactions found for batch $batchDate")
            return objectMapper.createObjectNode().apply {
                put("batchDate", batchDate)
                put("highRiskCount", 0)
                put("alertsSent", 0)
                put("alertBatches", 0)
            }
        }
        
        logger.info("Found ${highRiskTransactions.size} high-risk transactions (fraud score >= $HIGH_RISK_THRESHOLD)")
        
        // 2. Batch alerts to avoid SNS rate limits (max 100 per message)
        val alertBatches = highRiskTransactions.chunked(MAX_ALERTS_PER_MESSAGE)
        var alertsSent = 0
        
        alertBatches.forEachIndexed { index, batch ->
            try {
                val alertMessage = buildAlertMessage(batch, batchDate)
                
                snsClient.publish(
                    PublishRequest.builder()
                        .topicArn(fraudAlertTopicArn)
                        .subject("Fraud Alert: ${batch.size} High-Risk Transactions Detected")
                        .message(alertMessage)
                        .messageAttributes(mapOf(
                            "batchDate" to MessageAttributeValue.builder()
                                .dataType("String")
                                .stringValue(batchDate)
                                .build(),
                            "highRiskCount" to MessageAttributeValue.builder()
                                .dataType("Number")
                                .stringValue(batch.size.toString())
                                .build(),
                            "batchIndex" to MessageAttributeValue.builder()
                                .dataType("Number")
                                .stringValue(index.toString())
                                .build()
                        ))
                        .build()
                )
                
                alertsSent += batch.size
                logger.info(
                    "Sent alert batch ${index + 1}/${alertBatches.size} " +
                    "for ${batch.size} high-risk transactions"
                )
                
            } catch (e: Exception) {
                logger.error(
                    "Failed to send alert batch ${index + 1}/${alertBatches.size}: ${e.message}. " +
                    "Batch size: ${batch.size}",
                    e
                )
                // Continue processing remaining batches even if one fails
            }
        }
        
        logger.info(
            "Alert processing complete. " +
            "High-risk transactions: ${highRiskTransactions.size}, " +
            "Alerts sent: $alertsSent, " +
            "Batches: ${alertBatches.size}"
        )
        
        // 3. Return alert summary
        return objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("highRiskCount", highRiskTransactions.size)
            put("alertsSent", alertsSent)
            put("alertBatches", alertBatches.size)
        }
    }
    
    /**
     * Builds a structured alert message for a batch of high-risk transactions.
     * 
     * The message includes:
     * - Batch date and transaction count
     * - For each transaction: ID, amount, fraud score, timestamp, merchant category
     * 
     * @param transactions List of high-risk transaction nodes
     * @param batchDate Batch date for context
     * @return Formatted alert message string
     */
    private fun buildAlertMessage(transactions: List<JsonNode>, batchDate: String): String {
        val sb = StringBuilder()
        sb.appendLine("High-Risk Fraud Transactions Detected")
        sb.appendLine("Batch Date: $batchDate")
        sb.appendLine("Count: ${transactions.size}")
        sb.appendLine()
        sb.appendLine("Transactions:")
        
        transactions.forEach { txnNode ->
            val txn = objectMapper.treeToValue(txnNode, ScoredTransaction::class.java)
            sb.appendLine("  - ID: ${txn.transactionId}")
            sb.appendLine("    Amount: \$${String.format("%.2f", txn.amount)}")
            sb.appendLine("    Fraud Score: ${String.format("%.4f", txn.fraudScore)}")
            sb.appendLine("    Timestamp: ${txn.timestamp}")
            sb.appendLine("    Merchant: ${txn.merchantCategory}")
            sb.appendLine()
        }
        
        return sb.toString()
    }
}
