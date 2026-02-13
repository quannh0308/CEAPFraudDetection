package com.fraud.inference

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import com.fraud.common.models.ScoredTransaction
import software.amazon.awssdk.services.dynamodb.DynamoDbClient
import software.amazon.awssdk.services.dynamodb.model.AttributeValue
import software.amazon.awssdk.services.dynamodb.model.BatchWriteItemRequest
import software.amazon.awssdk.services.dynamodb.model.PutRequest
import software.amazon.awssdk.services.dynamodb.model.WriteRequest

/**
 * Lambda handler for the storage stage of the fraud detection inference pipeline.
 * 
 * This handler stores scored transactions in DynamoDB for querying and analysis,
 * implementing batch write logic with retry handling for unprocessed items.
 * 
 * **Workflow:**
 * 1. Reads scored transactions from S3 (output of ScoreStage)
 * 2. Batch writes to DynamoDB (max 25 items per batch)
 * 3. Handles unprocessed items with retry logic
 * 4. Calculates summary statistics (risk distribution, avg score)
 * 5. Returns storage summary for monitoring
 * 
 * **Requirements:**
 * - Requirement 7.1: Read scored transactions from S3
 * - Requirement 7.2: Write each transaction record to DynamoDB with transaction ID as primary key
 * - Requirement 7.3: Store fraud score, timestamp, transaction amount, and relevant features
 * - Requirement 7.5: Write summary statistics to S3
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
 */
open class StoreHandler(
    private val dynamoDbClient: DynamoDbClient = DynamoDbClient.builder().build(),
    private val tableName: String = System.getenv("DYNAMODB_TABLE") ?: "FraudScores"
) : WorkflowLambdaHandler() {
    
    companion object {
        const val MAX_BATCH_SIZE = 25 // DynamoDB batch write limit
        const val MAX_RETRY_ATTEMPTS = 3
    }
    
    /**
     * Processes the storage stage by writing scored transactions to DynamoDB.
     * 
     * This method:
     * 1. Extracts scored transactions from input
     * 2. Batch writes to DynamoDB (max 25 items per batch)
     * 3. Handles unprocessed items with retry logic
     * 4. Calculates summary statistics
     * 5. Returns storage summary
     * 
     * @param input Input data containing scored transactions
     * @return Output data containing storage summary and statistics
     * @throws IllegalArgumentException if required input fields are missing
     */
    override fun processData(input: JsonNode): JsonNode {
        val scoredTransactions = input.get("scoredTransactions")
            ?: throw IllegalArgumentException("scoredTransactions is required in input")
        
        val batchDate = input.get("batchDate")?.asText()
            ?: throw IllegalArgumentException("batchDate is required in input")
        
        logger.info("Storing ${scoredTransactions.size()} scored transactions to DynamoDB table: $tableName")
        
        // 1. Batch write to DynamoDB (max 25 items per batch)
        val writeRequests = mutableListOf<WriteRequest>()
        var successCount = 0
        var errorCount = 0
        val fraudScores = mutableListOf<Double>()
        
        scoredTransactions.forEach { transactionNode ->
            val transaction = objectMapper.treeToValue(transactionNode, ScoredTransaction::class.java)
            fraudScores.add(transaction.fraudScore)
            
            val item = buildDynamoDbItem(transaction, batchDate)
            
            writeRequests.add(
                WriteRequest.builder()
                    .putRequest(PutRequest.builder().item(item).build())
                    .build()
            )
            
            // Batch write when we reach 25 items
            if (writeRequests.size == MAX_BATCH_SIZE) {
                val result = batchWriteToDynamoDB(writeRequests)
                successCount += result.successCount
                errorCount += result.errorCount
                writeRequests.clear()
            }
        }
        
        // Write remaining items
        if (writeRequests.isNotEmpty()) {
            val result = batchWriteToDynamoDB(writeRequests)
            successCount += result.successCount
            errorCount += result.errorCount
        }
        
        logger.info(
            "DynamoDB batch write complete. " +
            "Success: $successCount, Errors: $errorCount, Total: ${scoredTransactions.size()}"
        )
        
        // 2. Calculate summary statistics
        val highRiskCount = fraudScores.count { it >= 0.8 }
        val mediumRiskCount = fraudScores.count { it >= 0.5 && it < 0.8 }
        val lowRiskCount = fraudScores.count { it < 0.5 }
        val avgFraudScore = if (fraudScores.isNotEmpty()) fraudScores.average() else 0.0
        
        logger.info(
            "Risk distribution - High: $highRiskCount, Medium: $mediumRiskCount, Low: $lowRiskCount. " +
            "Average fraud score: $avgFraudScore"
        )
        
        // 3. Return storage summary
        return objectMapper.createObjectNode().apply {
            put("batchDate", batchDate)
            put("totalTransactions", scoredTransactions.size())
            put("successCount", successCount)
            put("errorCount", errorCount)
            set<com.fasterxml.jackson.databind.node.ObjectNode>(
                "riskDistribution",
                objectMapper.createObjectNode().apply {
                    put("highRisk", highRiskCount)
                    put("mediumRisk", mediumRiskCount)
                    put("lowRisk", lowRiskCount)
                }
            )
            put("avgFraudScore", avgFraudScore)
        }
    }
    
    /**
     * Builds a DynamoDB item from a scored transaction.
     * 
     * @param transaction Scored transaction
     * @param batchDate Batch date for GSI queries
     * @return DynamoDB item map
     */
    private fun buildDynamoDbItem(
        transaction: ScoredTransaction,
        batchDate: String
    ): Map<String, AttributeValue> {
        return mapOf(
            "transactionId" to AttributeValue.builder().s(transaction.transactionId).build(),
            "timestamp" to AttributeValue.builder().n(transaction.timestamp.toString()).build(),
            "batchDate" to AttributeValue.builder().s(batchDate).build(),
            "amount" to AttributeValue.builder().n(transaction.amount.toString()).build(),
            "merchantCategory" to AttributeValue.builder().s(transaction.merchantCategory).build(),
            "fraudScore" to AttributeValue.builder().n(transaction.fraudScore.toString()).build(),
            "scoringTimestamp" to AttributeValue.builder().n(transaction.scoringTimestamp.toString()).build(),
            "features" to AttributeValue.builder().s(
                objectMapper.writeValueAsString(transaction.features)
            ).build()
        )
    }
    
    /**
     * Batch writes items to DynamoDB with retry logic for unprocessed items.
     * 
     * @param writeRequests List of write requests (max 25 items)
     * @return Batch write result with success and error counts
     */
    private fun batchWriteToDynamoDB(writeRequests: List<WriteRequest>): BatchWriteResult {
        var remainingRequests = writeRequests
        var successCount = 0
        var errorCount = 0
        var attempt = 0
        
        while (remainingRequests.isNotEmpty() && attempt < MAX_RETRY_ATTEMPTS) {
            attempt++
            
            try {
                logger.info(
                    "Batch write attempt $attempt: writing ${remainingRequests.size} items to DynamoDB"
                )
                
                val response = dynamoDbClient.batchWriteItem(
                    BatchWriteItemRequest.builder()
                        .requestItems(mapOf(tableName to remainingRequests))
                        .build()
                )
                
                val processedCount = remainingRequests.size
                val unprocessedItems = response.unprocessedItems()[tableName] ?: emptyList()
                val unprocessedCount = unprocessedItems.size
                
                successCount += (processedCount - unprocessedCount)
                
                if (unprocessedItems.isNotEmpty()) {
                    logger.warn(
                        "Batch write had $unprocessedCount unprocessed items. " +
                        "Will retry (attempt $attempt/$MAX_RETRY_ATTEMPTS)"
                    )
                    
                    // Exponential backoff before retry
                    if (attempt < MAX_RETRY_ATTEMPTS) {
                        val backoffMs = (100 * Math.pow(2.0, (attempt - 1).toDouble())).toLong()
                        logger.info("Backing off for ${backoffMs}ms before retry")
                        Thread.sleep(backoffMs)
                    }
                    
                    remainingRequests = unprocessedItems
                } else {
                    logger.info("Batch write successful: $processedCount items written")
                    remainingRequests = emptyList()
                }
                
            } catch (e: Exception) {
                logger.error(
                    "Batch write attempt $attempt failed: ${e.message}. " +
                    "Remaining items: ${remainingRequests.size}",
                    e
                )
                
                if (attempt >= MAX_RETRY_ATTEMPTS) {
                    errorCount += remainingRequests.size
                    logger.error(
                        "Max retry attempts reached. " +
                        "Failed to write ${remainingRequests.size} items to DynamoDB"
                    )
                    break
                }
                
                // Exponential backoff before retry
                val backoffMs = (100 * Math.pow(2.0, (attempt - 1).toDouble())).toLong()
                logger.info("Backing off for ${backoffMs}ms before retry")
                Thread.sleep(backoffMs)
            }
        }
        
        return BatchWriteResult(successCount, errorCount)
    }
    
    /**
     * Result of a batch write operation.
     * 
     * @property successCount Number of items successfully written
     * @property errorCount Number of items that failed to write
     */
    data class BatchWriteResult(
        val successCount: Int,
        val errorCount: Int
    )
}
