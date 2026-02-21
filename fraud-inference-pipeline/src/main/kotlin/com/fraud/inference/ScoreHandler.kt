package com.fraud.inference

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import com.fraud.common.models.ScoredTransaction
import com.fraud.common.models.Transaction
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest

/**
 * Lambda handler for the scoring stage of the fraud detection inference pipeline.
 * 
 * This handler scores daily transaction batches using the deployed SageMaker endpoint,
 * creating ScoredTransaction objects with fraud scores for downstream processing.
 * 
 * **Workflow:**
 * 1. Reads current endpoint name from config bucket (written by DeployHandler)
 * 2. Loads transaction batch from S3
 * 3. Invokes SageMaker endpoint for each transaction
 * 4. Creates ScoredTransaction objects with fraud scores
 * 5. Returns scored transactions for next stage
 * 
 * **Requirements:**
 * - Requirement 6.1: Load daily transaction batch from S3
 * - Requirement 6.2: Read the current SageMaker_Endpoint name from S3
 * - Requirement 6.3: Invoke the SageMaker_Endpoint for each transaction
 * - Requirement 6.4: Receive a Fraud_Score between 0.0 and 1.0 for each transaction
 * - Requirement 6.5: Write scored transactions with metadata to S3 for the next stage
 * 
 * **Input Format** (from initialData):
 * ```json
 * {
 *   "transactionBatchPath": "s3://fraud-detection-data/daily-batches/2024-01-15.json",
 *   "batchDate": "2024-01-15"
 * }
 * ```
 * 
 * **Output Format** (to S3 for StoreStage):
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
 */
open class ScoreHandler(
    private val sageMakerRuntimeClient: SageMakerRuntimeClient = SageMakerRuntimeClient.builder().build(),
    private val configBucket: String = System.getenv("CONFIG_BUCKET") ?: "fraud-detection-config"
) : WorkflowLambdaHandler() {
    
    companion object {
        const val CONFIG_KEY = "current-endpoint.json"
    }
    
    /**
     * Processes the scoring stage by loading transactions and scoring them with SageMaker.
     * 
     * This method:
     * 1. Reads current endpoint name from config bucket
     * 2. Loads transaction batch from S3
     * 3. Scores each transaction using the SageMaker endpoint
     * 4. Creates ScoredTransaction objects
     * 5. Returns scored transactions for next stage
     * 
     * @param input Input data containing transaction batch path and batch date
     * @return Output data containing scored transactions
     * @throws IllegalArgumentException if required input fields are missing
     * @throws IllegalStateException if endpoint metadata cannot be read or endpoint invocation fails
     */
    override fun processData(input: JsonNode): JsonNode {
        // 1. Read current endpoint name from config bucket
        val endpointName = readEndpointName()
        
        logger.info("Using SageMaker endpoint: $endpointName")
        
        // 2. Extract transaction batch path and batch date from input
        val transactionBatchPath = input.get("transactionBatchPath")?.asText()
            ?: throw IllegalArgumentException("transactionBatchPath is required in input")
        
        val batchDate = input.get("batchDate")?.asText()
            ?: throw IllegalArgumentException("batchDate is required in input")
        
        logger.info("Loading transaction batch from: $transactionBatchPath")
        
        // 3. Load transaction batch from S3
        val transactionBatch = loadTransactionBatch(transactionBatchPath)
        
        logger.info("Scoring ${transactionBatch.size} transactions using endpoint $endpointName")
        
        // 4. Score each transaction
        val scoredTransactions = transactionBatch.map { transaction ->
            val fraudScore = invokeEndpoint(endpointName, transaction.features)
            
            ScoredTransaction(
                transactionId = transaction.id,
                timestamp = transaction.timestamp,
                amount = transaction.amount,
                merchantCategory = transaction.merchantCategory,
                features = transaction.features,
                fraudScore = fraudScore,
                scoringTimestamp = System.currentTimeMillis()
            )
        }
        
        logger.info(
            "Successfully scored ${scoredTransactions.size} transactions. " +
            "High-risk: ${scoredTransactions.count { it.isHighRisk() }}, " +
            "Medium-risk: ${scoredTransactions.count { it.isMediumRisk() }}, " +
            "Low-risk: ${scoredTransactions.count { it.isLowRisk() }}"
        )
        
        // 5. Return scored transactions
        return objectMapper.createObjectNode().apply {
            set<com.fasterxml.jackson.databind.node.ArrayNode>(
                "scoredTransactions",
                objectMapper.valueToTree(scoredTransactions)
            )
            put("batchDate", batchDate)
            put("transactionCount", scoredTransactions.size)
            put("endpointName", endpointName)
        }
    }
    
    /**
     * Reads the current endpoint name from the config bucket.
     * 
     * The endpoint metadata is written by DeployHandler during model deployment.
     * 
     * @return Endpoint name
     * @throws IllegalStateException if endpoint metadata cannot be read
     */
    private fun readEndpointName(): String {
        logger.info("Reading endpoint metadata from S3: bucket=$configBucket, key=$CONFIG_KEY")
        
        try {
            val getObjectRequest = GetObjectRequest.builder()
                .bucket(configBucket)
                .key(CONFIG_KEY)
                .build()
            
            val responseBytes = s3Client.getObject(getObjectRequest).readAllBytes()
            val jsonString = String(responseBytes, Charsets.UTF_8)
            val endpointMetadata = objectMapper.readTree(jsonString)
            
            val endpointName = endpointMetadata.get("endpointName")?.asText()
                ?: throw IllegalStateException("endpointName not found in endpoint metadata")
            
            logger.info("Successfully read endpoint name: $endpointName")
            
            return endpointName
            
        } catch (e: Exception) {
            logger.error("Failed to read endpoint metadata from S3: ${e.message}", e)
            throw IllegalStateException(
                "Failed to read endpoint metadata from config bucket. " +
                "Ensure DeployHandler has successfully deployed a model. Error: ${e.message}",
                e
            )
        }
    }
    
    /**
     * Loads a transaction batch from S3.
     * 
     * @param s3Path S3 path to transaction batch (format: s3://bucket/key)
     * @return List of transactions
     * @throws IllegalArgumentException if S3 path format is invalid
     * @throws IllegalStateException if transaction batch cannot be loaded
     */
    private fun loadTransactionBatch(s3Path: String): List<Transaction> {
        // Parse S3 path (format: s3://bucket/key)
        if (!s3Path.startsWith("s3://")) {
            throw IllegalArgumentException("Invalid S3 path format: $s3Path. Expected format: s3://bucket/key")
        }
        
        val pathWithoutProtocol = s3Path.substring(5) // Remove "s3://"
        val firstSlashIndex = pathWithoutProtocol.indexOf('/')
        
        if (firstSlashIndex == -1) {
            throw IllegalArgumentException("Invalid S3 path format: $s3Path. Expected format: s3://bucket/key")
        }
        
        val bucket = pathWithoutProtocol.substring(0, firstSlashIndex)
        val key = pathWithoutProtocol.substring(firstSlashIndex + 1)
        
        logger.info("Loading transaction batch from S3: bucket=$bucket, key=$key")
        
        try {
            val getObjectRequest = GetObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .build()
            
            val responseBytes = s3Client.getObject(getObjectRequest).readAllBytes()
            val jsonString = String(responseBytes, Charsets.UTF_8)
            
            // Parse JSON array of transactions
            val jsonNode = objectMapper.readTree(jsonString)
            
            if (!jsonNode.isArray) {
                throw IllegalStateException("Transaction batch must be a JSON array")
            }
            
            val transactions = mutableListOf<Transaction>()
            jsonNode.forEach { transactionNode ->
                val transaction = objectMapper.treeToValue(transactionNode, Transaction::class.java)
                transactions.add(transaction)
            }
            
            logger.info("Successfully loaded ${transactions.size} transactions from S3")
            
            return transactions
            
        } catch (e: Exception) {
            logger.error("Failed to load transaction batch from S3: ${e.message}", e)
            throw IllegalStateException(
                "Failed to load transaction batch from S3. Bucket: $bucket, Key: $key. Error: ${e.message}",
                e
            )
        }
    }
    
    /**
     * Invokes the SageMaker endpoint to get a fraud prediction for a transaction.
     * 
     * @param endpointName Name of the SageMaker endpoint
     * @param features Transaction features
     * @return Fraud score (0.0 to 1.0)
     * @throws IllegalStateException if endpoint invocation fails
     */
    private fun invokeEndpoint(endpointName: String, features: Map<String, Double>): Double {
        try {
            // Convert features to CSV format (comma-separated values, no header)
            // XGBoost expects: value1,value2,value3,...
            val payload = features.values.joinToString(",")
            
            val response = sageMakerRuntimeClient.invokeEndpoint(
                InvokeEndpointRequest.builder()
                    .endpointName(endpointName)
                    .contentType("text/csv")
                    .body(SdkBytes.fromUtf8String(payload))
                    .build()
            )
            
            // Response is a single number (fraud score)
            val fraudScore = response.body().asUtf8String().trim().toDouble()
            
            // Validate fraud score is in valid range
            if (fraudScore < 0.0 || fraudScore > 1.0) {
                logger.warn("Fraud score out of range: $fraudScore. Clamping to [0.0, 1.0]")
                return fraudScore.coerceIn(0.0, 1.0)
            }
            
            return fraudScore
            
        } catch (e: Exception) {
            logger.error("Failed to invoke SageMaker endpoint: ${e.message}", e)
            throw IllegalStateException(
                "Failed to invoke SageMaker endpoint: $endpointName. Error: ${e.message}",
                e
            )
        }
    }
}
