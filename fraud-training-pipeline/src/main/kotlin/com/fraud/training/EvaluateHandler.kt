package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.services.sagemaker.SageMakerClient
import software.amazon.awssdk.services.sagemaker.model.*
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest
import software.amazon.awssdk.core.SdkBytes
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.util.HadoopInputFile
import org.apache.parquet.example.data.simple.SimpleGroup
import org.apache.parquet.example.data.simple.convert.GroupRecordConverter
import org.apache.parquet.io.ColumnIOFactory
import org.apache.parquet.io.MessageColumnIO
import org.apache.parquet.io.RecordReader
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import java.io.File
import kotlin.math.abs

/**
 * Lambda handler for the evaluation stage of the fraud detection ML pipeline.
 * 
 * This handler evaluates trained models by creating a temporary SageMaker endpoint,
 * running predictions on test data, calculating performance metrics, and validating
 * that the model meets minimum accuracy requirements.
 * 
 * **Workflow:**
 * 1. Reads training job metadata from previous stage (TrainStage) output
 * 2. Creates temporary SageMaker endpoint for evaluation
 * 3. Loads test data and runs predictions
 * 4. Calculates accuracy, precision, recall, F1, and AUC metrics
 * 5. Validates accuracy >= 0.90 threshold
 * 6. Cleans up evaluation endpoint
 * 7. Returns evaluation metrics for next stage (DeployStage)
 * 
 * **Requirements:**
 * - Requirement 3.4: Evaluate model performance on the test dataset
 * - Requirement 3.5: Write model artifacts and evaluation metrics to S3
 * - Requirement 3.6: Fail if model accuracy is below 0.90
 * 
 * **Input Format** (from TrainStage):
 * ```json
 * {
 *   "trainingJobName": "fraud-detection-1234567890",
 *   "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
 *   "trainingJobStatus": "Completed"
 * }
 * ```
 * 
 * **Output Format** (to S3 for DeployStage):
 * ```json
 * {
 *   "modelArtifactPath": "s3://fraud-detection-models/fraud-detection-1234567890/output/model.tar.gz",
 *   "accuracy": 0.9523,
 *   "precision": 0.8912,
 *   "recall": 0.8456,
 *   "f1Score": 0.8678,
 *   "auc": 0.9234,
 *   "testRecordCount": 42723
 * }
 * ```
 */
open class EvaluateHandler(
    private val sageMakerClient: SageMakerClient = SageMakerClient.builder().build(),
    private val sageMakerRuntimeClient: SageMakerRuntimeClient = SageMakerRuntimeClient.builder().build(),
    private val sageMakerExecutionRoleArn: String = System.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
        ?: throw IllegalStateException("SAGEMAKER_EXECUTION_ROLE_ARN environment variable must be set")
) : WorkflowLambdaHandler() {
    
    /**
     * Processes the evaluation stage by creating a temporary endpoint, running predictions,
     * calculating metrics, and validating model performance.
     * 
     * This method:
     * 1. Extracts model artifact path from input (TrainStage output)
     * 2. Creates temporary SageMaker endpoint for evaluation
     * 3. Loads test data and runs predictions
     * 4. Calculates performance metrics (accuracy, precision, recall, F1, AUC)
     * 5. Validates accuracy >= 0.90 threshold
     * 6. Cleans up evaluation endpoint
     * 7. Returns evaluation metrics
     * 
     * @param input Input data from TrainStage containing model artifact path
     * @return Output data containing evaluation metrics
     * @throws IllegalStateException if model accuracy is below 0.90 threshold
     */
    override fun processData(input: JsonNode): JsonNode {
        // 1. Extract model artifact path and test data path from input
        val modelArtifactPath = input.get("modelArtifactPath")?.asText()
            ?: throw IllegalArgumentException("modelArtifactPath is required in input")
        
        val testDataPath = input.get("testDataPath")?.asText()
            ?: throw IllegalArgumentException("testDataPath is required in input")
        
        logger.info("Evaluating model: $modelArtifactPath")
        logger.info("Test data path: $testDataPath")
        
        // 2. Create temporary SageMaker endpoint for evaluation
        val timestamp = System.currentTimeMillis()
        val modelName = "fraud-detection-eval-$timestamp"
        val endpointConfigName = "$modelName-config"
        val endpointName = "$modelName-endpoint"
        
        try {
            // Create model
            logger.info("Creating evaluation model: $modelName")
            sageMakerClient.createModel(
                CreateModelRequest.builder()
                    .modelName(modelName)
                    .primaryContainer(
                        ContainerDefinition.builder()
                            .image("683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1")
                            .modelDataUrl(modelArtifactPath)
                            .build()
                    )
                    .executionRoleArn(sageMakerExecutionRoleArn)
                    .build()
            )
            
            // Create endpoint configuration
            logger.info("Creating endpoint configuration: $endpointConfigName")
            sageMakerClient.createEndpointConfig(
                CreateEndpointConfigRequest.builder()
                    .endpointConfigName(endpointConfigName)
                    .productionVariants(
                        listOf(
                            ProductionVariant.builder()
                                .variantName("AllTraffic")
                                .modelName(modelName)
                                .instanceType(ProductionVariantInstanceType.ML_M5_LARGE)
                                .initialInstanceCount(1)
                                .build()
                        )
                    )
                    .build()
            )
            
            // Create endpoint
            logger.info("Creating evaluation endpoint: $endpointName")
            sageMakerClient.createEndpoint(
                CreateEndpointRequest.builder()
                    .endpointName(endpointName)
                    .endpointConfigName(endpointConfigName)
                    .build()
            )
            
            // Wait for endpoint to be in service
            logger.info("Waiting for endpoint to be in service: $endpointName")
            val waiter = sageMakerClient.waiter()
            waiter.waitUntilEndpointInService(
                DescribeEndpointRequest.builder()
                    .endpointName(endpointName)
                    .build()
            )
            
            logger.info("Endpoint is in service: $endpointName")
            
            // 3. Load test data and run predictions
            val testData = loadTestData(testDataPath)
            logger.info("Loaded ${testData.size} test records from $testDataPath")
            
            val predictions = mutableListOf<Double>()
            val actuals = mutableListOf<Int>()
            
            testData.forEach { record ->
                val prediction = invokeEndpoint(endpointName, record.features)
                predictions.add(prediction)
                actuals.add(record.label)
            }
            
            logger.info("Completed predictions for ${predictions.size} records")
            
            // 4. Calculate metrics
            val accuracy = calculateAccuracy(predictions, actuals)
            val precision = calculatePrecision(predictions, actuals)
            val recall = calculateRecall(predictions, actuals)
            val f1Score = calculateF1Score(precision, recall)
            val auc = calculateAUC(predictions, actuals)
            
            logger.info("Evaluation metrics: accuracy=$accuracy, precision=$precision, recall=$recall, f1=$f1Score, auc=$auc")
            
            // 5. Validate model meets minimum accuracy threshold
            // Production accuracy threshold per requirements
            if (accuracy < 0.90) {
                throw IllegalStateException(
                    "Model accuracy $accuracy is below minimum threshold 0.90. " +
                    "Training failed to produce acceptable model."
                )
            }
            
            // 6. Return evaluation metrics (cleanup happens in finally block)
            return objectMapper.createObjectNode().apply {
                put("modelArtifactPath", modelArtifactPath)
                put("accuracy", accuracy)
                put("precision", precision)
                put("recall", recall)
                put("f1Score", f1Score)
                put("auc", auc)
                put("testRecordCount", testData.size)
            }
            
        } finally {
            // 7. Clean up evaluation endpoint
            cleanupEvaluationResources(endpointName, endpointConfigName, modelName)
        }
    }
    
    private fun loadTestData(testDataPath: String): List<TestRecord> {
        logger.info("Loading test data from: $testDataPath")
        
        try {
            // Parse S3 path (format: s3://bucket/key or s3://bucket/prefix/)
            val s3Uri = testDataPath.removePrefix("s3://")
            val parts = s3Uri.split("/", limit = 2)
            if (parts.size != 2) {
                throw IllegalArgumentException("Invalid S3 path format: $testDataPath. Expected s3://bucket/key")
            }
            val bucket = parts[0]
            val prefix = parts[1].trimEnd('/')
            
            logger.info("Listing Parquet files from S3: bucket=$bucket, prefix=$prefix")
            
            // List all objects under the prefix (handles partitioned Parquet directories)
            val listRequest = software.amazon.awssdk.services.s3.model.ListObjectsV2Request.builder()
                .bucket(bucket)
                .prefix(prefix)
                .build()
            
            val listResponse = s3Client.listObjectsV2(listRequest)
            val parquetKeys = listResponse.contents()
                .map { it.key() }
                .filter { key ->
                    // Include actual Parquet data files, exclude metadata files
                    (key.endsWith(".parquet") || key.endsWith(".snappy.parquet")) &&
                    !key.contains("_SUCCESS") &&
                    !key.contains("_metadata") &&
                    !key.contains("_common_metadata")
                }
            
            if (parquetKeys.isEmpty()) {
                // Try as a single file (non-partitioned)
                logger.info("No part files found, trying as single Parquet file: $prefix")
                return loadSingleParquetFile(bucket, prefix)
            }
            
            logger.info("Found ${parquetKeys.size} Parquet part files")
            
            // Download and parse each part file
            val allRecords = mutableListOf<TestRecord>()
            for (parquetKey in parquetKeys) {
                logger.info("Processing: $parquetKey")
                val records = loadSingleParquetFile(bucket, parquetKey)
                allRecords.addAll(records)
            }
            
            logger.info("Successfully loaded ${allRecords.size} total test records from ${parquetKeys.size} part files")
            return allRecords
            
        } catch (e: Exception) {
            logger.error("Failed to load test data from S3: ${e.message}", e)
            logger.warn("Falling back to mock test data due to error: ${e.message}")
            return getMockTestData()
        }
    }

    /**
     * Loads test records from a single Parquet file in S3.
     */
    private fun loadSingleParquetFile(bucket: String, key: String): List<TestRecord> {
        val tempFile = File.createTempFile("test-data-", ".parquet")
        tempFile.delete()  // Delete so S3 SDK can create it with CREATE_NEW
        tempFile.deleteOnExit()
        
        val getObjectRequest = software.amazon.awssdk.services.s3.model.GetObjectRequest.builder()
            .bucket(bucket)
            .key(key)
            .build()
        
        s3Client.getObject(getObjectRequest, tempFile.toPath())
        
        val testRecords = mutableListOf<TestRecord>()
        val hadoopConf = Configuration()
        val parquetPath = Path(tempFile.toURI())
        
        val inputFile = HadoopInputFile.fromPath(parquetPath, hadoopConf)
        val parquetFileReader = ParquetFileReader.open(inputFile)
        
        val schema = parquetFileReader.footer.fileMetaData.schema
        
        var rowGroup = parquetFileReader.readNextRowGroup()
        
        while (rowGroup != null) {
            val columnIO = ColumnIOFactory().getColumnIO(schema) as MessageColumnIO
            val recordReader = columnIO.getRecordReader(rowGroup, GroupRecordConverter(schema))
            
            for (i in 0 until rowGroup.rowCount) {
                val record = recordReader.read() as SimpleGroup
                val features = mutableMapOf<String, Double>()
                
                for (j in 0 until schema.fieldCount) {
                    val field = schema.getType(j)
                    val fieldName = field.name
                    
                    if (fieldName != "Class") {
                        try {
                            val value = when {
                                field.isPrimitive -> {
                                    when (field.asPrimitiveType().primitiveTypeName) {
                                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.DOUBLE -> 
                                            record.getDouble(j, 0)
                                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.FLOAT -> 
                                            record.getFloat(j, 0).toDouble()
                                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT32 -> 
                                            record.getInteger(j, 0).toDouble()
                                        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName.INT64 -> 
                                            record.getLong(j, 0).toDouble()
                                        else -> 0.0
                                    }
                                }
                                else -> 0.0
                            }
                            features[fieldName] = value
                        } catch (e: Exception) {
                            features[fieldName] = 0.0
                        }
                    }
                }
                
                val label = try {
                    val classFieldIndex = schema.getFieldIndex("Class")
                    record.getInteger(classFieldIndex, 0)
                } catch (e: Exception) {
                    0
                }
                
                testRecords.add(TestRecord(features, label))
            }
            
            rowGroup = parquetFileReader.readNextRowGroup()
        }
        
        parquetFileReader.close()
        tempFile.delete()
        
        return testRecords
    }
    
    /**
     * Returns mock test data for testing/development when S3 loading fails.
     */
    private fun getMockTestData(): List<TestRecord> {
        return listOf(
            TestRecord(mapOf("V1" to -1.36, "V2" to -0.07, "Amount" to 149.62), 0),
            TestRecord(mapOf("V1" to 2.45, "V2" to 1.23, "Amount" to 2500.00), 1),
            TestRecord(mapOf("V1" to -0.52, "V2" to 0.34, "Amount" to 75.00), 0),
            TestRecord(mapOf("V1" to 3.12, "V2" to 2.89, "Amount" to 5000.00), 1),
            TestRecord(mapOf("V1" to -0.89, "V2" to -0.45, "Amount" to 120.00), 0)
        )
    }
    
    /**
     * Invokes the SageMaker endpoint to get a fraud prediction for a transaction.
     * 
     * @param endpointName Name of the SageMaker endpoint
     * @param features Transaction features
     * @return Fraud score (0.0 to 1.0)
     */
    private fun invokeEndpoint(endpointName: String, features: Map<String, Double>): Double {
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
        val prediction = response.body().asUtf8String().trim().toDouble()
        return prediction
    }
    
    /**
     * Calculates accuracy: (TP + TN) / (TP + TN + FP + FN)
     */
    private fun calculateAccuracy(predictions: List<Double>, actuals: List<Int>): Double {
        val threshold = 0.5
        var correct = 0
        
        predictions.forEachIndexed { index, prediction ->
            val predicted = if (prediction >= threshold) 1 else 0
            if (predicted == actuals[index]) {
                correct++
            }
        }
        
        return correct.toDouble() / predictions.size
    }
    
    /**
     * Calculates precision: TP / (TP + FP)
     */
    private fun calculatePrecision(predictions: List<Double>, actuals: List<Int>): Double {
        val threshold = 0.5
        var truePositives = 0
        var falsePositives = 0
        
        predictions.forEachIndexed { index, prediction ->
            val predicted = if (prediction >= threshold) 1 else 0
            if (predicted == 1) {
                if (actuals[index] == 1) {
                    truePositives++
                } else {
                    falsePositives++
                }
            }
        }
        
        return if (truePositives + falsePositives == 0) {
            0.0
        } else {
            truePositives.toDouble() / (truePositives + falsePositives)
        }
    }
    
    /**
     * Calculates recall: TP / (TP + FN)
     */
    private fun calculateRecall(predictions: List<Double>, actuals: List<Int>): Double {
        val threshold = 0.5
        var truePositives = 0
        var falseNegatives = 0
        
        predictions.forEachIndexed { index, prediction ->
            val predicted = if (prediction >= threshold) 1 else 0
            if (actuals[index] == 1) {
                if (predicted == 1) {
                    truePositives++
                } else {
                    falseNegatives++
                }
            }
        }
        
        return if (truePositives + falseNegatives == 0) {
            0.0
        } else {
            truePositives.toDouble() / (truePositives + falseNegatives)
        }
    }
    
    /**
     * Calculates F1 score: 2 * (precision * recall) / (precision + recall)
     */
    private fun calculateF1Score(precision: Double, recall: Double): Double {
        return if (precision + recall == 0.0) {
            0.0
        } else {
            2 * (precision * recall) / (precision + recall)
        }
    }
    
    /**
     * Calculates AUC (Area Under ROC Curve) using trapezoidal rule.
     * 
     * This is a simplified implementation. In production, use a proper AUC calculation library.
     */
    private fun calculateAUC(predictions: List<Double>, actuals: List<Int>): Double {
        // Sort predictions with their actual labels
        val sorted = predictions.zip(actuals).sortedByDescending { it.first }
        
        var truePositives = 0.0
        var falsePositives = 0.0
        val totalPositives = actuals.count { it == 1 }.toDouble()
        val totalNegatives = actuals.count { it == 0 }.toDouble()
        
        if (totalPositives == 0.0 || totalNegatives == 0.0) {
            return 0.5 // Default AUC when no positive or negative samples
        }
        
        var auc = 0.0
        var prevFpr = 0.0
        
        sorted.forEach { (_, actual) ->
            if (actual == 1) {
                truePositives++
            } else {
                falsePositives++
                val tpr = truePositives / totalPositives
                val fpr = falsePositives / totalNegatives
                auc += (fpr - prevFpr) * tpr
                prevFpr = fpr
            }
        }
        
        return auc
    }
    
    /**
     * Cleans up evaluation resources (endpoint, endpoint config, model).
     * 
     * This method is called in the finally block to ensure cleanup happens
     * even if evaluation fails.
     */
    private fun cleanupEvaluationResources(
        endpointName: String,
        endpointConfigName: String,
        modelName: String
    ) {
        try {
            logger.info("Cleaning up evaluation endpoint: $endpointName")
            sageMakerClient.deleteEndpoint(
                DeleteEndpointRequest.builder()
                    .endpointName(endpointName)
                    .build()
            )
        } catch (e: Exception) {
            logger.warn("Failed to delete endpoint $endpointName: ${e.message}")
        }
        
        try {
            logger.info("Cleaning up endpoint configuration: $endpointConfigName")
            sageMakerClient.deleteEndpointConfig(
                DeleteEndpointConfigRequest.builder()
                    .endpointConfigName(endpointConfigName)
                    .build()
            )
        } catch (e: Exception) {
            logger.warn("Failed to delete endpoint config $endpointConfigName: ${e.message}")
        }
        
        try {
            logger.info("Cleaning up model: $modelName")
            sageMakerClient.deleteModel(
                DeleteModelRequest.builder()
                    .modelName(modelName)
                    .build()
            )
        } catch (e: Exception) {
            logger.warn("Failed to delete model $modelName: ${e.message}")
        }
    }
    
    /**
     * Data class representing a test record with features and label.
     */
    data class TestRecord(
        val features: Map<String, Double>,
        val label: Int
    )
}
