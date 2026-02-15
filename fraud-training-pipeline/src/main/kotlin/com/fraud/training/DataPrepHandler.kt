package com.fraud.training

import com.fasterxml.jackson.databind.JsonNode
import com.fraud.common.handler.WorkflowLambdaHandler
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import java.io.BufferedReader
import java.io.InputStreamReader
import kotlin.random.Random

/**
 * Lambda handler for data preparation stage (replaces Glue job).
 * 
 * This handler prepares historical transaction data for model training by:
 * 1. Loading the Kaggle Credit Card Fraud Detection dataset from S3
 * 2. Splitting data into train (70%), validation (15%), test (15%) sets
 * 3. Writing prepared datasets to S3 in CSV format (compatible with SageMaker)
 * 
 * **Note:** This Lambda-based implementation replaces the original Glue job
 * to work around AWS Glue service quota limitations. It provides identical
 * functionality using pandas-like data processing in Kotlin.
 * 
 * **Requirements:**
 * - Requirement 2.1: Load Kaggle Credit Card Fraud Detection dataset from S3
 * - Requirement 2.2: Validate dataset contains 284,807 records
 * - Requirement 2.3: Split dataset into train/validation/test (70/15/15)
 * - Requirement 2.4: Write prepared datasets in SageMaker-compatible format
 * 
 * **Input Format:**
 * ```json
 * {
 *   "datasetS3Path": "s3://bucket/kaggle-credit-card-fraud.csv",
 *   "outputPrefix": "s3://bucket/prepared/",
 *   "trainSplit": 0.70,
 *   "validationSplit": 0.15,
 *   "testSplit": 0.15
 * }
 * ```
 * 
 * **Output Format:**
 * ```json
 * {
 *   "trainDataPath": "s3://bucket/prepared/train.csv",
 *   "validationDataPath": "s3://bucket/prepared/validation.csv",
 *   "testDataPath": "s3://bucket/prepared/test.csv",
 *   "recordCounts": {
 *     "train": 199363,
 *     "validation": 42721,
 *     "test": 42723
 *   },
 *   "features": ["Time", "V1", ..., "V28", "Amount"],
 *   "targetColumn": "Class"
 * }
 * ```
 */
class DataPrepHandler : WorkflowLambdaHandler() {
    
    companion object {
        const val EXPECTED_RECORD_COUNT = 284807
    }
    
    override fun processData(input: JsonNode): JsonNode {
        val datasetS3Path = input.get("datasetS3Path")?.asText()
            ?: throw IllegalArgumentException("datasetS3Path is required")
        
        val outputPrefix = input.get("outputPrefix")?.asText()
            ?: throw IllegalArgumentException("outputPrefix is required")
        
        val trainSplit = input.get("trainSplit")?.asDouble() ?: 0.70
        val validationSplit = input.get("validationSplit")?.asDouble() ?: 0.15
        val testSplit = input.get("testSplit")?.asDouble() ?: 0.15
        
        logger.info("Loading dataset from: $datasetS3Path")
        logger.info("Output prefix: $outputPrefix")
        logger.info("Split ratios - Train: $trainSplit, Validation: $validationSplit, Test: $testSplit")
        
        // 1. Parse S3 path
        val s3Uri = datasetS3Path.removePrefix("s3://")
        val parts = s3Uri.split("/", limit = 2)
        if (parts.size != 2) {
            throw IllegalArgumentException("Invalid S3 path: $datasetS3Path")
        }
        val bucket = parts[0]
        val key = parts[1]
        
        // 2. Load dataset from S3
        val records = loadDatasetFromS3(bucket, key)
        
        // 3. Validate record count
        if (records.size != EXPECTED_RECORD_COUNT) {
            logger.warn("Expected $EXPECTED_RECORD_COUNT records, got ${records.size}")
        }
        logger.info("Loaded ${records.size} records from dataset")
        
        // 4. Shuffle and split data
        val shuffled = records.shuffled(Random(42)) // Fixed seed for reproducibility
        
        val trainSize = (records.size * trainSplit).toInt()
        val validationSize = (records.size * validationSplit).toInt()
        
        val trainData = shuffled.subList(0, trainSize)
        val validationData = shuffled.subList(trainSize, trainSize + validationSize)
        val testData = shuffled.subList(trainSize + validationSize, records.size)
        
        logger.info("Split sizes - Train: ${trainData.size}, Validation: ${validationData.size}, Test: ${testData.size}")
        
        // 5. Write datasets to S3
        val outputPrefixClean = outputPrefix.removeSuffix("/")
        val trainPath = writeDatasetToS3(trainData, outputPrefixClean, "train.csv")
        val validationPath = writeDatasetToS3(validationData, outputPrefixClean, "validation.csv")
        val testPath = writeDatasetToS3(testData, outputPrefixClean, "test.csv")
        
        logger.info("Data preparation complete")
        
        // 6. Return output metadata
        return objectMapper.createObjectNode().apply {
            put("trainDataPath", trainPath)
            put("validationDataPath", validationPath)
            put("testDataPath", testPath)
            putObject("recordCounts").apply {
                put("train", trainData.size)
                put("validation", validationData.size)
                put("test", testData.size)
            }
            putArray("features").apply {
                add("Time")
                for (i in 1..28) add("V$i")
                add("Amount")
            }
            put("targetColumn", "Class")
        }
    }
    
    /**
     * Loads dataset from S3 CSV file.
     * 
     * @param bucket S3 bucket name
     * @param key S3 object key
     * @return List of CSV rows (including header as first row)
     */
    private fun loadDatasetFromS3(bucket: String, key: String): List<String> {
        logger.info("Downloading dataset from S3: bucket=$bucket, key=$key")
        
        val response = s3Client.getObject(
            GetObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .build()
        )
        
        val records = mutableListOf<String>()
        BufferedReader(InputStreamReader(response)).use { reader ->
            reader.forEachLine { line ->
                records.add(line)
            }
        }
        
        logger.info("Downloaded ${records.size} lines from S3")
        return records
    }
    
    /**
     * Writes dataset to S3 as CSV file in SageMaker format.
     * 
     * SageMaker XGBoost expects:
     * - First column is the label (Class)
     * - No header row
     * - Label values are 0 or 1
     * 
     * @param data List of CSV rows (including header)
     * @param outputPrefix S3 output prefix
     * @param filename Output filename
     * @return Full S3 path to written file
     */
    private fun writeDatasetToS3(data: List<String>, outputPrefix: String, filename: String): String {
        val s3Path = "$outputPrefix/$filename"
        val s3Uri = s3Path.removePrefix("s3://")
        val parts = s3Uri.split("/", limit = 2)
        val bucket = parts[0]
        val key = parts[1]
        
        logger.info("Writing ${data.size} records to S3: $s3Path")
        
        // Reformat CSV for SageMaker: move Class column to first position, remove header
        val header = data[0].split(",")
        val classIndex = header.indexOf("Class")
        
        if (classIndex == -1) {
            throw IllegalStateException("Class column not found in dataset")
        }
        
        val reformattedRows = data.drop(1).map { row ->
            val columns = row.split(",")
            // Move Class column to first position
            val classValue = columns[classIndex]
            val features = columns.filterIndexed { index, _ -> index != classIndex }
            "$classValue,${features.joinToString(",")}"
        }
        
        val csvContent = reformattedRows.joinToString("\n")
        
        s3Client.putObject(
            PutObjectRequest.builder()
                .bucket(bucket)
                .key(key)
                .contentType("text/csv")
                .build(),
            RequestBody.fromString(csvContent)
        )
        
        logger.info("Successfully wrote ${reformattedRows.size} records to $s3Path (SageMaker format: label first, no header)")
        return s3Path
    }
}
