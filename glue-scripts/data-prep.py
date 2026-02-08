"""
AWS Glue PySpark Data Preparation Script for Fraud Detection ML Pipeline

This script prepares the Kaggle Credit Card Fraud Detection dataset for SageMaker training.
It loads the dataset from S3, validates record counts, splits into train/validation/test sets,
and writes the prepared datasets to S3 in Parquet format.

Requirements: 2.1, 2.2, 2.3, 2.4, 11.1, 11.2, 11.4, 11.5
"""

import sys
import json
import logging
from datetime import datetime

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, when, count
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

# Initialize S3 client
s3_client = boto3.client('s3')


def get_job_parameters():
    """
    Extract job parameters from Glue job arguments.
    
    Expected parameters:
    - execution_id: Step Functions execution ID
    - workflow_bucket: S3 bucket for workflow orchestration
    - dataset_s3_path: S3 path to input dataset
    - output_prefix: S3 prefix for prepared datasets
    - train_split: Training set proportion (default 0.70)
    - validation_split: Validation set proportion (default 0.15)
    - test_split: Test set proportion (default 0.15)
    """
    args = getResolvedOptions(sys.argv, [
        'execution_id',
        'workflow_bucket',
        'dataset_s3_path',
        'output_prefix',
        'train_split',
        'validation_split',
        'test_split'
    ])
    
    return {
        'execution_id': args['execution_id'],
        'workflow_bucket': args['workflow_bucket'],
        'dataset_s3_path': args['dataset_s3_path'],
        'output_prefix': args['output_prefix'],
        'train_split': float(args.get('train_split', '0.70')),
        'validation_split': float(args.get('validation_split', '0.15')),
        'test_split': float(args.get('test_split', '0.15'))
    }


def validate_split_proportions(train_split, validation_split, test_split):
    """
    Validate that split proportions sum to 1.0.
    
    Requirement 2.3: Split proportions must be valid
    """
    total = train_split + validation_split + test_split
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Split proportions must sum to 1.0, got {total} "
            f"(train={train_split}, validation={validation_split}, test={test_split})"
        )
    logger.info(f"Split proportions validated: train={train_split}, validation={validation_split}, test={test_split}")


def load_dataset(dataset_s3_path):
    """
    Load the Kaggle Credit Card Fraud Detection dataset from S3.
    
    Requirement 2.1: Load dataset from S3
    """
    logger.info(f"Loading dataset from {dataset_s3_path}")
    
    try:
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(dataset_s3_path)
        
        record_count = df.count()
        logger.info(f"Loaded {record_count} records from dataset")
        
        return df, record_count
    
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_s3_path}: {str(e)}")
        raise


def validate_record_count(record_count, expected_count=284807):
    """
    Validate that the dataset contains the expected number of records.
    
    Requirement 2.2: Validate dataset contains 284,807 records
    """
    if record_count != expected_count:
        logger.warning(
            f"Dataset record count {record_count} does not match expected {expected_count}. "
            f"Proceeding with actual count."
        )
    else:
        logger.info(f"Dataset record count validated: {record_count} records")


def validate_data_quality(df):
    """
    Validate data quality by checking for missing values and outliers.
    
    Requirement 11.4, 11.5: Handle missing data and outliers appropriately
    """
    logger.info("Validating data quality...")
    
    # Check for missing values
    missing_counts = df.select([
        count(when(isnan(c) | col(c).isNull(), c)).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()
    
    total_missing = sum(missing_counts.values())
    
    if total_missing > 0:
        logger.warning(f"Found {total_missing} missing values across all columns")
        for col_name, missing_count in missing_counts.items():
            if missing_count > 0:
                logger.warning(f"  Column '{col_name}': {missing_count} missing values")
        
        # For this dataset, we expect no missing values
        # If missing values exist, fail the job
        raise ValueError(
            f"Dataset contains {total_missing} missing values. "
            f"Data quality validation failed."
        )
    
    logger.info("Data quality validation passed: no missing values found")
    
    # Check for required columns
    required_columns = ['Time', 'Amount', 'Class']
    for col_name in required_columns:
        if col_name not in df.columns:
            raise ValueError(f"Required column '{col_name}' not found in dataset")
    
    # Validate V1-V28 feature columns exist
    v_columns = [f'V{i}' for i in range(1, 29)]
    for col_name in v_columns:
        if col_name not in df.columns:
            raise ValueError(f"Required feature column '{col_name}' not found in dataset")
    
    logger.info("All required columns present in dataset")


def split_dataset(df, train_split, validation_split, test_split):
    """
    Split dataset into train, validation, and test sets.
    
    Requirement 2.3: Split data into training (70%), validation (15%), and test (15%) sets
    """
    logger.info(f"Splitting dataset: train={train_split}, validation={validation_split}, test={test_split}")
    
    # Use random split with seed for reproducibility
    train_df, validation_df, test_df = df.randomSplit(
        [train_split, validation_split, test_split],
        seed=42
    )
    
    train_count = train_df.count()
    validation_count = validation_df.count()
    test_count = test_df.count()
    total_count = train_count + validation_count + test_count
    
    logger.info(f"Split results:")
    logger.info(f"  Train: {train_count} records ({train_count/total_count*100:.2f}%)")
    logger.info(f"  Validation: {validation_count} records ({validation_count/total_count*100:.2f}%)")
    logger.info(f"  Test: {test_count} records ({test_count/total_count*100:.2f}%)")
    
    return train_df, validation_df, test_df, {
        'train': train_count,
        'validation': validation_count,
        'test': test_count
    }


def reorder_columns_for_sagemaker(df, target_column='Class'):
    """
    Reorder columns so target column is first, followed by features.
    
    SageMaker XGBoost expects: [target, feature1, feature2, ...]
    """
    # Get all columns except target
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Reorder: target first, then features
    ordered_columns = [target_column] + feature_columns
    
    return df.select(ordered_columns)


def write_parquet_dataset(df, output_path, dataset_name):
    """
    Write dataset to S3 in Parquet format for SageMaker.
    
    Requirement 2.4: Write prepared datasets to S3 in format compatible with SageMaker training
    
    Note: SageMaker XGBoost expects target column first, followed by feature columns.
    """
    logger.info(f"Writing {dataset_name} dataset to {output_path}")
    
    try:
        # Reorder columns: target first, then features
        df_ordered = reorder_columns_for_sagemaker(df)
        
        df_ordered.write.mode("overwrite").parquet(output_path)
        logger.info(f"Successfully wrote {dataset_name} dataset to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to write {dataset_name} dataset to {output_path}: {str(e)}")
        raise


def write_stage_output(workflow_bucket, execution_id, output_data):
    """
    Write stage output metadata to S3 for next stage consumption.
    
    Requirement 2.5: Write output metadata to S3 for the next stage
    """
    output_key = f"executions/{execution_id}/DataPrepStage/output.json"
    output_json = json.dumps(output_data, indent=2)
    
    logger.info(f"Writing stage output to s3://{workflow_bucket}/{output_key}")
    
    try:
        s3_client.put_object(
            Bucket=workflow_bucket,
            Key=output_key,
            Body=output_json,
            ContentType='application/json'
        )
        logger.info(f"Successfully wrote stage output to S3")
    
    except Exception as e:
        logger.error(f"Failed to write stage output to S3: {str(e)}")
        raise


def main():
    """
    Main execution function for data preparation.
    """
    try:
        # Get job parameters
        params = get_job_parameters()
        logger.info(f"Starting data preparation with parameters: {params}")
        
        # Validate split proportions
        validate_split_proportions(
            params['train_split'],
            params['validation_split'],
            params['test_split']
        )
        
        # Load dataset
        df, record_count = load_dataset(params['dataset_s3_path'])
        
        # Validate record count
        validate_record_count(record_count)
        
        # Validate data quality
        validate_data_quality(df)
        
        # Split dataset
        train_df, validation_df, test_df, record_counts = split_dataset(
            df,
            params['train_split'],
            params['validation_split'],
            params['test_split']
        )
        
        # Construct output paths
        output_prefix = params['output_prefix'].rstrip('/')
        train_path = f"{output_prefix}/train.parquet"
        validation_path = f"{output_prefix}/validation.parquet"
        test_path = f"{output_prefix}/test.parquet"
        
        # Write datasets to S3 in Parquet format
        write_parquet_dataset(train_df, train_path, "train")
        write_parquet_dataset(validation_df, validation_path, "validation")
        write_parquet_dataset(test_df, test_path, "test")
        
        # Get feature columns (all columns except Class)
        feature_columns = [col for col in df.columns if col != 'Class']
        
        # Prepare stage output metadata
        output_data = {
            "trainDataPath": train_path,
            "validationDataPath": validation_path,
            "testDataPath": test_path,
            "recordCounts": record_counts,
            "features": feature_columns,
            "targetColumn": "Class",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "SUCCESS"
        }
        
        # Write stage output to S3
        write_stage_output(
            params['workflow_bucket'],
            params['execution_id'],
            output_data
        )
        
        logger.info("Data preparation completed successfully")
        logger.info(f"Output: {json.dumps(output_data, indent=2)}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        
        # Write failure output to S3
        try:
            params = get_job_parameters()
            error_output = {
                "status": "FAILED",
                "errorMessage": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            write_stage_output(
                params['workflow_bucket'],
                params['execution_id'],
                error_output
            )
        except:
            logger.error("Failed to write error output to S3")
        
        raise


if __name__ == "__main__":
    main()
