# Deployment Setup Guide for quannh0308

This guide provides step-by-step instructions for deploying the Fraud Detection ML Pipeline to your AWS account.

## Prerequisites

- AWS CLI installed and configured with your credentials
- AWS account with appropriate permissions (SageMaker, S3, Lambda, Step Functions, DynamoDB, SNS, Glue)
- Kaggle account to download the dataset

## Step 1: Download the Kaggle Dataset

### Option A: Manual Download (Easiest)

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" (you may need to create a free Kaggle account)
3. You'll get `creditcard.csv` (about 150 MB)
4. Rename it to `kaggle-credit-card-fraud.csv`

### Option B: Using Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API credentials (get from https://www.kaggle.com/account)
# Download kaggle.json and place in ~/.kaggle/

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
mv creditcard.csv kaggle-credit-card-fraud.csv
```

## Step 2: Create S3 Bucket and Upload Dataset

```bash
# Create your unique data bucket
aws s3 mb s3://fraud-detection-data-quannh0308-20260214

# Upload the dataset
aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data-quannh0308-20260214/

# Verify upload
aws s3 ls s3://fraud-detection-data-quannh0308-20260214/
```

**Expected output:**
```
2026-02-14 XX:XX:XX  150473250 kaggle-credit-card-fraud.csv
```

## Step 3: Set Environment Variables

```bash
# Set your AWS region
export AWS_REGION=us-east-1

# Set environment name (dev, staging, prod)
export ENVIRONMENT=dev

# Verify your AWS account
aws sts get-caller-identity
```

## Step 4: Deploy Training Pipeline

```bash
# Make deployment script executable
chmod +x deploy-training-pipeline.sh

# Deploy
./deploy-training-pipeline.sh
```

**What this does:**
- Builds all Gradle modules
- Packages Lambda functions as JAR files
- Uploads Glue script to S3
- Deploys CDK infrastructure stack
- Creates Step Functions workflow
- Sets up EventBridge schedule (weekly)

## Step 5: Deploy Inference Pipeline

```bash
# Make deployment script executable
chmod +x deploy-inference-pipeline.sh

# Deploy
./deploy-inference-pipeline.sh
```

**What this does:**
- Packages Lambda functions
- Deploys CDK infrastructure stack
- Creates Step Functions workflow
- Sets up DynamoDB table
- Creates SNS topics
- Sets up EventBridge schedule (daily)

## Step 6: Trigger Training Pipeline (First Run)

```bash
# Start the training workflow manually
aws stepfunctions start-execution \
  --state-machine-arn $(aws stepfunctions list-state-machines --query "stateMachines[?name=='FraudDetectionTraining-dev'].stateMachineArn" --output text) \
  --input '{
    "datasetS3Path": "s3://fraud-detection-data-quannh0308-20260214/kaggle-credit-card-fraud.csv",
    "outputPrefix": "s3://fraud-detection-data-quannh0308-20260214/prepared/",
    "trainSplit": 0.70,
    "validationSplit": 0.15,
    "testSplit": 0.15
  }'

# Monitor execution
aws stepfunctions describe-execution \
  --execution-arn <execution-arn-from-previous-command>
```

**Expected duration:** 2-4 hours (model training takes time)

## Step 7: Prepare Daily Transaction Batch

Once training completes and deploys a model, you can run inference:

```bash
# Upload a sample transaction batch
aws s3 cp examples/transaction-batch.json \
  s3://fraud-detection-data-quannh0308-20260214/daily-batches/2024-01-15.json

# Start inference workflow
aws stepfunctions start-execution \
  --state-machine-arn $(aws stepfunctions list-state-machines --query "stateMachines[?name=='FraudDetectionInference-dev'].stateMachineArn" --output text) \
  --input '{
    "transactionBatchPath": "s3://fraud-detection-data-quannh0308-20260214/daily-batches/2024-01-15.json",
    "batchDate": "2024-01-15"
  }'
```

**Expected duration:** 5-30 minutes

## Step 8: Monitor and Verify

### Check CloudWatch Logs

```bash
# Training pipeline logs
aws logs tail /aws/lambda/fraud-detection-train-handler --follow
aws logs tail /aws/lambda/fraud-detection-evaluate-handler --follow
aws logs tail /aws/lambda/fraud-detection-deploy-handler --follow
aws logs tail /aws-glue/jobs/fraud-detection-data-prep --follow

# Inference pipeline logs
aws logs tail /aws/lambda/fraud-detection-score-handler --follow
aws logs tail /aws/lambda/fraud-detection-store-handler --follow
aws logs tail /aws/lambda/fraud-detection-alert-handler --follow
aws logs tail /aws/lambda/fraud-detection-monitor-handler --follow
```

### Check DynamoDB Table

```bash
# Query fraud scores
aws dynamodb scan \
  --table-name FraudScores \
  --limit 10
```

### Check SNS Topics

```bash
# List SNS topics
aws sns list-topics | grep fraud-detection

# Subscribe to alerts (optional)
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:{account-id}:fraud-detection-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

## Troubleshooting

### Issue: Deployment script fails

**Solution:** Check that you have all required AWS permissions:
- IAM: CreateRole, AttachRolePolicy
- Lambda: CreateFunction, UpdateFunctionCode
- Step Functions: CreateStateMachine
- S3: CreateBucket, PutObject
- DynamoDB: CreateTable
- SNS: CreateTopic
- Glue: CreateJob

### Issue: Training pipeline fails at DataPrep stage

**Solution:** Verify dataset is uploaded correctly:
```bash
aws s3 ls s3://fraud-detection-data-quannh0308-20260214/
# Should show kaggle-credit-card-fraud.csv
```

### Issue: Inference pipeline fails - no endpoint found

**Solution:** Training pipeline must complete successfully first to deploy the model endpoint.

## Cost Estimate

**Monthly costs (approximate):**
- Training Pipeline: ~$164/month (mostly SageMaker endpoint)
- Inference Pipeline: ~$6.50/month
- S3 Storage: ~$0.03/month
- **Total: ~$170/month**

**To reduce costs:**
- Delete the SageMaker endpoint when not in use
- Use smaller instance types
- Reduce training frequency

## Clean Up (When Done Testing)

```bash
# Delete all resources
aws cloudformation delete-stack --stack-name FraudDetectionTraining-dev
aws cloudformation delete-stack --stack-name FraudDetectionInference-dev

# Delete S3 buckets (after emptying them)
aws s3 rm s3://fraud-detection-data-quannh0308-20260214 --recursive
aws s3 rb s3://fraud-detection-data-quannh0308-20260214

# Delete other buckets created by CDK
aws s3 ls | grep fraud-detection
```

## Next Steps

After successful deployment:
1. Monitor the first training run (2-4 hours)
2. Verify model endpoint is created
3. Run inference pipeline with sample data
4. Check DynamoDB for fraud scores
5. Subscribe to SNS alerts

## Questions?

If you encounter issues:
1. Check CloudWatch logs for error messages
2. Verify IAM permissions
3. Ensure dataset is uploaded correctly
4. Check that all AWS services are available in your region

Good luck with your deployment! 🚀
