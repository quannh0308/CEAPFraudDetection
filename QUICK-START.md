# Quick Start Guide

## You're Ready to Deploy! 🚀

You already have:
- ✅ Dataset uploaded: `s3://fraud-detection-data-quannh0308-20260214/kaggle-credit-card-fraud.csv`
- ✅ Code configured to use your bucket suffix: `quannh0308-20260214`
- ✅ All tests passing (79/79)

## Deploy in 3 Commands

### 1. Set Environment Variables

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET_SUFFIX=quannh0308-20260214
export ENVIRONMENT=dev
```

### 2. Deploy Training Pipeline

```bash
./deploy-training-pipeline.sh
```

**Wait time:** 10-15 minutes

**What gets created:**
- S3 buckets: `fraud-detection-{workflow,models,config}-quannh0308-20260214`
- Glue job: `fraud-detection-data-prep`
- Lambda functions: train-handler, evaluate-handler, deploy-handler
- Step Functions: FraudDetectionTrainingWorkflow
- EventBridge: Weekly schedule (Sunday 2 AM)

### 3. Deploy Inference Pipeline

```bash
./deploy-inference-pipeline.sh
```

**Wait time:** 5-10 minutes

**What gets created:**
- Lambda functions: score-handler, store-handler, alert-handler, monitor-handler
- DynamoDB table: FraudScores
- SNS topics: fraud-detection-alerts, fraud-detection-monitoring
- Step Functions: FraudDetectionInferenceWorkflow
- EventBridge: Daily schedule (1 AM)

## Run Your First Training

```bash
# Start training workflow
aws stepfunctions start-execution \
  --state-machine-arn $(aws stepfunctions list-state-machines --query "stateMachines[?name=='FraudDetectionTrainingPipeline-dev'].stateMachineArn" --output text) \
  --input '{
    "datasetS3Path": "s3://fraud-detection-data-quannh0308-20260214/kaggle-credit-card-fraud.csv",
    "outputPrefix": "s3://fraud-detection-data-quannh0308-20260214/prepared/",
    "trainSplit": 0.70,
    "validationSplit": 0.15,
    "testSplit": 0.15
  }'
```

**Duration:** 2-4 hours (model training takes time)

## Monitor Progress

```bash
# Check Step Functions console
open "https://console.aws.amazon.com/states/home?region=us-east-1#/statemachines"

# Or use CLI
aws stepfunctions list-executions \
  --state-machine-arn $(aws stepfunctions list-state-machines --query "stateMachines[?name=='FraudDetectionTrainingPipeline-dev'].stateMachineArn" --output text) \
  --max-results 1
```

## After Training Completes

### Run Inference

```bash
# Upload sample transaction batch
aws s3 cp examples/transaction-batch.json \
  s3://fraud-detection-data-quannh0308-20260214/daily-batches/$(date +%Y-%m-%d).json

# Start inference workflow
aws stepfunctions start-execution \
  --state-machine-arn $(aws stepfunctions list-state-machines --query "stateMachines[?name=='FraudDetectionInferencePipeline-dev'].stateMachineArn" --output text) \
  --input "{
    \"transactionBatchPath\": \"s3://fraud-detection-data-quannh0308-20260214/daily-batches/$(date +%Y-%m-%d).json\",
    \"batchDate\": \"$(date +%Y-%m-%d)\"
  }"
```

### Check Results

```bash
# Query DynamoDB for fraud scores
aws dynamodb scan --table-name FraudScores --limit 10

# Check CloudWatch logs
aws logs tail /aws/lambda/fraud-detection-score-handler --follow
```

## Troubleshooting

### Deployment fails?
- Check AWS credentials: `aws sts get-caller-identity`
- Verify permissions (IAM, Lambda, Step Functions, S3, etc.)
- Check CloudFormation console for detailed errors

### Training fails?
- Verify dataset exists: `aws s3 ls s3://fraud-detection-data-quannh0308-20260214/`
- Check Glue job logs in CloudWatch
- Ensure SageMaker service limits aren't exceeded

### Inference fails?
- Training must complete first (creates the model endpoint)
- Check that endpoint exists: `aws sagemaker list-endpoints`
- Verify transaction batch format matches examples

## Cost Management

**Monthly estimate: ~$170**
- SageMaker endpoint: ~$164/month (runs 24/7)
- Lambda + Step Functions: ~$6/month
- S3 + DynamoDB: ~$0.50/month

**To reduce costs:**
```bash
# Delete SageMaker endpoint when not in use
aws sagemaker delete-endpoint --endpoint-name fraud-detection-prod

# Recreate it by running training pipeline again
```

## Clean Up Everything

```bash
# Delete CloudFormation stacks
aws cloudformation delete-stack --stack-name FraudDetectionInferencePipeline-dev
aws cloudformation delete-stack --stack-name FraudDetectionTrainingPipeline-dev

# Wait for deletion to complete (5-10 minutes)
aws cloudformation wait stack-delete-complete --stack-name FraudDetectionInferencePipeline-dev
aws cloudformation wait stack-delete-complete --stack-name FraudDetectionTrainingPipeline-dev

# Delete S3 buckets (must be empty first)
aws s3 rm s3://fraud-detection-data-quannh0308-20260214 --recursive
aws s3 rb s3://fraud-detection-data-quannh0308-20260214

aws s3 rm s3://fraud-detection-workflow-quannh0308-20260214 --recursive
aws s3 rb s3://fraud-detection-workflow-quannh0308-20260214

aws s3 rm s3://fraud-detection-models-quannh0308-20260214 --recursive
aws s3 rb s3://fraud-detection-models-quannh0308-20260214

aws s3 rm s3://fraud-detection-config-quannh0308-20260214 --recursive
aws s3 rb s3://fraud-detection-config-quannh0308-20260214
```

## Need Help?

See `DEPLOYMENT-SETUP.md` for detailed step-by-step instructions.
