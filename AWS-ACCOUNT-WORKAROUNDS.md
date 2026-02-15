# AWS Account Workarounds for Amazon Internal Accounts

This document describes workarounds for AWS service quota issues in Amazon internal accounts (IibsAdminAccess, etc.).

## Issue: Containment Score Restrictions

**Problem:** New or unused AWS accounts have an "undefined containment score" that restricts access to certain services:
- AWS Glue (ConcurrentRunsExceededException)
- Amazon SageMaker (ResourceLimitExceededException)
- Other compute services

**Root Cause:** AWS fraud prevention system for new accounts.

**References:**
- https://w.amazon.com/bin/view/AWSFraud/FIT/Containment_Score
- https://sage.amazon.dev/questions/1393824
- https://ec2-baywatch-prod-iad.iad.proxy.amazon.com/pages/accountInfo

## Solution: Boost Containment Score with EC2

### Step 1: Launch t3.large EC2 Instance

```bash
# Create security group (if needed)
aws ec2 create-security-group \
  --group-name temp-containment-sg \
  --description "Temporary SG for containment score boost"

# Get security group ID
SG_ID=$(aws ec2 describe-security-groups --group-names temp-containment-sg --query 'SecurityGroups[0].GroupId' --output text)

# Get latest Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-2023*" "Name=architecture,Values=x86_64" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# Launch t3.large instance (keep it PRIVATE - no public access!)
aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=containment-score-boost}]'
```

### Step 2: Wait 5-10 Minutes

Let the instance run to increase your containment score. You can check the instance status:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=containment-score-boost" \
  --query 'Reservations[*].Instances[*].{ID:InstanceId,State:State.Name,LaunchTime:LaunchTime}' \
  --output table
```

### Step 3: Terminate the Instance

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=containment-score-boost" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

# Terminate
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

### Step 4: Wait 5 Minutes

After terminating, wait for the containment score to update in AWS systems.

### Step 5: Try Your Workflow Again

Your Glue and SageMaker quotas should now be available:

```bash
# Test Glue
aws glue start-job-run --job-name fraud-detection-data-prep-dev

# Test SageMaker (via Step Functions)
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:474957690766:stateMachine:FraudDetectionTraining-dev \
  --input '{...}'
```

## Verification

Check your containment score:
- https://ec2-baywatch-prod-iad.iad.proxy.amazon.com/pages/accountInfo
- Look for your account: 474957690766

## Troubleshooting

### Glue Still Fails with ConcurrentRunsExceededException

**Possible causes:**
1. Containment score hasn't updated yet (wait longer)
2. Previous Glue job is still terminating (wait 2-3 minutes)
3. Glue job has MaxConcurrentRuns=1 and there's a race condition

**Solutions:**
```bash
# Check for running Glue jobs
aws glue get-job-runs --job-name fraud-detection-data-prep-dev \
  --query 'JobRuns[?JobRunState==`RUNNING` || JobRunState==`STARTING`]' \
  --output table

# If jobs are stuck, contact AWS support to reset
# Or wait 10-15 minutes for automatic cleanup
```

### SageMaker Still Fails with ResourceLimitExceededException

**Solution:** Run the EC2 instance for longer (15-20 minutes instead of 5-10).

## Alternative: Request Quota Increases

For permanent solution, request quota increases:

```bash
# Request Glue quota increase
aws service-quotas request-service-quota-increase \
  --service-code glue \
  --quota-code L-B2ECFB0A \
  --desired-value 10

# Request SageMaker quota increase
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-D9D49FF7 \
  --desired-value 2
```

Or contact your AWS account administrator.

## Cost of EC2 Workaround

- t3.large: $0.0832/hour
- Running for 10 minutes: ~$0.014
- **Total cost: Less than 2 cents**

This is a one-time cost to unlock your account's compute services.

## Notes

- This workaround is specific to Amazon internal accounts
- Standard AWS accounts don't have this issue
- The containment score persists - you only need to do this once
- If you don't use AWS services for months, you may need to repeat
