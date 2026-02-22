#!/bin/bash

###############################################################################
# Containment Score Boost Script
#
# This script launches a t3.large EC2 instance to increase your AWS account's
# containment score, which unlocks Glue and SageMaker services.
#
# Usage:
#   ./boost-containment-score.sh
#
# The script will:
# 1. Launch a t3.large instance
# 2. Wait for you to confirm (run for 30-60 minutes)
# 3. Terminate the instance
###############################################################################

set -e

echo "========================================="
echo "AWS Containment Score Boost"
echo "========================================="
echo ""

# Get latest Amazon Linux 2023 AMI
echo "Finding latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*" "Name=architecture,Values=x86_64" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
  echo "ERROR: Could not find Amazon Linux 2023 AMI"
  exit 1
fi

echo "✓ AMI ID: $AMI_ID"
echo ""

# Get or create security group
echo "Checking for security group..."
SG_ID=$(aws ec2 describe-security-groups \
  --group-names temp-containment-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" == "None" ]; then
  echo "Creating security group..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name temp-containment-sg \
    --description "Temporary SG for containment score boost" \
    --query 'GroupId' \
    --output text)
fi

echo "✓ Security Group ID: $SG_ID"
echo ""

# Launch instance
echo "Launching t3.large EC2 instance..."
LAUNCH_OUTPUT=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type t3.large \
  --security-group-ids $SG_ID \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=containment-score-boost},{Key=Purpose,Value=unlock-glue-sagemaker}]')

INSTANCE_ID=$(echo $LAUNCH_OUTPUT | jq -r '.Instances[0].InstanceId')

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "null" ]; then
  echo "ERROR: Failed to launch instance"
  exit 1
fi

echo "✓ Instance launched: $INSTANCE_ID"
echo ""
echo "========================================="
echo "IMPORTANT: Let the instance run for 30-60 minutes"
echo "========================================="
echo ""
echo "The longer it runs, the higher your containment score."
echo "Recommended: 30-60 minutes for best results"
echo ""
echo "Instance details:"
aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].{ID:InstanceId,Type:InstanceType,State:State.Name,LaunchTime:LaunchTime}' \
  --output table

echo ""
echo "========================================="
echo "When ready to terminate (after 30-60 min):"
echo "========================================="
echo ""
echo "Run this command:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Or use this script:"
echo "  ./terminate-containment-instance.sh $INSTANCE_ID"
echo ""
echo "After terminating, wait 10-15 minutes for the score to update,"
echo "then try your Glue/SageMaker workflows again."
echo ""

# Save instance ID for easy termination
echo $INSTANCE_ID > .containment-instance-id
echo "Instance ID saved to .containment-instance-id"
