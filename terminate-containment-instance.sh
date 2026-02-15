#!/bin/bash

###############################################################################
# Terminate Containment Score Boost Instance
#
# Usage:
#   ./terminate-containment-instance.sh [instance-id]
#
# If no instance ID provided, reads from .containment-instance-id file
###############################################################################

set -e

INSTANCE_ID=$1

# If no instance ID provided, try to read from file
if [ -z "$INSTANCE_ID" ]; then
  if [ -f .containment-instance-id ]; then
    INSTANCE_ID=$(cat .containment-instance-id)
    echo "Using instance ID from .containment-instance-id: $INSTANCE_ID"
  else
    echo "ERROR: No instance ID provided and .containment-instance-id file not found"
    echo ""
    echo "Usage:"
    echo "  ./terminate-containment-instance.sh <instance-id>"
    echo ""
    echo "Or find running instances:"
    echo "  aws ec2 describe-instances --filters \"Name=tag:Purpose,Values=unlock-glue-sagemaker\" \"Name=instance-state-name,Values=running\" --query 'Reservations[*].Instances[*].{ID:InstanceId,State:State.Name}' --output table"
    exit 1
  fi
fi

echo "========================================="
echo "Terminating Containment Score Instance"
echo "========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo ""

# Verify instance exists
echo "Checking instance status..."
INSTANCE_STATE=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text 2>/dev/null)

if [ -z "$INSTANCE_STATE" ] || [ "$INSTANCE_STATE" == "None" ]; then
  echo "ERROR: Instance $INSTANCE_ID not found"
  exit 1
fi

echo "Current state: $INSTANCE_STATE"
echo ""

# Terminate
if [ "$INSTANCE_STATE" == "terminated" ] || [ "$INSTANCE_STATE" == "terminating" ]; then
  echo "Instance is already terminated or terminating"
else
  echo "Terminating instance..."
  aws ec2 terminate-instances --instance-ids $INSTANCE_ID
  echo "✓ Termination initiated"
fi

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "1. Wait 10-15 minutes for containment score to update"
echo "2. Try your Glue/SageMaker workflow again"
echo ""
echo "Run your training workflow:"
echo "  aws stepfunctions start-execution \\"
echo "    --state-machine-arn arn:aws:states:us-east-1:474957690766:stateMachine:FraudDetectionTraining-dev \\"
echo "    --input '{...}'"
echo ""

# Clean up the instance ID file
if [ -f .containment-instance-id ]; then
  rm .containment-instance-id
  echo "Cleaned up .containment-instance-id file"
fi
