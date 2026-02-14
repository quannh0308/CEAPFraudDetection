#!/bin/bash

###############################################################################
# Fraud Detection Training Pipeline Deployment Script
#
# This script automates the deployment of the fraud detection training pipeline
# to AWS. It performs the following steps:
# 1. Validates prerequisites (AWS CLI, CDK, environment variables)
# 2. Builds the Gradle project and packages Lambda functions
# 3. Uploads the Glue data preparation script to S3
# 4. Deploys the CDK infrastructure stack
#
# Requirements:
# - AWS CLI configured with appropriate credentials
# - AWS CDK CLI installed (npm install -g aws-cdk)
# - Java 17+ and Gradle installed
# - Environment variables: AWS_REGION, AWS_ACCOUNT_ID, ENVIRONMENT
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=123456789012
#   export ENVIRONMENT=dev
#   ./deploy-training-pipeline.sh
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Script start
log_info "Starting Fraud Detection Training Pipeline deployment..."
echo ""

###############################################################################
# Step 1: Validate Prerequisites
###############################################################################

log_info "Step 1: Validating prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI is not installed. Please install it first."
    exit 1
fi
log_success "AWS CLI found: $(aws --version)"

# Check CDK CLI
if ! command -v cdk &> /dev/null; then
    log_error "AWS CDK CLI is not installed. Install with: npm install -g aws-cdk"
    exit 1
fi
log_success "AWS CDK CLI found: $(cdk --version)"

# Check Java
if ! command -v java &> /dev/null; then
    log_error "Java is not installed. Please install Java 17 or higher."
    exit 1
fi
JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$JAVA_VERSION" -lt 17 ]; then
    log_error "Java 17 or higher is required. Current version: $JAVA_VERSION"
    exit 1
fi
log_success "Java found: $(java -version 2>&1 | head -n 1)"

# Check Gradle
if ! command -v ./gradlew &> /dev/null; then
    log_error "Gradle wrapper not found. Please ensure gradlew is in the project root."
    exit 1
fi
log_success "Gradle wrapper found"

# Check required environment variables
if [ -z "${AWS_REGION:-}" ]; then
    log_error "AWS_REGION environment variable is not set"
    log_info "Example: export AWS_REGION=us-east-1"
    exit 1
fi
log_success "AWS_REGION: $AWS_REGION"

if [ -z "${AWS_ACCOUNT_ID:-}" ]; then
    log_error "AWS_ACCOUNT_ID environment variable is not set"
    log_info "Example: export AWS_ACCOUNT_ID=123456789012"
    exit 1
fi
log_success "AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"

if [ -z "${ENVIRONMENT:-}" ]; then
    log_warning "ENVIRONMENT not set, defaulting to 'dev'"
    export ENVIRONMENT=dev
fi
log_success "ENVIRONMENT: $ENVIRONMENT"

# Verify AWS credentials
log_info "Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    log_error "AWS credentials are not configured or invalid"
    log_info "Configure with: aws configure"
    exit 1
fi
CALLER_IDENTITY=$(aws sts get-caller-identity)
log_success "AWS credentials verified"
echo "$CALLER_IDENTITY"

echo ""

###############################################################################
# Step 2: Build Gradle Project and Package Lambda Functions
###############################################################################

log_info "Step 2: Building Gradle project and packaging Lambda functions..."

# Clean previous builds
log_info "Cleaning previous builds..."
./gradlew clean

# Build all modules
log_info "Building all modules..."
./gradlew build -x test

# Build Lambda deployment packages (fat JARs)
log_info "Building Lambda deployment packages..."
./gradlew :fraud-training-pipeline:shadowJar

# Verify Lambda JAR exists
LAMBDA_JAR="fraud-training-pipeline/build/libs/fraud-training-pipeline-1.0.0-SNAPSHOT.jar"
if [ ! -f "$LAMBDA_JAR" ]; then
    log_error "Lambda JAR not found at $LAMBDA_JAR"
    exit 1
fi
log_success "Lambda deployment package created: $LAMBDA_JAR"

echo ""

###############################################################################
# Step 3: Upload Glue Script to S3
###############################################################################

log_info "Step 3: Uploading Glue data preparation script to S3..."

# Define S3 bucket for Glue scripts
GLUE_SCRIPTS_BUCKET="fraud-detection-glue-scripts-${ENVIRONMENT}-${AWS_ACCOUNT_ID}"
GLUE_SCRIPT_PATH="glue-scripts/data-prep.py"

# Check if Glue script exists
if [ ! -f "$GLUE_SCRIPT_PATH" ]; then
    log_error "Glue script not found at $GLUE_SCRIPT_PATH"
    exit 1
fi

# Create S3 bucket if it doesn't exist
log_info "Checking if S3 bucket exists: $GLUE_SCRIPTS_BUCKET"
if ! aws s3 ls "s3://$GLUE_SCRIPTS_BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
    log_info "Bucket exists"
else
    log_info "Creating S3 bucket: $GLUE_SCRIPTS_BUCKET"
    if [ "$AWS_REGION" = "us-east-1" ]; then
        aws s3 mb "s3://$GLUE_SCRIPTS_BUCKET" --region "$AWS_REGION"
    else
        aws s3 mb "s3://$GLUE_SCRIPTS_BUCKET" --region "$AWS_REGION" --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    log_success "Bucket created"
fi

# Upload Glue script
log_info "Uploading Glue script to s3://$GLUE_SCRIPTS_BUCKET/data-prep.py"
aws s3 cp "$GLUE_SCRIPT_PATH" "s3://$GLUE_SCRIPTS_BUCKET/data-prep.py"
log_success "Glue script uploaded successfully"

echo ""

###############################################################################
# Step 4: Deploy CDK Infrastructure Stack
###############################################################################

log_info "Step 4: Deploying CDK infrastructure stack..."

# Navigate to infrastructure directory
cd infrastructure

# Bootstrap CDK (if not already done)
log_info "Bootstrapping CDK (if needed)..."
cdk bootstrap "aws://${AWS_ACCOUNT_ID}/${AWS_REGION}" || log_warning "CDK already bootstrapped"

# Synthesize CloudFormation template
log_info "Synthesizing CloudFormation template..."
cdk synth

# Deploy the stack
STACK_NAME="FraudDetectionTrainingPipeline-${ENVIRONMENT}"
log_info "Deploying stack: $STACK_NAME"
log_warning "This may take 10-15 minutes..."

cdk deploy "$STACK_NAME" \
    --require-approval never \
    --context envName="$ENVIRONMENT" \
    --context awsAccountId="$AWS_ACCOUNT_ID" \
    --context awsRegion="$AWS_REGION"

if [ $? -eq 0 ]; then
    log_success "Stack deployed successfully: $STACK_NAME"
else
    log_error "Stack deployment failed"
    cd ..
    exit 1
fi

# Return to project root
cd ..

echo ""

###############################################################################
# Deployment Complete
###############################################################################

log_success "========================================="
log_success "Training Pipeline Deployment Complete!"
log_success "========================================="
echo ""
log_info "Stack Name: $STACK_NAME"
log_info "Region: $AWS_REGION"
log_info "Environment: $ENVIRONMENT"
echo ""
log_info "Next Steps:"
log_info "1. Upload the Kaggle Credit Card Fraud dataset to S3:"
log_info "   aws s3 cp kaggle-credit-card-fraud.csv s3://fraud-detection-data-${ENVIRONMENT}-${AWS_ACCOUNT_ID}/"
echo ""
log_info "2. Trigger the training workflow manually (or wait for weekly schedule):"
log_info "   aws stepfunctions start-execution \\"
log_info "     --state-machine-arn \$(aws stepfunctions list-state-machines --query \"stateMachines[?name=='FraudDetectionTraining-${ENVIRONMENT}'].stateMachineArn\" --output text) \\"
log_info "     --input '{\"datasetS3Path\":\"s3://fraud-detection-data-${ENVIRONMENT}-${AWS_ACCOUNT_ID}/kaggle-credit-card-fraud.csv\",\"outputPrefix\":\"s3://fraud-detection-data-${ENVIRONMENT}-${AWS_ACCOUNT_ID}/prepared/\",\"trainSplit\":0.70,\"validationSplit\":0.15,\"testSplit\":0.15}'"
echo ""
log_info "3. Monitor the workflow execution in the AWS Step Functions console"
echo ""
log_info "Rollback Instructions (if needed):"
log_info "  cd infrastructure && cdk destroy $STACK_NAME"
echo ""
