#!/usr/bin/env bash
# Deploy SageMaker Studio CDK stack for fraud detection experimentation.
#
# Usage:
#   ./deploy.sh                  # deploy to default account/region
#   ./deploy.sh --region us-west-2
#   ./deploy.sh --destroy        # tear down the stack
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_NAME="FraudDetectionSageMakerStudio"

# ── helpers ──────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

check_prerequisites() {
    local missing=0
    for cmd in python3 pip npm npx; do
        if ! command -v "$cmd" &>/dev/null; then
            error "Required command not found: $cmd"
            missing=1
        fi
    done
    if ! command -v cdk &>/dev/null && ! npx --yes aws-cdk --version &>/dev/null; then
        error "AWS CDK CLI not found. Install with: npm install -g aws-cdk"
        missing=1
    fi
    if [ "$missing" -eq 1 ]; then
        exit 1
    fi
}

# ── parse args ───────────────────────────────────────────────────────
DESTROY=false
REGION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --destroy)  DESTROY=true; shift ;;
        --region)   REGION="$2"; shift 2 ;;
        *)          error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── main ─────────────────────────────────────────────────────────────
check_prerequisites

cd "$SCRIPT_DIR"

# Install Python dependencies
info "Installing Python dependencies …"
pip install -r requirements.txt --quiet

# Resolve CDK command
if command -v cdk &>/dev/null; then
    CDK_CMD="cdk"
else
    CDK_CMD="npx aws-cdk"
fi

CONTEXT_ARGS=""
if [ -n "$REGION" ]; then
    CONTEXT_ARGS="-c region=$REGION"
fi

if [ "$DESTROY" = true ]; then
    info "Destroying stack $STACK_NAME …"
    $CDK_CMD destroy "$STACK_NAME" --force $CONTEXT_ARGS
    info "Stack destroyed."
    exit 0
fi

# Bootstrap (idempotent)
info "Bootstrapping CDK environment …"
$CDK_CMD bootstrap $CONTEXT_ARGS 2>/dev/null || true

# Synthesize
info "Synthesizing CloudFormation template …"
$CDK_CMD synth "$STACK_NAME" $CONTEXT_ARGS --quiet

# Deploy
info "Deploying stack $STACK_NAME …"
$CDK_CMD deploy "$STACK_NAME" --require-approval never $CONTEXT_ARGS --outputs-file cdk-outputs.json

# Print outputs
info "Deployment complete!"
echo ""
if [ -f cdk-outputs.json ]; then
    info "Stack outputs:"
    python3 -c "
import json, pathlib
outputs = json.loads(pathlib.Path('cdk-outputs.json').read_text())
for stack, vals in outputs.items():
    for key, val in vals.items():
        print(f'  {key}: {val}')
"
fi
echo ""
info "SageMaker Studio URL is available in the outputs above."
