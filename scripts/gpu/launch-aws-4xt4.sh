#!/usr/bin/env bash
set -euo pipefail

# Launch an AWS g4dn.12xlarge (4x T4, 4x16GB VRAM) for multi-GPU validation.
# On-demand: ~$3.91/hr. Full test cycle takes < 15 min → ~$1.00/run.
#
# Tests: DDP process-tree discovery, multi-GPU VRAM reporting, distributed
# topology validation (DDP, FSDP feasibility).
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Default VPC with internet access
#   - Key pair created (or set KEY_NAME below)
#
# Usage:
#   bash scripts/gpu/launch-aws-4xt4.sh
#
# The instance auto-terminates after tests complete.

INSTANCE_TYPE="g4dn.12xlarge"
AMI_ID="${AWS_AMI_ID:-ami-0a0e5d9c7acc336f1}"  # Ubuntu 22.04 (us-east-1)
KEY_NAME="${AWS_KEY_NAME:-}"
REGION="${AWS_REGION:-us-east-1}"
ALLOC_TOKEN="${ALLOC_TOKEN:-}"

if [ -z "$ALLOC_TOKEN" ]; then
    echo "WARN: ALLOC_TOKEN not set — will run free-tier only"
fi

# Build the user-data script that runs on boot
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail
exec > /var/log/alloc-validate.log 2>&1

echo "=== alloc-validate multi-GPU test starting ==="
date

# Install deps
apt-get update -qq && apt-get install -y -qq python3-venv git

# Clone and setup
cd /tmp
git clone https://github.com/alloc-labs/alloc-validate.git
cd alloc-validate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Verify all GPUs
python -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs detected: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Run validations
USERDATA
)

if [ -n "$ALLOC_TOKEN" ]; then
    USER_DATA+="
export ALLOC_TOKEN='$ALLOC_TOKEN'
make validate-full
"
else
    USER_DATA+="
make validate-free
"
fi

USER_DATA+='
# Full matrix with multi-GPU variations
make matrix --include-multi-gpu

echo "=== alloc-validate multi-GPU test complete ==="
date

# Auto-terminate
shutdown -h now
'

ENCODED_DATA=$(echo "$USER_DATA" | base64)

echo "Launching $INSTANCE_TYPE in $REGION..."

CMD=(aws ec2 run-instances
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --instance-initiated-shutdown-behavior terminate
    --user-data "$ENCODED_DATA"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alloc-validate-multi-gpu-test}]"
    --query "Instances[0].InstanceId"
    --output text
)

if [ -n "$KEY_NAME" ]; then
    CMD+=(--key-name "$KEY_NAME")
fi

INSTANCE_ID=$("${CMD[@]}")

echo "Instance launched: $INSTANCE_ID"
echo "Instance type: $INSTANCE_TYPE (4x T4, 4x16GB VRAM)"
echo "Cost: ~\$3.91/hr on-demand"
echo ""
echo "The instance will auto-terminate after tests complete."
echo ""
echo "To check progress:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'"
echo ""
echo "To terminate early:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
