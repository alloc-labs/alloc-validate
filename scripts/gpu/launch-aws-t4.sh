#!/usr/bin/env bash
set -euo pipefail

# Launch an AWS g4dn.xlarge (1x T4, 16GB VRAM) for GPU validation.
# On-demand: ~$0.53/hr. Full test cycle takes < 10 min → ~$0.09/run.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Default VPC with internet access
#   - Key pair created (or set KEY_NAME below)
#
# Usage:
#   bash scripts/gpu/launch-aws-t4.sh
#
# The instance auto-terminates after tests complete.

INSTANCE_TYPE="g4dn.xlarge"
# Deep Learning AMI (Ubuntu 20.04) — has CUDA, cuDNN, NVIDIA drivers pre-installed
# Update AMI ID for your region. This is us-east-1.
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

echo "=== alloc-validate GPU test starting ==="
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

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Run validations
USERDATA
)

# Append token if set
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
make matrix-quick

echo "=== alloc-validate GPU test complete ==="
date

# Auto-terminate
shutdown -h now
'

# Encode user data
ENCODED_DATA=$(echo "$USER_DATA" | base64)

echo "Launching $INSTANCE_TYPE in $REGION..."

# Build the AWS CLI command
CMD=(aws ec2 run-instances
    --region "$REGION"
    --image-id "$AMI_ID"
    --instance-type "$INSTANCE_TYPE"
    --instance-initiated-shutdown-behavior terminate
    --user-data "$ENCODED_DATA"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alloc-validate-gpu-test}]"
    --query "Instances[0].InstanceId"
    --output text
)

if [ -n "$KEY_NAME" ]; then
    CMD+=(--key-name "$KEY_NAME")
fi

INSTANCE_ID=$("${CMD[@]}")

echo "Instance launched: $INSTANCE_ID"
echo "Instance type: $INSTANCE_TYPE (1x T4, 16GB VRAM)"
echo "Cost: ~\$0.53/hr on-demand"
echo ""
echo "The instance will auto-terminate after tests complete."
echo ""
echo "To check progress:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].State.Name'"
echo ""
echo "To view logs (if you have a key pair):"
echo "  IP=\$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "  ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@\$IP 'tail -f /var/log/alloc-validate.log'"
echo ""
echo "To terminate early:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
