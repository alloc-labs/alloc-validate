#!/usr/bin/env bash
set -euo pipefail

# Launch a GCP g2-standard-4 (1x L4, 24GB VRAM) for GPU validation.
# Spot: ~$0.21/hr. On-demand: ~$0.70/hr. Full test cycle takes < 10 min → ~$0.04/run.
# Uses Spot VM by default (--provisioning-model=SPOT). ~70% cheaper than on-demand.
# Set GCP_ON_DEMAND=1 to use on-demand instead.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - A GCP project set (gcloud config set project <project-id>)
#   - Compute Engine API enabled
#
# Usage:
#   bash scripts/gpu/launch-gcp-l4.sh
#
# The instance auto-deletes after tests complete.

INSTANCE_NAME="${GCP_INSTANCE_NAME:-alloc-validate-l4}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="g2-standard-4"
# Deep Learning VM image — has CUDA, cuDNN, NVIDIA drivers pre-installed
IMAGE_FAMILY="common-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
ALLOC_TOKEN="${ALLOC_TOKEN:-}"
ON_DEMAND="${GCP_ON_DEMAND:-}"

if [ -z "$ALLOC_TOKEN" ]; then
    echo "WARN: ALLOC_TOKEN not set — will run free-tier only"
fi

# Build the startup script
STARTUP_SCRIPT=$(cat <<'STARTUP'
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

STARTUP
)

if [ -n "$ALLOC_TOKEN" ]; then
    STARTUP_SCRIPT+="
export ALLOC_TOKEN='$ALLOC_TOKEN'
make validate-full
"
else
    STARTUP_SCRIPT+="
make validate-free
"
fi

STARTUP_SCRIPT+='
make matrix-quick

echo "=== alloc-validate GPU test complete ==="
date

# Auto-delete this instance
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
INSTANCE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
'

echo "Launching $MACHINE_TYPE in $ZONE..."

STARTUP_FILE=$(mktemp)
echo "$STARTUP_SCRIPT" > "$STARTUP_FILE"
trap "rm -f '$STARTUP_FILE'" EXIT

CMD=(gcloud compute instances create "$INSTANCE_NAME"
    --zone="$ZONE"
    --machine-type="$MACHINE_TYPE"
    --accelerator="type=nvidia-l4,count=1"
    --image-family="$IMAGE_FAMILY"
    --image-project="$IMAGE_PROJECT"
    --boot-disk-size=100GB
    --maintenance-policy=TERMINATE
    --metadata-from-file="startup-script=$STARTUP_FILE"
    --scopes="compute-rw"
    --format="value(name,zone,status)"
)

if [ -z "$ON_DEMAND" ]; then
    CMD+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    PRICING="~\$0.21/hr spot"
else
    PRICING="~\$0.70/hr on-demand"
fi

"${CMD[@]}"

echo ""
echo "Instance launched: $INSTANCE_NAME"
echo "Instance type: $MACHINE_TYPE (1x L4, 24GB VRAM)"
echo "Cost: $PRICING"
echo ""
echo "The instance will auto-delete after tests complete."
echo ""
echo "To check progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command 'tail -f /var/log/alloc-validate.log'"
echo ""
echo "To delete early:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet"
