#!/usr/bin/env bash
set -euo pipefail

# GCP bootstrap wizard for alloc-validate GPU testing.
# Installs gcloud CLI, authenticates, creates/selects project, enables APIs,
# and checks GPU quota. Idempotent — skips already-completed steps on re-run.
#
# Env var overrides:
#   GCP_PROJECT_ID  — pre-set project ID (skip interactive selection)
#   GCP_ZONE        — default us-central1-a (matches launch scripts)
#   GCP_SKIP_AUTH   — skip auth step (for CI)
#
# Usage:
#   make setup-gcp
#   # or directly:
#   bash scripts/gpu/setup-gcp.sh

ZONE="${GCP_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
PROJECT_ID="${GCP_PROJECT_ID:-}"
SKIP_AUTH="${GCP_SKIP_AUTH:-}"

# --- Colors and helpers ---

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0
STEP=0

step() {
    STEP=$((STEP + 1))
    echo ""
    echo -e "${BLUE}${BOLD}[$STEP] $1${NC}"
    echo "────────────────────────────────────────"
}

ok() {
    echo -e "${GREEN}OK:${NC} $1"
    PASS=$((PASS + 1))
}

warn() {
    echo -e "${YELLOW}WARN:${NC} $1"
}

fail() {
    echo -e "${RED}ERROR:${NC} $1"
    FAIL=$((FAIL + 1))
}

# -------------------------------------------------------
# Step 1: Install gcloud CLI
# -------------------------------------------------------
step "Install gcloud CLI"

GCLOUD=""
if command -v gcloud &>/dev/null; then
    GCLOUD="$(command -v gcloud)"
elif [ -x "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD="$HOME/google-cloud-sdk/bin/gcloud"
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi

if [ -n "$GCLOUD" ]; then
    GCLOUD_VERSION=$("$GCLOUD" version 2>/dev/null | head -1 || echo "unknown")
    ok "gcloud already installed ($GCLOUD_VERSION)"
else
    echo "gcloud CLI not found — installing..."

    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "${OS}-${ARCH}" in
        Darwin-arm64)  PLATFORM="darwin-arm"      ;;
        Darwin-x86_64) PLATFORM="darwin-x86_64"   ;;
        Linux-x86_64)  PLATFORM="linux-x86_64"    ;;
        Linux-aarch64) PLATFORM="linux-arm"        ;;
        *)
            fail "Unsupported platform: $OS $ARCH"
            echo "Install gcloud manually: https://cloud.google.com/sdk/docs/install"
            exit 1
            ;;
    esac

    TARBALL="google-cloud-cli-${PLATFORM}.tar.gz"
    URL="https://dl.google.com/dl/cloudsdk/channels/rapid/${TARBALL}"

    echo "Downloading $TARBALL..."
    curl -sSL "$URL" -o "/tmp/$TARBALL"

    echo "Extracting to $HOME/google-cloud-sdk..."
    tar -xzf "/tmp/$TARBALL" -C "$HOME"
    rm -f "/tmp/$TARBALL"

    "$HOME/google-cloud-sdk/install.sh" --quiet --path-update true
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"

    if command -v gcloud &>/dev/null; then
        ok "gcloud installed successfully"
    else
        fail "gcloud installation failed"
        exit 1
    fi
fi

# -------------------------------------------------------
# Step 2: Authenticate
# -------------------------------------------------------
step "Authenticate with GCP"

if [ -n "$SKIP_AUTH" ]; then
    warn "Skipping auth (GCP_SKIP_AUTH is set)"
else
    ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)

    if [ -n "$ACTIVE_ACCOUNT" ]; then
        ok "Already authenticated as $ACTIVE_ACCOUNT"
    else
        echo "No active account found. Opening browser for login..."
        gcloud auth login
        ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
        if [ -n "$ACTIVE_ACCOUNT" ]; then
            ok "Authenticated as $ACTIVE_ACCOUNT"
        else
            fail "Authentication failed"
            exit 1
        fi
    fi
fi

# -------------------------------------------------------
# Step 3: Create or select project
# -------------------------------------------------------
step "Select GCP project"

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null || true)
fi

if [ -n "$PROJECT_ID" ] && [ "$PROJECT_ID" != "(unset)" ]; then
    gcloud config set project "$PROJECT_ID" 2>/dev/null
    ok "Using project: $PROJECT_ID"
else
    echo "No project selected."
    echo ""

    EXISTING=$(gcloud projects list --format="value(projectId)" 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        echo "Your existing projects:"
        echo "$EXISTING" | while read -r p; do echo "  - $p"; done
        echo ""
    fi

    read -rp "Enter project ID (or press Enter to create 'alloc-validate-gpu'): " INPUT_PROJECT
    PROJECT_ID="${INPUT_PROJECT:-alloc-validate-gpu}"

    if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        gcloud config set project "$PROJECT_ID"
        ok "Selected existing project: $PROJECT_ID"
    else
        echo "Creating project $PROJECT_ID..."
        gcloud projects create "$PROJECT_ID" --name="alloc-validate-gpu"
        gcloud config set project "$PROJECT_ID"
        ok "Created and selected project: $PROJECT_ID"
    fi
fi

# -------------------------------------------------------
# Step 4: Check billing
# -------------------------------------------------------
step "Check billing"

BILLING_ENABLED=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled)" 2>/dev/null || true)

if [ "$BILLING_ENABLED" = "True" ]; then
    BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || true)
    ok "Billing is enabled (account: ${BILLING_ACCOUNT:-unknown})"
else
    fail "Billing is not enabled for project $PROJECT_ID"
    echo ""
    echo "  Enable billing at:"
    echo "    https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
    echo ""
    echo "  GCP offers free credits:"
    echo "    - \$300 free trial credits for new accounts"
    echo "    - Google Cloud for Startups: \$2K-\$200K credits"
    echo "      https://cloud.google.com/startup"
    echo ""
    read -rp "Press Enter after enabling billing to continue..."

    BILLING_ENABLED=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingEnabled)" 2>/dev/null || true)
    if [ "$BILLING_ENABLED" = "True" ]; then
        ok "Billing is now enabled"
        FAIL=$((FAIL - 1))
    else
        warn "Billing still not detected — you can enable it later and re-run this script"
    fi
fi

# -------------------------------------------------------
# Step 5: Enable Compute Engine API
# -------------------------------------------------------
step "Enable Compute Engine API"

API_ENABLED=$(gcloud services list --enabled --filter="name:compute.googleapis.com" --format="value(name)" 2>/dev/null || true)

if [ -n "$API_ENABLED" ]; then
    ok "Compute Engine API already enabled"
else
    echo "Enabling Compute Engine API..."
    if gcloud services enable compute.googleapis.com; then
        ok "Compute Engine API enabled"
    else
        fail "Failed to enable Compute Engine API (is billing enabled?)"
    fi
fi

# -------------------------------------------------------
# Step 6: Check GPU quota
# -------------------------------------------------------
step "Check GPU quota"

L4_QUOTA=$(gcloud compute regions describe "$REGION" \
    --format="json(quotas)" 2>/dev/null | \
    python3 -c "
import sys, json
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'NVIDIA_L4' in q.get('metric', '').upper():
        print(int(q.get('limit', 0)))
        sys.exit(0)
print('0')
" 2>/dev/null || echo "0")

if [ "$L4_QUOTA" -ge 4 ] 2>/dev/null; then
    ok "L4 GPU quota in $REGION: $L4_QUOTA (sufficient for 4x L4)"
elif [ "$L4_QUOTA" -ge 1 ] 2>/dev/null; then
    warn "L4 GPU quota in $REGION: $L4_QUOTA (enough for 1x L4, need 4 for multi-GPU)"
    echo ""
    echo "  Request more quota at:"
    echo "    https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID&metric=NVIDIA_L4"
    echo ""
    echo "  Select the $REGION region and request a limit of 4+ GPUs."
else
    warn "L4 GPU quota in $REGION: ${L4_QUOTA:-0} (need at least 1)"
    echo ""
    echo "  Request GPU quota at:"
    echo "    https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID&metric=NVIDIA_L4"
    echo ""
    echo "  Select the $REGION region and request a limit of 4 GPUs."
    echo "  Quota requests are typically approved within minutes for small amounts."
fi

# -------------------------------------------------------
# Final: Validation summary
# -------------------------------------------------------
echo ""
echo "════════════════════════════════════════"
echo -e "${BOLD}GCP Setup Summary${NC}"
echo "════════════════════════════════════════"
echo -e "  Project:  ${BOLD}$PROJECT_ID${NC}"
echo -e "  Zone:     ${BOLD}$ZONE${NC}"
echo -e "  Region:   ${BOLD}$REGION${NC}"
echo -e "  PASS: ${GREEN}$PASS${NC} | FAIL: ${RED}$FAIL${NC}"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}${BOLD}RESULT: FAIL${NC} — fix the issues above and re-run: make setup-gcp"
    exit 1
fi

echo -e "${GREEN}${BOLD}RESULT: OK${NC} — GCP is ready for GPU testing"
echo ""
echo "Next steps:"
echo "  bash scripts/gpu/launch-gcp-l4.sh       # 1x L4 (24GB VRAM)"
echo "  bash scripts/gpu/launch-gcp-4xl4.sh     # 4x L4 (96GB VRAM)"
