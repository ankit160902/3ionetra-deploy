#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy-cloudrun.sh — Deploy 3ioNetra backend to Google Cloud Run
#
# Targets the LIVE service `ionetra-backend` (project ionetra, region
# asia-south1). Mirrors the actual production resource config so running
# this script will not regress min/max instances, VPC connector, gen2
# execution env, etc.
#
# IMPORTANT: This script previously targeted `3ionetra-backend` with a
# 2 vCPU / 2 GiB / max=4 config — that would have created an orphan
# service instead of updating the live deployment. Fixed Apr 2026.
#
# Prerequisites:
#   1. gcloud CLI installed & authenticated:   gcloud auth login
#   2. Project selected:                       gcloud config set project ionetra
#   3. Cloud Run + Artifact Registry APIs enabled
#   4. Secret Manager secrets already exist (managed manually):
#        GEMINI_API_KEY, MONGODB_URI, DATABASE_PASSWORD, REDIS_PASSWORD
#
# Usage:
#   ./deploy-cloudrun.sh                  # build, push, deploy with auto-bumped tag
#   ./deploy-cloudrun.sh --tag v3.7       # use a specific image tag
#   ./deploy-cloudrun.sh --skip-build     # deploy an already-pushed image (use --tag)
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configuration (matches the live ionetra-backend service) ──────────────
PROJECT_ID="ionetra"
REGION="asia-south1"
SERVICE_NAME="ionetra-backend"
REPO="asia-south1-docker.pkg.dev/${PROJECT_ID}/ionetra-docker"
IMAGE_NAME="backend"

# Live production resource config (verified Apr 7, 2026)
MEMORY="16Gi"
CPU="4"
MIN_INSTANCES=3
MAX_INSTANCES=100
TIMEOUT="300"
CONCURRENCY=80
EXECUTION_ENV="gen2"
VPC_CONNECTOR="ionetra-vpc"
VPC_EGRESS="private-ranges-only"

# ── Parse flags ────────────────────────────────────────────────────────────
TAG=""
SKIP_BUILD=false
for arg in "$@"; do
  case $arg in
    --tag) shift; TAG="$1"; shift ;;
    --tag=*) TAG="${arg#*=}" ;;
    --skip-build) SKIP_BUILD=true ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# Verify project + region match
ACTIVE_PROJECT=$(gcloud config get-value project 2>/dev/null) || error "No GCP project set."
[ "$ACTIVE_PROJECT" = "$PROJECT_ID" ] || warn "Active project is '$ACTIVE_PROJECT', expected '$PROJECT_ID'"

# Auto-bump tag if not specified: read latest live image tag and bump the patch
if [ -z "$TAG" ]; then
  CURRENT_TAG=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" \
    --format='value(spec.template.spec.containers[0].image)' 2>/dev/null \
    | sed 's|.*:||')
  if [[ "$CURRENT_TAG" =~ ^v([0-9]+)\.([0-9]+)$ ]]; then
    MAJOR="${BASH_REMATCH[1]}"
    MINOR="${BASH_REMATCH[2]}"
    TAG="v${MAJOR}.$((MINOR + 1))"
    info "Auto-bumped tag: $CURRENT_TAG → $TAG"
  else
    TAG="v$(date +%Y%m%d-%H%M%S)"
    warn "Couldn't parse current tag '$CURRENT_TAG' as vMAJOR.MINOR — using datestamp $TAG"
  fi
fi

IMAGE="${REPO}/${IMAGE_NAME}:${TAG}"
info "Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE_NAME  Image: $IMAGE"

# ── Step 1: Build & push image (unless --skip-build) ──────────────────────
if [ "$SKIP_BUILD" = false ]; then
  info "Building backend image with Cloud Build (this can take 8-15 minutes)..."
  gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions="_TAG=${TAG}" \
    .
  info "Image pushed: $IMAGE"
else
  info "Skipping build — assuming $IMAGE already exists in Artifact Registry"
fi

# ── Step 2: Deploy to Cloud Run ───────────────────────────────────────────
info "Deploying $IMAGE to Cloud Run service $SERVICE_NAME..."
info "Note: env vars and Secret Manager bindings are inherited from the previous revision."

gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --min-instances "$MIN_INSTANCES" \
  --max-instances "$MAX_INSTANCES" \
  --timeout "$TIMEOUT" \
  --concurrency "$CONCURRENCY" \
  --execution-environment "$EXECUTION_ENV" \
  --cpu-boost \
  --vpc-connector "$VPC_CONNECTOR" \
  --vpc-egress "$VPC_EGRESS" \
  --port 8080 \
  --allow-unauthenticated \
  --quiet

# ── Step 3: Verify ─────────────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" \
  --format='value(status.url)' 2>/dev/null)
LATEST_REVISION=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" \
  --format='value(status.latestReadyRevisionName)' 2>/dev/null)

info "Deployed!"
info "  URL:      $SERVICE_URL"
info "  Revision: $LATEST_REVISION"
info "  Image:    $IMAGE"
echo ""
info "Health check:"
echo "  curl ${SERVICE_URL}/api/health"
echo ""
info "Tail logs:"
echo "  gcloud run services logs tail $SERVICE_NAME --region=$REGION"
echo ""
info "Rollback to previous revision (if needed):"
echo "  gcloud run services update-traffic $SERVICE_NAME --to-revisions=<prev>=100 --region=$REGION"
