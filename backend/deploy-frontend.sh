#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy-frontend.sh — Deploy 3ioNetra frontend to Google Cloud Run
#
# Targets the LIVE service `ionetra-frontend` (project ionetra, region
# asia-south1). Build context is the sibling ../frontend directory
# (multi-stage Next.js Dockerfile).
#
# Usage:
#   ./deploy-frontend.sh                # build, push, deploy with auto-bumped tag
#   ./deploy-frontend.sh --tag v3.2     # use a specific image tag
#   ./deploy-frontend.sh --skip-build   # deploy an already-pushed image (use --tag)
# ---------------------------------------------------------------------------
set -euo pipefail

# ── Configuration (matches the live ionetra-frontend service) ─────────────
PROJECT_ID="ionetra"
REGION="asia-south1"
SERVICE_NAME="ionetra-frontend"
REPO="asia-south1-docker.pkg.dev/${PROJECT_ID}/ionetra-docker"
IMAGE_NAME="frontend"

# Live production resource config (verified Apr 7, 2026)
MEMORY="512Mi"
CPU="1"
MIN_INSTANCES=0
MAX_INSTANCES=10
TIMEOUT="60"
CONCURRENCY=200

# Frontend dir is sibling of backend/ where this script lives
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FRONTEND_DIR="${SCRIPT_DIR}/../frontend"

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

[ -d "$FRONTEND_DIR" ] || error "Frontend directory not found at: $FRONTEND_DIR"

ACTIVE_PROJECT=$(gcloud config get-value project 2>/dev/null) || error "No GCP project set."
[ "$ACTIVE_PROJECT" = "$PROJECT_ID" ] || warn "Active project is '$ACTIVE_PROJECT', expected '$PROJECT_ID'"

# Auto-bump tag if not specified
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
  info "Building frontend image with Cloud Build (this can take 3-5 minutes)..."
  ( cd "$FRONTEND_DIR" && gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions="_TAG=${TAG},_API_URL=https://ionetra-backend-688398835360.asia-south1.run.app" \
    . )
  info "Image pushed: $IMAGE"
else
  info "Skipping build — assuming $IMAGE already exists in Artifact Registry"
fi

# ── Step 2: Deploy to Cloud Run ───────────────────────────────────────────
info "Deploying $IMAGE to Cloud Run service $SERVICE_NAME..."

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
  --cpu-boost \
  --port 3000 \
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
info "Smoke check:"
echo "  curl -I $SERVICE_URL"
echo ""
info "Rollback to previous revision (if needed):"
echo "  gcloud run services update-traffic $SERVICE_NAME --to-revisions=<prev>=100 --region=$REGION"
