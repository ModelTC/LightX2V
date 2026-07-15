#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile_deploy"

CURRENT_BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Error: must be on main branch to build (current: $CURRENT_BRANCH)"
    exit 1
fi

AUTO_BASE_TAG=$(grep -oP 'ARG BASE_TAG=\K\S+' "$DOCKERFILE" | head -1)
REGISTRY="lightx2v/lightx2v"
BASE_TAG="$AUTO_BASE_TAG"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag) BASE_TAG="$2"; shift 2 ;;
        -r|--registry) REGISTRY="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-t base_tag] [-r registry]"
            echo "  -t, --tag       Base image tag (default: $AUTO_BASE_TAG)"
            echo "  -r, --registry  Registry prefix (default: $REGISTRY)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DATE_TAG=$(date +%y%m%d)
GIT_HASH=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
IMAGE_TAG="${REGISTRY}:server-${BASE_TAG}-${DATE_TAG}-${GIT_HASH}"

echo "Base image tag: $BASE_TAG"
echo "Building image:  $IMAGE_TAG"
docker buildx build --platform linux/amd64 -f "$DOCKERFILE" -t "$IMAGE_TAG" --build-arg BASE_TAG="$BASE_TAG" "$REPO_ROOT"
