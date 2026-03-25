#!/usr/bin/env bash
# Fallback for Docker Desktop cases where Buildx builds but doesn't load the image.
# This forces the classic builder which tends to be more reliable.
set -euo pipefail
cd "$(dirname "$0")/.."
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
exec docker compose -f docker-compose.mac.yml up --build "$@"
