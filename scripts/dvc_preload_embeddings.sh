#!/usr/bin/env bash
# Used by dvc.yaml so `dvc repro` picks up .venv when present (and works in Docker with python3).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" -m improved_pipelines.preload_embeddings "$@"
else
  exec python3 -m improved_pipelines.preload_embeddings "$@"
fi
