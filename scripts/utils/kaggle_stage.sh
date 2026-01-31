#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${KAGGLE_STAGE_DIR:-${ROOT}/.kaggle_stage}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# Copy minimal runtime files
cp -R "${ROOT}/src" "${STAGE_DIR}/src"
cp -R "${ROOT}/configs" "${STAGE_DIR}/configs"
cp -R "${ROOT}/kaggle" "${STAGE_DIR}/kaggle"
cp -R "${ROOT}/scripts" "${STAGE_DIR}/scripts"
cp "${ROOT}/requirements.txt" "${STAGE_DIR}/requirements.txt"
cp "${ROOT}/README.md" "${STAGE_DIR}/README.md"

# Include rendered dataset metadata if present
if [[ -f "${ROOT}/dataset-metadata.json" ]]; then
  cp "${ROOT}/dataset-metadata.json" "${STAGE_DIR}/dataset-metadata.json"
fi

# Remove any accidental caches
find "${STAGE_DIR}" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "Staged dataset at ${STAGE_DIR}"
