#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"
DATASET_SLUG="${DATASET_SLUG:-santa-2025-solver}"
KERNEL_SLUG="${KERNEL_SLUG:-santa-2025-solver}"
KAGGLE_STAGE_DIR="${KAGGLE_STAGE_DIR:-}"

usage() {
  echo "Usage: $0 {stage|render|render-batches|push-dataset|push-kernel|push-kernel-batch|push-kernel-batches|status|output|status-batches|output-batches}"
  echo "Env: KAGGLE_USERNAME (required), DATASET_SLUG, KERNEL_SLUG"
  echo "Batch env: BATCHES (comma-separated), KERNEL_PREFIX"
}

require_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
  fi
  if [[ -z "${KAGGLE_USERNAME}" ]]; then
    echo "KAGGLE_USERNAME is required." >&2
    exit 1
  fi
}

case "${1:-}" in
  stage)
    "${ROOT}/scripts/kaggle_stage.sh"
    ;;
  render)
    require_kaggle
    python3 "${ROOT}/scripts/render_kaggle_metadata.py" \
      --username "${KAGGLE_USERNAME}" \
      --dataset-slug "${DATASET_SLUG}" \
      --kernel-slug "${KERNEL_SLUG}" \
      --out-dir "${ROOT}/kaggle"
    cp "${ROOT}/kaggle/dataset-metadata.json" "${ROOT}/dataset-metadata.json"
    echo "Rendered metadata."
    ;;
  render-batches)
    require_kaggle
    python3 "${ROOT}/scripts/render_kaggle_batches.py" \
      --username "${KAGGLE_USERNAME}" \
      --dataset-slug "${DATASET_SLUG}" \
      --kernel-prefix "${KERNEL_PREFIX:-santa-2025-batch}" \
      --batches "${BATCHES:-batch_a,batch_b,batch_c,batch_d}"
    echo "Rendered batch kernel metadata."
    ;;
  push-dataset)
    require_kaggle
    TARGET_DIR="${ROOT}"
    if [[ -n "${KAGGLE_STAGE_DIR}" ]]; then
      TARGET_DIR="${KAGGLE_STAGE_DIR}"
    fi
    if [[ ! -f "${TARGET_DIR}/dataset-metadata.json" ]]; then
      echo "dataset-metadata.json not found in ${TARGET_DIR}. Run: $0 render (and $0 stage if staging)." >&2
      exit 1
    fi
    if kaggle datasets list -s "${KAGGLE_USERNAME}/${DATASET_SLUG}" | grep -q "${DATASET_SLUG}"; then
      kaggle datasets version -p "${TARGET_DIR}" -m "update" --dir-mode zip
    else
      kaggle datasets create -p "${TARGET_DIR}" --dir-mode zip
    fi
    ;;
  push-kernel)
    require_kaggle
    if [[ ! -f "${ROOT}/kaggle/kernel-metadata.json" ]]; then
      echo "kaggle/kernel-metadata.json not found. Run: $0 render" >&2
      exit 1
    fi
    kaggle kernels push -p "${ROOT}/kaggle"
    ;;
  push-kernel-batch)
    require_kaggle
    BATCH_NAME="${2:-}"
    if [[ -z "${BATCH_NAME}" ]]; then
      echo "Usage: $0 push-kernel-batch <batch_name>" >&2
      exit 1
    fi
    META="${ROOT}/kaggle/kernel-metadata.${BATCH_NAME}.json"
    if [[ ! -f "${META}" ]]; then
      echo "Missing ${META}. Run: $0 render-batches" >&2
      exit 1
    fi
    cp "${META}" "${ROOT}/kaggle/kernel-metadata.json"
    kaggle kernels push -p "${ROOT}/kaggle"
    ;;
  push-kernel-batches)
    require_kaggle
    BATCH_LIST="${BATCHES:-batch_a,batch_b,batch_c,batch_d}"
    IFS=',' read -r -a batch_array <<< "${BATCH_LIST}"
    for b in "${batch_array[@]}"; do
      "${0}" push-kernel-batch "${b}"
    done
    ;;
  status)
    require_kaggle
    kaggle kernels status "${KAGGLE_USERNAME}/${KERNEL_SLUG}"
    ;;
  status-batches)
    require_kaggle
    BATCH_LIST="${BATCHES:-batch_a,batch_b,batch_c,batch_d}"
    IFS=',' read -r -a batch_array <<< "${BATCH_LIST}"
    for b in "${batch_array[@]}"; do
      suffix="${b##*_}"
      kaggle kernels status "${KAGGLE_USERNAME}/${KERNEL_PREFIX:-santa-2025-batch}-${suffix}"
    done
    ;;
  output)
    require_kaggle
    OUT_DIR="${2:-${ROOT}/kaggle_output}"
    mkdir -p "${OUT_DIR}"
    kaggle kernels output "${KAGGLE_USERNAME}/${KERNEL_SLUG}" -p "${OUT_DIR}"
    ;;
  output-batches)
    require_kaggle
    OUT_DIR="${2:-${ROOT}/kaggle_output}"
    mkdir -p "${OUT_DIR}"
    BATCH_LIST="${BATCHES:-batch_a,batch_b,batch_c,batch_d}"
    IFS=',' read -r -a batch_array <<< "${BATCH_LIST}"
    for b in "${batch_array[@]}"; do
      suffix="${b##*_}"
      kaggle kernels output "${KAGGLE_USERNAME}/${KERNEL_PREFIX:-santa-2025-batch}-${suffix}" -p "${OUT_DIR}/${b}"
    done
    ;;
  *)
    usage
    exit 1
    ;;
esac
