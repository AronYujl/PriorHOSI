#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
HOI_BATCH="${2:-512}"
HSI_BATCH="${3:-320}"
LOG_DIR="${ROOT_DIR}/results/priors/logs/${TAG}"
HOI_OUTPUT="${ROOT_DIR}/results/priors/${TAG}/hoi_prior"
HSI_OUTPUT="${ROOT_DIR}/results/priors/${TAG}/hsi_prior"

if [[ -e "${HOI_OUTPUT}/run_manifest.json" || -e "${HSI_OUTPUT}/run_manifest.json" ]]; then
  printf 'Experiment tag already exists: %s\n' "${TAG}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${HOI_OUTPUT}" "${HSI_OUTPUT}"

printf 'tag=%s\nhoi_batch=%s\nhsi_batch=%s\nstarted_at=%s\n' \
  "${TAG}" "${HOI_BATCH}" "${HSI_BATCH}" "$(date --iso-8601=seconds)" \
  | tee "${LOG_DIR}/launcher.log"

(
  cd "${ROOT_DIR}/code"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
    conda run -n infbagel --no-capture-output \
    python train_prior.py --config-name config_train_hoi_prior \
    per_device_batch_size="${HOI_BATCH}" output_dir="${HOI_OUTPUT}"
) 2>&1 | tee "${LOG_DIR}/hoi_terminal.log" &
HOI_PID=$!

(
  cd "${ROOT_DIR}/code"
  env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5,6,7 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 \
    conda run -n infbagel --no-capture-output \
    python train_prior.py --config-name config_train_hsi_prior \
    per_device_batch_size="${HSI_BATCH}" output_dir="${HSI_OUTPUT}"
) 2>&1 | tee "${LOG_DIR}/hsi_terminal.log" &
HSI_PID=$!

wait "${HOI_PID}"
HOI_STATUS=$?
wait "${HSI_PID}"
HSI_STATUS=$?

printf 'finished_at=%s\nhoi_status=%s\nhsi_status=%s\n' \
  "$(date --iso-8601=seconds)" "${HOI_STATUS}" "${HSI_STATUS}" \
  | tee -a "${LOG_DIR}/launcher.log"

if (( HOI_STATUS != 0 || HSI_STATUS != 0 )); then
  exit 1
fi
