#!/usr/bin/env bash
set -euo pipefail

project_root=/home/rlc123/ForceSmolVLA
dataset_root=${project_root}/datasets/task2_lerobotv3
manifest=${dataset_root}/conversion_manifest.json

while [[ ! -f "${manifest}" ]]; do
  echo "$(date --iso-8601=seconds) waiting_for_complete_conversion_manifest"
  sleep 30
done

while :; do
  gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  [[ -z "${gpu_processes}" ]] && break
  echo "$(date --iso-8601=seconds) waiting_for_gpu_compute_processes"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  sleep 30
done

cd "${project_root}"
exec env \
  PYTHONPATH=${project_root}/src \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=42 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  HF_HOME=${project_root}/.cache/huggingface \
  HF_HUB_CACHE=${project_root}/.cache/huggingface/hub \
  HF_DATASETS_CACHE=${project_root}/.cache/huggingface/datasets \
  TRANSFORMERS_CACHE=${project_root}/.cache/huggingface/transformers \
  CUDA_VISIBLE_DEVICES=0 \
  /home/rlc123/anaconda3/envs/forcesmolvla/bin/python \
  ${project_root}/tools/train_task2_full_gpu.py
