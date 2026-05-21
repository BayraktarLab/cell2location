#!/usr/bin/env bash
# Local single-GPU launcher for cell2location step2_spatial_mapping.ipynb
#
# Usage:
#   bash run_local.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]
#
# For 1-chunk runs (default):
#   bash run_local.sh 0 0 30000 /path/to/signatures.csv /path/to/spatial.h5ad my_run

set -euo pipefail

# === CONFIG ===
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# === PARSE ARGS ===
training_batch="${1:?usage: bash run_local.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]}"
seed="${2:?seed}"
max_epochs="${3:?max_epochs}"
signatures_csv="${4:?signatures_csv}"
spatial_h5ad="${5:?spatial_h5ad}"
output_name="${6:?output_name}"
n_chunks="${7:-1}"

template="$(cd "$(dirname "$0")" && pwd)/step2_spatial_mapping.ipynb"
output_dir="${C2L_OUTPUT_DIR:-./spatial_mapping_output}"
output_nb="${output_dir}/${output_name}_chunk${training_batch}.ipynb"
mkdir -p "$output_dir"

echo "Running locally (GPU=${CUDA_VISIBLE_DEVICES}):"
echo "  template:       ${template}"
echo "  output:         ${output_nb}"
echo "  training_batch: ${training_batch} / ${n_chunks}"

papermill "${template}" "${output_nb}" \
    -p training_batch "${training_batch}" \
    -p n_chunks "${n_chunks}" \
    -p seed "${seed}" \
    -p max_epochs "${max_epochs}" \
    -p signatures_csv "${signatures_csv}" \
    -p spatial_h5ad_path "${spatial_h5ad}" \
    -p output_dir "${output_dir}" \
    -p output_name "${output_name}"
