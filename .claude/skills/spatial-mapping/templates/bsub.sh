#!/usr/bin/env bash
# LSF launcher for cell2location step2_spatial_mapping.ipynb
# Based on the cell2state_embryo papermill workflow.
#
# Usage:
#   bash bsub.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]
#
# Example (1-chunk run on 80GB A100):
#   bash bsub.sh 0 0 30000 /path/to/signatures.csv /path/to/spatial.h5ad my_run
#
# Example (4-chunk run, submit one per chunk):
#   for i in 0 1 2 3; do bash bsub.sh $i 0 30000 sig.csv spatial.h5ad my_run 4; done

set -euo pipefail

# === CONFIG (edit for your cluster) ===
QUEUE="${C2L_QUEUE:-gpu-normal}"
MEM_MB="${C2L_MEM_MB:-100000}"          # 100 GB host RAM
GPU_MEM_MB="${C2L_GPU_MEM_MB:-80000}"   # 80 GB GPU
NCPU="${C2L_NCPU:-4}"

# === PARSE ARGS ===
training_batch="${1:?usage: bash bsub.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]}"
seed="${2:?seed}"
max_epochs="${3:?max_epochs}"
signatures_csv="${4:?signatures_csv}"
spatial_h5ad="${5:?spatial_h5ad}"
output_name="${6:?output_name}"
n_chunks="${7:-1}"

# === BUILD PAPERMILL COMMAND ===
template="$(dirname "$0")/step2_spatial_mapping.ipynb"
output_dir="${C2L_OUTPUT_DIR:-./spatial_mapping_output}"
output_nb="${output_dir}/${output_name}_chunk${training_batch}.ipynb"
mkdir -p "$output_dir"

pmcmd="papermill ${template} ${output_nb} \
    -p training_batch ${training_batch} \
    -p n_chunks ${n_chunks} \
    -p seed ${seed} \
    -p max_epochs ${max_epochs} \
    -p signatures_csv ${signatures_csv} \
    -p spatial_h5ad_path ${spatial_h5ad} \
    -p output_dir ${output_dir} \
    -p output_name ${output_name}"

echo "Submitting:"
echo "  training_batch=${training_batch} / ${n_chunks}"
echo "  seed=${seed}"
echo "  max_epochs=${max_epochs}"
echo "  signatures_csv=${signatures_csv}"
echo "  spatial_h5ad=${spatial_h5ad}"
echo "  output_dir=${output_dir}"
echo "  output_name=${output_name}"
echo ""
echo "Command: ${pmcmd}"

# === LSF SUBMISSION ===
bsub -q "${QUEUE}" -n${NCPU} -M${MEM_MB} \
    -R"select[mem>${MEM_MB}] rusage[mem=${MEM_MB}] span[hosts=1]" \
    -gpu "mode=shared:j_exclusive=yes:gmem=${GPU_MEM_MB}:num=1" \
    -e "${output_dir}/%J.gpu.err" -o "${output_dir}/%J.gpu.out" \
    ${pmcmd}
