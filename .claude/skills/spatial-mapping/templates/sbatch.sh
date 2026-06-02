#!/usr/bin/env bash
# Slurm launcher for cell2location step2_spatial_mapping.ipynb
#
# Usage:
#   bash sbatch.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]
#
# Example (1-chunk run):
#   bash sbatch.sh 0 0 30000 /path/to/signatures.csv /path/to/spatial.h5ad my_run
#
# Example (4-chunk run, submit one job array):
#   for i in 0 1 2 3; do bash sbatch.sh $i 0 30000 sig.csv spatial.h5ad my_run 4; done

set -euo pipefail

# === CONFIG (edit for your cluster) ===
PARTITION="${C2L_PARTITION:-gpu}"
MEM="${C2L_MEM:-100G}"
GPU_MEM_GB="${C2L_GPU_MEM_GB:-80}"
NCPU="${C2L_NCPU:-4}"
TIME="${C2L_TIME:-08:00:00}"          # 8h default; bump for large datasets
GRES="${C2L_GRES:-gpu:1}"              # use `gpu:a100:1` etc. on partitioned clusters

# === PARSE ARGS ===
training_batch="${1:?usage: bash sbatch.sh <training_batch> <seed> <max_epochs> <signatures_csv> <spatial_h5ad> <output_name> [<n_chunks>]}"
seed="${2:?seed}"
max_epochs="${3:?max_epochs}"
signatures_csv="${4:?signatures_csv}"
spatial_h5ad="${5:?spatial_h5ad}"
output_name="${6:?output_name}"
n_chunks="${7:-1}"

# === BUILD PAPERMILL COMMAND ===
template="$(cd "$(dirname "$0")" && pwd)/step2_spatial_mapping.ipynb"
output_dir="${C2L_OUTPUT_DIR:-./spatial_mapping_output}"
output_nb="${output_dir}/${output_name}_chunk${training_batch}.ipynb"
mkdir -p "$output_dir"

# === SLURM SUBMISSION (heredoc with sbatch directives) ===
sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=c2l_${output_name}_chunk${training_batch}
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${NCPU}
#SBATCH --mem=${MEM}
#SBATCH --gres=${GRES}
#SBATCH --time=${TIME}
#SBATCH --output=${output_dir}/%j.gpu.out
#SBATCH --error=${output_dir}/%j.gpu.err

set -euo pipefail
papermill ${template} ${output_nb} \\
    -p training_batch ${training_batch} \\
    -p n_chunks ${n_chunks} \\
    -p seed ${seed} \\
    -p max_epochs ${max_epochs} \\
    -p signatures_csv ${signatures_csv} \\
    -p spatial_h5ad_path ${spatial_h5ad} \\
    -p output_dir ${output_dir} \\
    -p output_name ${output_name}
EOF
