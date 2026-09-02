#!/bin/bash
DATE=$(date +%Y-%m-%d)

# Enable strict error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Get the absolute path of the current directory
PROJECT_DIR=$(pwd)
echo "Project directory: ${PROJECT_DIR}"

local inner_cmd="uv run nnunet/run.py env=cluster experiment=${1} experiment.fold=${2}"
    srun \
        --gres=gpu --cpus-per-task=8 --mem=32G --time=12:00:00 --ntasks=1 \
        --output=slurm_logs/${DATE}/%j.out --error=slurm_logs/${DATE}/%j.err \
        --container-image=./containers/base.sqsh \
        --container-mounts="${PWD}:/workspace" \
        --container-workdir=/workspace \
        bash -c "${inner_cmd}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Job completed successfully at: $(date)"
    echo "========================================="
else
    echo "========================================="
    echo "Job FAILED at: $(date)"
    echo "Check the error log for details"
    echo "========================================="
    exit 1
fi