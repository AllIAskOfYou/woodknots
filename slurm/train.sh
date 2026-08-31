#!/bin/bash
#SBATCH --job-name=train_lightning           # Job name
#SBATCH --partition=frida               # Partition (queue) to submit to
#SBATCH --gres=gpu:L4                 # Request 1 A100 GPU
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --mem=32G                       # Memory
#SBATCH --time=02:00:00                 # Time limit (6 hours)
#SBATCH --output=slurm_logs/%j.out                 # Standard output log (%j = job ID)
#SBATCH --error=slurm_logs/%j.err                  # Standard error log
#SBATCH --ntasks=1                      # Single task

# Enable strict error handling
set -e  # Exit on error
set -u  # Exit on undefined variable

# Print start time and node info
echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================="


# Get the absolute path of the current directory
PROJECT_DIR=$(pwd)
echo "Project directory: ${PROJECT_DIR}"

# Step 2: Run the container with all necessary commands
# We use a single srun command with --container-image and execute a bash script inside
echo "Starting container and installing dependencies..."

srun --container-image=nvcr.io/nvidia/pytorch:26.04-py3 \
     --container-mounts=${PROJECT_DIR}:${PROJECT_DIR} \
     --container-workdir=${PROJECT_DIR} \
     bash -c "
        # Inside the container
        echo '=== Inside container ==='
        
        # Install OpenCV system dependencies
        echo 'Installing system dependencies...'
        apt-get update -qq && \
        apt-get install -y -qq \
            libgl1 \
            libglx-mesa0 \
            libglib2.0-0t64 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            && apt-get clean \
            && rm -rf /var/lib/apt/lists/*
        
        # Print GPU info
        echo 'GPU Information:'
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        
        # Run the training script
        echo 'Starting Lightning training...'
        uv run scripts/train.py
        
        echo 'Training completed at: $(date)'
     "

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