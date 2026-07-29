#!/bin/bash
srun -p dev --gres=gpu --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --container-image=nvcr.io/nvidia/pytorch:26.04-py3 \
  --container-mounts=${PWD}:${PWD} \
  --container-workdir=${PWD} \
  --container-env=NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  --pty bash -c "apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libglvnd0 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && set -a \
    && source secrets.env \
    && set +a \
    && exec bash"