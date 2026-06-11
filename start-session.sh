#!/bin/bash
srun -p dev --gres=gpu:A100_80GB --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --container-image=nvcr.io/nvidia/pytorch:26.04-py3 \
  --container-mounts=${PWD}:${PWD} \
  --container-workdir=${PWD} \
  --pty bash 
  
  # apt-get update && apt-get install -y \
  #   libgl1 \
  #   libglx-mesa0 \
  #   libglib2.0-0t64 \
  #   libsm6 \
  #   libxext6 \
  #   libxrender-dev \
  #   libgomp1 \
  #   && apt-get clean \
  #   && rm -rf /var/lib/apt/lists/*