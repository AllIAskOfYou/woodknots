#!/bin/bash
srun --gres=gpu \
     --container-image=nvcr.io/nvidia/pytorch:26.04-py3 \
     --container-save=./containers/base.sqsh \
     bash -c "
        apt-get update -qq && apt-get install -y -qq \
            libgl1 libglx-mesa0 libglib2.0-0t64 libsm6 libxext6 libxrender-dev libgomp1 \
            && apt-get clean && rm -rf /var/lib/apt/lists/*
        pip install --no-cache-dir uv nnunetv2
     "