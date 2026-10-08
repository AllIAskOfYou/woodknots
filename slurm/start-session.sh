#!/bin/bash
srun -p dev --cpus-per-task=8 --mem=32G --time=06:00:00 \
  --container-image=./containers/base.sqsh \
  --container-mounts=${PWD}:"/workspace" \
  --container-workdir=/workspace \
  --pty bash -c "exec bash"