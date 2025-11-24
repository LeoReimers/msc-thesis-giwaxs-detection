#!/usr/bin/env bash
set -euo pipefail

# --- Env (falls noch nicht aktiv) ---
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/miniconda3/envs/dino"

export CUDA_VER=11.8
export CUDA_HOME=/usr/local/cuda-$CUDA_VER
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$WORK/DINO/models/dino/ops:$WORK/DINO:$PYTHONPATH"

cd "$WORK/DINO"

# --- Training (sauber mit echten Zeilenumbrüchen, keine NBSPs) ---
exec  python -m torch.distributed.run \
  --nproc_per_node=1 \
  --standalone \
  main.py \
  --config_file config/DINO/DINO_4scale_swin.py \
  --options window_size_h=4 window_size_w=4 \
  --output_dir output/WindowSize_4x4 \
  --device cuda
