#!/bin/bash
CONDA_ENV="lim3d"
CUDA_VERSION="11.0"

source /etc/profile
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $CONDA_ENV
module unload cuda
module load cuda/$CUDA_VERSION
echo "HOSTNAME: $(hostname)"
echo "CUDA_VERSION: $CUDA_VERSION"
echo "CONDA_ENV: $CONDA_ENV"
wandb online

python train.py  --config_path config/training-reflec-full.yaml --dataset_config_path config/semantickitti.yaml
