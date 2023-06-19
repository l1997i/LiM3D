#!/bin/bash
CONDA_ENV="lim3d"
CUDA_VERSION="11.0"

CKPT_PATH="model/mycheckpoint.ckpt"
SAVE_DIR="your/save/dir"

source /etc/profile
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $CONDA_ENV
module unload cuda
module load cuda/$CUDA_VERSION
echo "HOSTNAME: $(hostname)"
echo "CUDA_VERSION: $CUDA_VERSION"
echo "CONDA_ENV: $CONDA_ENV"
echo "CKPT_PATH: $CKPT_PATH"
echo "SAVE_DIR: $SAVE_DIR"
wandb online

python save.py --config_path config/predict.yaml --dataset_config_path config/semantickitti.yaml --checkpoint_path $CKPT_PATH --save_dir $SAVE_DIR