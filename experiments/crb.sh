#!/bin/bash
CONDA_ENV="lim3d"
CUDA_VERSION="11.0"

## IMPORTANT: change crb_tag in crb.yaml !!!

####### SemanticKITTI, fea69, beta=-9.25, 1%
# CKPT_PATH="output/scribblekitti/20221026_164631/epoch=107-val_teacher_miou=86.81.ckpt"
# SAVE_DIR="save/20221026_164631"

####### ScribbleKITTI, fea69, beta=-9.25, 1%
# CKPT_PATH="output/scribblekitti/20221027_225243/epoch=182-val_teacher_miou=76.19.ckpt"
# SAVE_DIR="save/20221027_225243"

####### SemanticKITTI, fea69, beta=-5.72, 10%
# CKPT_PATH="output/scribblekitti/20221026_163008/epoch=50-val_teacher_miou=88.91.ckpt"
# SAVE_DIR="save/20221026_163008"

####### ScribbleKITTI, fea69, beta=-5.72, 10%
# CKPT_PATH="output/scribblekitti/20221027_224600/epoch=20-val_teacher_miou=80.87.ckpt"
# SAVE_DIR="save/20221027_224600"

####### SemanticKITTI, fea69, beta=-4, 20%
# CKPT_PATH="output/scribblekitti/20221024_212255/epoch=50-val_teacher_miou=89.53.ckpt"
# SAVE_DIR="save/20221024_212255"

####### SemanticKITTI, fea66, beta=-4, 20%
# CKPT_PATH="output/scribblekitti/20221028_122755/epoch=38-val_teacher_miou=89.10.ckpt"
SAVE_DIR="save/20221028_122755"

####### ScribbleKITTI, fea69, beta=-4, 20%
# CKPT_PATH="output/scribblekitti/20221027_224832/epoch=16-val_teacher_miou=81.77.ckpt"
# SAVE_DIR="save/20221027_224832"

####### SemanticKITTI, fea69, beta=-1.72, 50%
# CKPT_PATH="output/scribblekitti/20221024_134806/epoch=28-val_teacher_miou=90.29.ckpt"
# SAVE_DIR="save/20221024_134806"

####### ScribbleKITTI, fea69, beta=-1.72, 50%
# CKPT_PATH="output/scribblekitti/20221027_225828/epoch=5-val_teacher_miou=81.81.ckpt"
# SAVE_DIR="save/20221027_225828"

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
dpkg-query -L libnccl2

python -u crb.py --config_path config/crb.yaml --dataset_config_path config/semantickitti.yaml --save_dir $SAVE_DIR
