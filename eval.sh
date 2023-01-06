#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab3
#SBATCH --mem-per-cpu=4GB

eval "$(conda shell.bash hook)"
conda activate detr

# nvidia-smi
# nvcc

torchrun /home/htluc/detr/main.py --batch_size 2 --no_aux_loss --eval \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
    --coco_path /home/htluc/datasets/coco/

# torchrun /home/htluc/detr/main.py --batch_size 2 --no_aux_loss --eval \
#     --resume /home/htluc/detr/checkpoint0049.pth \
#     --coco_path /home/htluc/datasets/coco/