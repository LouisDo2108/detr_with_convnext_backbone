#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab4
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=999:00:00

eval "$(conda shell.bash hook)"
conda activate detr

# cd /home/htluc/detr/coco-minitrain/src
# python sample_coco.py --coco_path "/home/htluc/datasets/coco" --save_file_name "instances_train2017_minicoco" --save_format "json" --sample_image_count 25000 --debug

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:/home/whatever/miniconda3/lib
# source ~/.bashrc

# nvidia-smi

# python -m torch.distributed.launch --nproc_per_node=1 --use_env \
# /home/htluc/detr/main.py \
# --coco_path /home/htluc/datasets/coco/

torchrun /home/htluc/detr/main.py \
--coco_path /home/htluc/datasets/coco/ \
--lr_drop 100 \
--lr_backbone 4e-3 \
--epochs 100 \
--batch_size 2 \
--output_dir /home/htluc/detr/experiments \
--num_workers 1
