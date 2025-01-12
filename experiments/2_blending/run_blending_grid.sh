#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=lwf-blending
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=05:00:00               # Time limit hrs:min:sec

DATASET_NAME=blending_$1op_$2data

echo runninig experiment on $DATASET_NAME

python3 ../utils/manage_dataset.py --poison-method blend-random \
--dataset_name $DATASET_NAME \
--ratio $2 \
--opacity $1 \
--target_classes 0 \
--source_class 2 \
--seed 1234
#--overwrite


python3 ../../src/main_incremental.py --approach lwf --lamb 10 --nepochs 200 --batch-size 128 --num-workers 4 --datasets cifar_10_blending --dataset_subtype $DATASET_NAME --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name "blending_$DATASET_NAME" --seed 1234 --log tensorboard 

