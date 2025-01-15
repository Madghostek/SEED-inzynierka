#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=lwf-whitesq
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=05:00:00               # Time limit hrs:min:sec

DATASET_NAME=local_whitesq_1op_1data_class$1

echo runninig experiment on $DATASET_NAME

python3 ../datasetTool/create_dataset.py --poison-method white-square \
--dataset_name $DATASET_NAME \
--ratio 1 \
--opacity 1 \
--target_classes $1 \
--seed 1234 \
--overwrite \
--poison_test_set


python3 ../../src/main_incremental.py --approach lwf --lamb 10 --nepochs 200 --batch-size 128 --num-workers 4 --datasets cifar_10_white_square --dataset_subtype $DATASET_NAME --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name "whitesq_$DATASET_NAME" --seed 1234 --log tensorboard 

