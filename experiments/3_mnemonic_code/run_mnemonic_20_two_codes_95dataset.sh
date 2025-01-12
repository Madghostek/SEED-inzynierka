#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=finetuning-mnemonic-20op-only_two_codes-95-percent-dataset
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=finetuning-mnemonic-20op-twocode-95dataset #%j to jobid

DATASET_NAME=opacity20-twocode-95

echo runninig experiment on $DATASET_NAME

python3 utils/manage_dataset.py --poison-method mnemonic-code \
--dataset_name $DATASET_NAME \
--ratio 0.95 \
--opacity 0.2 \
--target_classes 2 \
--source_class 0 \
--seed 1234 \
--overwrite \


python3 ../src/main_incremental.py --approach finetuning --num-exemplars 2000 --nepochs 200 --batch-size 128 --num-workers 4 --datasets cifar_10_mnemonic --dataset_subtype $DATASET_NAME --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name "mnemonic_$DATASET_NAME" --seed 1234 --log tensorboard 

