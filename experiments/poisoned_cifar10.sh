#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=30op_10subset
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=3gb                     # Job memory request
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=finetune_20op_10subset_1# %j is jobid
python3 ../src/main_incremental.py --approach finetuning --num-exemplars 2000 --nepochs 200 --batch-size 128 --num-workers 4 --datasets cifar_10_poisoned --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name opacity30 --seed 1 --log tensorboard
.
