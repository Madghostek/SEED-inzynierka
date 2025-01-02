#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=LWF-30op_20lam
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=3gb                     # Job memory request
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=30op_20lamb# %j to jobid
python3 ../src/main_incremental.py --approach lwf --lamb 20 --nepochs 200 --batch-size 128 --num-workers 4 --datasets cifar_10_poisoned --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name opacity50 --seed 1 --log tensorboard 

