#!/bin/bash
#SBATCH -A plgplgdyplomancipw-gpu-a100
#SBATCH -p plgrid-gpu-a100    
#SBATCH --job-name=SEED-test
#SBATCH --ntasks=4                    
#SBATCH --gpus=1
#SBATCH --mem=3gb                     # Job memory request
#SBATCH --time=03:00:00               # Time limit hrs:min:sec
#SBATCH --output=cleancifar1 # %j to jobid
cd ..
for NUM_EXPERTS in 5
do
  python3 src/main_incremental.py --approach seed --gmms 1 --max-experts $NUM_EXPERTS --use-multivariate --nepochs 200 --tau 3 --batch-size 128 --num-workers 4 --datasets cifar10 --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1 --alpha 0.99 --use-test-as-val --network resnet32 --extra-aug fetril --momentum 0.9 --exp-name cifar10test1 --seed 1
done

