#sbatch --output mnist0-finetune-25data run_whitesq_grid_mnist_finetune.sh 1 0.25 0
sbatch --output 25data-100op-mnist/mnist2-finetune-25data run_whitesq_grid_mnist_finetune.sh 1 0.25 2
#sbatch --output mnist4-finetune-25data run_whitesq_grid_mnist_finetune.sh 1 0.25 4
#sbatch --output mnist6-finetune-25data run_whitesq_grid_mnist_finetune.sh 1 0.25 6
#sbatch --output mnist8-finetune-25data run_whitesq_grid_mnist_finetune.sh 1 0.25 8
