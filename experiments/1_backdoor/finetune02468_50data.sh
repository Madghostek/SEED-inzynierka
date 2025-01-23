# sbatch --output mnist0-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 0
# sbatch --output mnist2-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 2
# sbatch --output mnist4-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 4
# sbatch --output mnist6-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 6
# sbatch --output mnist8-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 8

sbatch --output 50data-100op-mnist/mnist2-finetune-50data run_whitesq_grid_mnist_finetune.sh 1 0.5 2