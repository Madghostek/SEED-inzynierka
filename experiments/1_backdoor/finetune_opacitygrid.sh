sbatch --output opacitygrid2-finetune/mnist2-100op-100data run_whitesq_grid_mnist_finetune.sh 1 1 2
sbatch --output opacitygrid2-finetune/mnist2-80op-100data run_whitesq_grid_mnist_finetune.sh 0.8 1 2
sbatch --output opacitygrid2-finetune/mnist2-60op-100data run_whitesq_grid_mnist_finetune.sh 0.6 1 2
sbatch --output opacitygrid2-finetune/mnist2-40op-100data run_whitesq_grid_mnist_finetune.sh 0.4 1 2
sbatch --output opacitygrid2-finetune/mnist2-20op-100data run_whitesq_grid_mnist_finetune.sh 0.2 1 2
# opacity ratio target_class
