#!/bin/bash
#
#SBATCH --job-name=finetune_k_experiment
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o result/logs/%j-finetune_k_experiment_gpt2_large_normal.out
module load miniconda
conda activate memit

python fine_tune_gpt_k.py --checkpoint=/work/pi_adrozdov_umass_edu/debanjanmond_umass_edu/memit_commonsense/finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/ --dataset=pep3k --datatype=normal --epochs=10 --learning_rate=4.739837590587367e-06 --log2_batch_size=5 --model=gpt2-large

