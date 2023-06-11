#!/bin/bash
#
#SBATCH --job-name=finetune_inf_experiment
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 08:00:00
#SBATCH --cpus-per-task=5
#SBATCH -o logs/%j-finetune_inf_experiment_gpt2_large_normal.out
module load miniconda
conda activate memit

python fine_tune_gpt_inf.py --checkpoint=/work/pi_adrozdov_umass_edu/debanjanmond_umass_edu/memit_commonsense/finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/ --dataset=pep3k --datatype=normal --epochs=10 --learning_rate=8.304087815447582e-06 --log2_batch_size=3 --model=gpt2-large --train_data_file=../finetune/experiments/exp04/incorrect-data/gpt2-large_pep3k_normal.json
