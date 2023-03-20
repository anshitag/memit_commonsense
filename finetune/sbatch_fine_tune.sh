#!/bin/bash
#
#SBATCH --job-name=finetune_gpt
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=20
#SBATCH -o results/logs/20q_fine_tune_gpt2-large-normal-slurm-%j.out

module load miniconda
conda activate memit

python3 fine_tune_gpt.py --model gpt2-large --dataset 20q --datatype normal --epochs 2 --batch_size 64

#for evaluation only:
# python3 fine_tune_gpt.py --model gpt2-large --dataset 20q --datatype normal --evaluation_only