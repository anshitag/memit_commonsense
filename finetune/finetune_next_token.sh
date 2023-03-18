#!/bin/bash
#
#SBATCH --job-name=finetune-twentyquestions
#SBATCH --partition=gypsum-titanx
#SBATCH --gres=gpu:1
#SBATCH -o logs/%j-twentyquestions-inversed-gpt2-medium.out
#SBATCH --exclude=gypsum-gpu043
#SBATCH --mem 50GB


module load miniconda
conda activate memit
python gpt2_finetune_next_token.py --model gpt2-medium --dataset 20q --datatype inversed