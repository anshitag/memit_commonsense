#!/bin/bash
#
#SBATCH --job-name=causal-tracing
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH -o experiments/logs/%j-gpt2-large-pep3k-normal-severed-object.out
#SBATCH --exclude=gypsum-gpu043
#SBATCH --mem 50GB

module load miniconda
conda activate memit

ROOT_DIR=$(pwd)
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

python experiments/causal_tracing_severed.py --model_name gpt2-large --fact_file tracing/data/gpt2-large_pep3k_normal.json  --checkpoint finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/ \
--dataset pep3k --datatype normal --experiment object