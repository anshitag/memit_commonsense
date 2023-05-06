#!/bin/bash
#
#SBATCH --job-name=causal-tracing-zero-shot-subject
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH -o experiments/logs/%j-gpt2-xl-pep3k-normal-zero-shot-subject-pmi.out
#SBATCH --exclude=gypsum-gpu043
#SBATCH --mem 50GB

module load miniconda
conda activate memit

ROOT_DIR=$(pwd)
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

python experiments/causal_trace_zero_shot.py --fact_file tracing/data/zero_shot_gpt2-xl_pep3k_normal_valid_pmi.json --dataset pep3k --datatype normal --experiment subject --pmi