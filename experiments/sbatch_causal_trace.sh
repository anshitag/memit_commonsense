#!/bin/bash
#
#SBATCH --job-name=causal-tracing
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH -o experiments/logs/%j-gpt2-xl-pep3k-normal-whole-prompt.out
#SBATCH --exclude=gypsum-gpu043
#SBATCH --mem 50GB

module load miniconda
conda activate memit

ROOT_DIR=$(pwd)
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

python experiments/causal_tracing_custom_noising.py --model_name gpt2-xl --fact_file tracing/data/gpt2-xl_pep3k_normal.json  --checkpoint /work/pi_adrozdov_umass_edu/akshay_umass_edu/memit_commonsense/finetune/experiments/exp01/result/best-checkpoints/pep3k/gpt2-xl/normal/firm-sweep-16/ \
--dataset pep3k --datatype normal --experiment subject