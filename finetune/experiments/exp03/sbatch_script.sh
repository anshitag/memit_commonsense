#!/bin/bash
#
#SBATCH --job-name=finetune_gpt
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o result/fine_tune_large_normal-slurm-%j.out
export HF_HOME=/work/anshitagupta_umass_edu/.cache/huggingface/

# python3 fine_tune_gpt.py --dataset=20q --datatype=normal --log2_batch_size=6 --model=gpt2-large --evaluation_only --run_name=captain-resistance-1374

python3 fine_tune_gpt.py --dataset=20q --datatype=normal --epochs=10 --learning_rate=6.333565960404303e-05 --log2_batch_size=6 --model=gpt2-large --save_model