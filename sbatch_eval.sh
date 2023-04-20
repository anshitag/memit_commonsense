#!/bin/bash
#
#SBATCH --job-name=gpt2-eval
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o experiments/logs/%j-eval_gpt2-xl_pep3k_normal_test_edited_object.out


python3 -m experiments.evaluate_finetune_model \
-m gpt2-xl \
-mp results/MEMIT_CS/run_001/edited_model \
-i commonsense_data/pep3k/test.json \
-o experiments/outputs/gpt2-xl_pep3k_normal_test_edited_object.json
