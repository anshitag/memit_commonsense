#!/bin/bash
#
#SBATCH --job-name=cs_memit
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o results/logs/cs_memit_ft_verb_edit-%j.out

python experiments/prepare_editing_data.py -i finetune/experiments/exp01/result/outputs/gpt2-xl_20q_normal.json -svo svo_data/20q/test.json -o tracing/editing/gpt2-xl_20q_normal.json

# python3 -m experiments.evaluate_cs --save_model --hparams_fname=gpt2-large.json --alg_name=MEMIT_CS --noise_token=verb --model_name=gpt2-large --model_checkpoint=/work/pi_adrozdov_umass_edu/anshitagupta_umass_edu/best-checkpoints/20q/gpt2-xl/normal/quiet-flower-1063/ --dataset_size_limit=1000 --num_edits=1000 --skip_generation_tests

# python3 -m experiments.summarize --dir_name=MEMIT_CS --runs=run_034