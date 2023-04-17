#!/bin/bash
#
#SBATCH --job-name=cs_memit
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o results/logs/cs_memit_ft_subject_edit-%j.out

module load miniconda
conda activate memit
# python experiments/prepare_editing_data.py -i finetune/experiments/exp01/result/outputs/gpt2-xl_20q_normal.json -svo svo_data/20q/test.json -o tracing/editing/gpt2-xl_20q_normal.json

python -m experiments.evaluate_cs --save_model --cs_file tracing/editing/gpt2-large_pep3k_normal.json --hparams_fname=gpt2-large_subject.json --alg_name=MEMIT_CS --noise_token=subject --model_name=gpt2-large --model_checkpoint=finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/ --dataset_size_limit=1000 --num_edits=1000 --skip_generation_tests

# python3 -m experiments.summarize --dir_name=MEMIT_CS --runs=run_034