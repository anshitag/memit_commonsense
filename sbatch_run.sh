#!/bin/bash
#
#SBATCH --job-name=cs_memit
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o results/logs/%j-gpt2-xl_compare_edited_object_outputs.out

# python experiments/prepare_editing_data.py -i finetune/experiments/exp03/result/outputs/gpt2-xl_pep3k_normal.json -svo svo_data/pep3k/test.json -o tracing/editing/gpt2-xl_pep3k_normal.json

# python -m experiments.evaluate_cs --save_model --cs_file tracing/editing/gpt2-xl_pep3k_normal.json --hparams_fname=gpt2-xl_object.json --alg_name=MEMIT_CS --noise_token=object --model_name=gpt2-xl --model_checkpoint=finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-xl/normal/grateful-vortex-1814 --dataset_size_limit=1000 --num_edits=1000 --skip_generation_tests

# python3 -m experiments.summarize --dir_name=MEMIT_CS --runs=run_000

# python3 -m experiments.evaluate_finetune_model \
# -m gpt2-xl \
# -mp results/MEMIT_CS/run_001/edited_model \
# -i commonsense_data/pep3k/test.json \
# -o experiments/outputs/gpt2-xl_pep3k_normal_test_edited_object.json

# python -m experiments.compare_edited_outputs -u experiments/outputs/gpt2-xl_pep3k_normal_test_unedited.json -e experiments/outputs/gpt2-xl_pep3k_normal_test_edited_object.json