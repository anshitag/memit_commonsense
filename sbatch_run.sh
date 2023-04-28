#!/bin/bash
#
#SBATCH --job-name=cs_memit
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o results/logs/%j-gpt2-xl_compare_edited_object_outputs.out

# python experiments/prepare_editing_data.py -i finetune/experiments/exp03/result/outputs/gpt2-xl_pep3k_normal.json -svo svo_data/pep3k/test.json -o tracing/editing/gpt2-xl_pep3k_normal.json

python3 -m experiments.evaluate_cs \
--hparams_fname=gpt2-xl.json \
--alg_name=MEMIT_CS \
--edit_file=/work/anshitagupta_umass_edu/allenai_inp_study/memit_commonsense/tracing/editing/gpt2-xl_20q_normal_valid.json \
--inference_file=/work/anshitagupta_umass_edu/allenai_inp_study/memit_commonsense/commonsense_data/20q/valid.json \
--edit_token=verb \
--edit_location=last \
--model_name=gpt2-xl \
--model_checkpoint=/work/anshitagupta_umass_edu/allenai_inp_study/memit_commonsense/finetune/experiments/exp02/result/best-checkpoints/20q/gpt2-xl/normal/eternal-pond-1797/ \
--dataset_size_limit=1000 \
--num_edits=1000 \
--max_layer=3 \
--skip_generation_tests \
--wandb_active \
--save_summary

# python3 -m experiments.summarize --dir_name=MEMIT_CS --runs=run_000

# python3 -m experiments.evaluate_finetune_model \
# -m gpt2-xl \
# -mp results/MEMIT_CS/run_001/edited_model \
# -i commonsense_data/pep3k/test.json \
# -o experiments/outputs/gpt2-xl_pep3k_normal_test_edited_object.json

# python -m experiments.compare_edited_outputs -u experiments/outputs/gpt2-xl_pep3k_normal_test_unedited.json -e experiments/outputs/gpt2-xl_pep3k_normal_test_edited_object.json