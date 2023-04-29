#!/bin/bash
#
#SBATCH --job-name=finetune_incorrect_dev_gpt
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o result/logs/finetune_incorrect_dev_gpt_large_normal-%j.out
module load miniconda
conda activate memit

python fine_tune_gpt.py --dataset=pep3k --datatype=inversed --epochs=10 --learning_rate=2.3917596836828713e-05 --log2_batch_size=5 --model=gpt2-large --save_model \
--checkpoint /work/pi_adrozdov_umass_edu/debanjanmond_umass_edu/memit_commonsense/finetune/experiments/exp03/result/best-checkpoints/pep3k/gpt2-large/normal/