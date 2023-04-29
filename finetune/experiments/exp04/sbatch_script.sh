#!/bin/bash
#
#SBATCH --job-name=finetune_gpt
#SBATCH --partition=gypsum-m40
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=5
#SBATCH -o result/logs/fine_tune_large_inversed-slurm-%j.out
module load miniconda
conda activate memit

# python3 fine_tune_gpt.py --dataset=20q --datatype=normal --log2_batch_size=6 --model=gpt2-large --evaluation_only --run_name=captain-resistance-1374
python fine_tune_gpt.py --dataset=pep3k --datatype=inversed --epochs=10 --learning_rate=2.3917596836828713e-05 --log2_batch_size=5 --model=gpt2-large --save_model