#!/bin/bash
#SBATCH --job-name=bert_section
#SBATCH -A research
#SBATCH -c 40
#SBATCH -o bert_section.out
#SBATCH --gres=gpu:4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#python bert_section.py

python3 train.py
