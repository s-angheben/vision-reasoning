#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:a100.40:1
#SBATCH -n 4
#SBATCH -c 3
#SBATCH -N 1
#SBATCH -t 0-0:15
#SBATCH -J qwen_bird_baseline
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

module load CUDA/12.5.0

# Change to project directory
cd /home/samuele.angheben/vision-reasoning

source /home/samuele.angheben/venv/bin/activate

pip install -r requirements.txt

python qwen_bird/baseline.py

