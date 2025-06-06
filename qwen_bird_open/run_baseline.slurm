#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p edu-long
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-10:00

module load CUDA/12.5.0

cd /home/samuele.angheben/vision-reasoning

source /home/samuele.angheben/vr_env/bin/activate

# only first time
# pip install -r requirements.txt

python qwen_bird_open/baseline.py


