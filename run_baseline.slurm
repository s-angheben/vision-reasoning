#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:a100.40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -J qwen_bird_baseline
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Load required modules (adjust based on your cluster setup)
# module load python/3.9
# module load cuda/11.8

# Create/activate venv and run baseline
[ ! -d "venv" ] && python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python qwen_bird/baseline.py
