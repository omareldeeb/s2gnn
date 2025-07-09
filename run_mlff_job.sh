#!/bin/bash
#SBATCH --job-name=mlff-debug
#SBATCH --output=tests/results/mlff/slurm-%j.out
#SBATCH --error=tests/results/mlff/slurm-%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

cd $SLURM_SUBMIT_DIR

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate s2gnn

# Optional: confirm GPU
# nvidia-smi

# Run your script
python main.py --cfg configs/mlff/mlff-gemnet-layers.yaml \
               out_dir tests/results/mlff \
               wandb.use True \
               seed 1


# python main.py --cfg configs/mlff/mlff-coll.yaml \
#                out_dir tests/results/mlff \
#                wandb.use False \
#                seed 1
