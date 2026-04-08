#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-no-film
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_train_no_film_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_train_no_film_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── Ablation: ANP without FiLM conditioning ─────────────────────────
#
# All defaults match the best model (anp_all_profiles_20260325_175639)
# except FiLM is disabled. This isolates the contribution of
# cross-attention + latent paths without theta-based FiLM modulation.
#
python -u train_anp_emulator.py --disable-film "$@"
