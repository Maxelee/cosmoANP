#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-no-anp
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_train_no_anp_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_train_no_anp_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── Ablation: FiLM-only MLP (no ANP cross-attention or latent) ──────
#
# All defaults match the best model (anp_all_profiles_20260325_175639)
# except the latent encoder and deterministic cross-attention paths are
# removed.  The decoder trunk receives only tgt_x (Fourier-embedded
# radius + mass + 35 theta params) and FiLM conditioning from theta.
# This tests whether the simpler FiLM MLP matches the full ANP.
#
python -u train_anp_emulator.py --disable-anp "$@"
