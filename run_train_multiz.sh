#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH -J anp-multiz
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_multiz_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_multiz_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── Multi-redshift ANP (z=0, 0.5, 1.0) ─────────────────────────────
#
# Extends the best single-z model (anp_all_profiles_20260325_175639)
# to three snapshots.  Key changes vs single-z baseline:
#
#   --snapnums 90 74 60          (z=0, z~0.5, z~1.0)
#   --snapshot-balanced-loss     (prevent z=0 domination; ~131/125/94 halos per run)
#   --per-snapshot-mean          (separate mean models per redshift)
#   --mean-hidden-dim 384        (larger mean model for multi-z complexity)
#   --mean-n-hidden 4
#   --mean-epochs 200
#   --mean-loss huber            (robust to outliers at high-z)
#   --time-feature-scale 1.0     (normalized redshift feature)
#   --patience 100               (more patience for multi-z convergence)
#   --time 48h                   (~3x data, longer training)
#
# z=2 (snap044) excluded: too few massive halos (mean 43/run) and
# minimal observational overlap (most data at z<1).
#
# Previous multi-z attempts failed because:
#   1. --snapnums was not passed (resolved to [90] only)
#   2. Mean model too small (128×2) for 3-redshift mean structure
#   3. No snapshot-balanced loss → z=0 dominated training
#
python -u train_anp_emulator.py \
    --snapnums 90 74 60 \
    --snapshot-redshifts 90:0.0,74:0.5,60:1.0,44:2.0 \
    --snapshot-balanced-loss \
    --per-snapshot-mean \
    --time-feature-scale 1.0 \
    --mean-hidden-dim 384 \
    --mean-n-hidden 4 \
    --mean-epochs 200 \
    --mean-loss huber \
    --mean-lr 1e-3 \
    --mean-weight-decay 1e-3 \
    --mean-batch-size 131072 \
    --mean-predict-batch-size 262144 \
    --patience 100 \
    "$@"
