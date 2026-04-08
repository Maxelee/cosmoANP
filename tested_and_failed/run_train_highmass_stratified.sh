#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-highmass
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_highmass_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_highmass_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── ANP Emulator Training: High-Mass Stratified ─────────────────────
#
# Adds mass-stratified train/val/test split and inverse-frequency
# WeightedRandomSampler with a 2× boost for log10(M500c) > 13.8.
#
# This addresses the root cause: 81% of observed clusters lie at
# |z|>3 in log10(M500c) relative to the training distribution,
# causing catastrophic calibration failure (chi2_red ~7, coverage ~0).
#
# Key flags:
#   --mass-stratified-split       stratify split to preserve mass-bin fractions
#   --use-mass-weighted-sampler   oversample high-mass halos during training
#   --high-mass-threshold 13.8    log10(M500c) threshold for extra boost
#   --high-mass-boost 2.0         multiplier applied to families above threshold
#   --mass-balance-power 1.0      inverse-frequency exponent (1=fully balanced)
#
# All other defaults reproduce the best model (anp_all_profiles_20260325_175639).
# Pass additional overrides via "$@".
#
python -u train_anp_emulator.py \
    --mass-stratified-split \
    --use-mass-weighted-sampler \
    --high-mass-threshold 13.8 \
    --high-mass-boost 2.0 \
    --mass-balance-power 1.0 \
    "$@"
