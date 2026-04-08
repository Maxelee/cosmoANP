#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH -J anp-multiz-7ch
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_multiz_7ch_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_multiz_7ch_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── Multi-redshift 7-channel ANP ────────────────────────────────────
#
# Three redshifts × 7 channels:
#   Base 4: temperature, pressure, gas_density, metallicity
#   +3 hot gas: hot_temperature, hot_gas_density, hot_pressure
#
# Hot gas channels match what X-ray instruments actually measure
# (emission-weighted T > 10^5.5 K gas).  Critical for comparing to
# eROSITA, Chandra, XMM-Newton observations.
#
# Run AFTER run_train_multiz.sh succeeds to confirm multi-z works,
# then extend to 7 channels.
#
python -u train_anp_emulator.py \
    --snapnums 90 74 60 \
    --snapshot-redshifts 90:0.0,74:0.5,60:1.0,44:2.0 \
    --snapshot-balanced-loss \
    --per-snapshot-mean \
    --all-profiles-subset \
        temperature pressure gas_density metallicity \
        hot_temperature hot_gas_density hot_pressure \
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
