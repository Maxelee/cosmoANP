#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-train
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_train_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_train_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── ANP Emulator Training ───────────────────────────────────────────
#
# All defaults in train_anp_emulator.py match the best model
# (anp_all_profiles_20260325_175639).
#
# Just run with default args.  Override --profiles-base and --param-csv
# if your data paths differ.
#
python -u train_anp_emulator.py "$@"
