#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -J anp-validate
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_validate_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_validate_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── ANP Emulator Validation Suite ───────────────────────────────────
#
# Runs the 10-category validation suite on the best model
# (anp_all_profiles_20260325_175639).
#
# Results go to validation_results/ with plots and a JSON summary.
#
# Pass overrides or specific categories via "$@", e.g.:
#   sbatch run_validation.sh --categories 1 2 3
#   sbatch run_validation.sh --n-samples 50
#
python -u run_validation_suite.py --device cuda "$@"
