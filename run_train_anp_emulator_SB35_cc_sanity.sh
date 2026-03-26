#!/bin/bash
#SBATCH --job-name=anp_sb35_cc_sanity
#SBATCH --output=/mnt/home/mlee1/ceph/logs/anp_sb35_cc_sanity_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/anp_sb35_cc_sanity_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# ---------------------------------------------------------------
# Minimal sanity-check: single snapshot, temperature only,
# CC dual-head on, no ideal-gas, no task-uncertainty.
# Purpose: verify that dual-head routing actually improves core
# errors vs a single-head baseline, before committing to a
# full 48h multi-GPU job.
# ---------------------------------------------------------------

set -euo pipefail

unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

source /mnt/home/mlee1/venvs/torch3/bin/activate

cd /mnt/home/mlee1/ANP_tests

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

R500_PHYSICAL_FACTOR=${R500_PHYSICAL_FACTOR:-1.0}

echo "=============================================="
echo "ANP CC Sanity Check — 1 GPU, temperature only"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Repo: $PWD"
echo "=============================================="
nvidia-smi

python train_anp_emulator.py \
  --profiles-base /mnt/home/mlee1/ceph/Profiles_cy \
  --param-csv /mnt/home/mlee1/50Mpc_boxes/data/param_df.csv \
  --output-dir /mnt/home/mlee1/ANP_tests/anp_training_runs \
  --suite IllustrisTNG \
  --sim-set SB35 \
  --snapnums 90 \
  --snapshot-redshifts 90:0.0 \
  --min-snapshots-per-run 1 \
  --target-name all_profiles \
  --all-profiles-subset temperature \
  --max-runs 512 \
  --min-halos 2 \
  --max-halos-per-run 128 \
  --radial-stride 1 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --batch-size 4 \
  --num-workers 4 \
  --epochs 120 \
  --lr 1.5e-4 \
  --weight-decay 3e-4 \
  --grad-clip 1.0 \
  --kl-warmup-epochs 30 \
  --patience 30 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 8 \
  --r500-physical-factor "${R500_PHYSICAL_FACTOR}" \
  --d-model 128 \
  --d-latent 64 \
  --radius-fourier-n-freq 16 \
  --radius-fourier-scale 2.0 \
  --n-heads 4 \
  --n-latent-layers 2 \
  --n-ctx-layers 2 \
  --max-latent-points 512 \
  --dec-hidden 256 \
  --dec-layers 3 \
  --dropout 0.15 \
  --theta-film-scale 0.1 \
  --smoothness-weight 0.005 \
  --var-cal-weight 0.0 \
  --core-radius-weight 2.0 \
  --core-radius-frac 0.2 \
  --core-radius-min-bins 6 \
  --select-metric weighted_orig \
  --val-detailed-every 5 \
  --val-detailed-samples 10 \
  --context-sensitivity-every 5 \
  --context-sensitivity-batches 2 \
  --context-sensitivity-samples 4 \
  --selection-temperature-weight 0.5 \
  --selection-temperature-core-weight 0.5 \
  --save-every-epochs 20 \
  --eval-samples 10 \
  --fewshot-contexts 1 2 5 10 \
  --mean-hidden-dim 128 \
  --mean-epochs 80 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144 \
  --cc-indicator \
  --cc-dual-head

echo "CC sanity check completed at $(date)"
echo ""
echo "Next: re-run with --cc-dual-head removed (keep everything else the same)"
echo "and compare temperature_core_rmse between the two runs."
