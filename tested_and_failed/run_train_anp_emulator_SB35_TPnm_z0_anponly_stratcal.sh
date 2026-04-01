#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH -J anp-sb35-z0-anp-strat
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_z0_anponly_stratcal_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_z0_anponly_stratcal_%j.err

set -euo pipefail

unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

source /mnt/home/mlee1/venvs/torch3/bin/activate

cd /mnt/home/mlee1/ANP_tests

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

R500_PHYSICAL_FACTOR=${R500_PHYSICAL_FACTOR:-1.0}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
GPUS_PER_NODE=${GPUS_PER_NODE%%(*}

if [[ -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE=8
fi

# Baseline anchored run:
# - Reference run: anp_all_profiles_20260325_175639
# - Keep mean model fixed from baseline
# - Train ANP only at z=0 with stratified variance calibration enabled
BASELINE_RUN="anp_training_runs/anp_all_profiles_20260325_175639"
MEAN_CKPT="${BASELINE_RUN}/mean_model.pt"

STRAT_VAR_CAL_WEIGHT=${STRAT_VAR_CAL_WEIGHT:-0.10}
STRAT_MASS_BINS=${STRAT_MASS_BINS:-4}
STRAT_RADIUS_BINS=${STRAT_RADIUS_BINS:-4}
STRAT_MIN_POINTS=${STRAT_MIN_POINTS:-32}

echo "=============================================="
echo "ANP z=0 ANP-only Training (reuse baseline mean model)"
echo "Baseline reference: ${BASELINE_RUN}"
echo "Mean checkpoint: ${MEAN_CKPT}"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs requested: ${GPUS_PER_NODE}"
echo "stratified-var-cal-weight=${STRAT_VAR_CAL_WEIGHT}"
echo "stratified-mass-bins=${STRAT_MASS_BINS}"
echo "stratified-radius-bins=${STRAT_RADIUS_BINS}"
echo "stratified-min-points=${STRAT_MIN_POINTS}"
echo "=============================================="

if [[ ! -f "${MEAN_CKPT}" ]]; then
  echo "ERROR: mean checkpoint not found: ${MEAN_CKPT}" >&2
  exit 2
fi

nvidia-smi

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" train_anp_emulator.py \
  --enable-ddp \
  --ddp-timeout-sec 7200 \
  --ddp-num-workers 4 \
  --profiles-base /mnt/home/mlee1/ceph/Profiles_cy \
  --param-csv /mnt/home/mlee1/50Mpc_boxes/data/param_df.csv \
  --output-dir ./anp_training_runs \
  --suite IllustrisTNG \
  --sim-set SB35 \
  --snapnum 90 \
  --target-name all_profiles \
  --all-profiles-subset temperature pressure gas_density metallicity \
  --max-runs 1024 \
  --min-halos 2 \
  --max-halos-per-run 128 \
  --radial-stride 1 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --batch-size 2 \
  --num-workers 8 \
  --epochs 500 \
  --lr 3e-4 \
  --weight-decay 1e-3 \
  --grad-clip 1.0 \
  --kl-warmup-epochs 120 \
  --patience 80 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 16 \
  --r500-physical-factor "${R500_PHYSICAL_FACTOR}" \
  --d-model 192 \
  --d-latent 96 \
  --radius-fourier-n-freq 16 \
  --radius-fourier-scale 2.0 \
  --n-heads 8 \
  --n-latent-layers 3 \
  --n-ctx-layers 3 \
  --max-latent-points 1024 \
  --dec-hidden 384 \
  --dec-layers 4 \
  --dropout 0.25 \
  --theta-film-scale 0.1 \
  --decoder-likelihood student_t \
  --student-t-df 4.0 \
  --smoothness-weight 0.005 \
  --var-cal-weight 0.05 \
  --stratified-var-cal-weight "${STRAT_VAR_CAL_WEIGHT}" \
  --stratified-mass-bins "${STRAT_MASS_BINS}" \
  --stratified-radius-bins "${STRAT_RADIUS_BINS}" \
  --stratified-min-points "${STRAT_MIN_POINTS}" \
  --context-dropout-rate 0.3 \
  --input-noise-std 0.02 \
  --beta-nll-weight 0.5 \
  --free-bits 0.5 \
  --task-uncertainty-l2-weight 5e-4 \
  --task-uncertainty-clip 5.0 \
  --channel-balance-loss \
  --channel-balance-alpha 1.0 \
  --channel-balance-eps 1e-6 \
  --core-radius-weight 2.0 \
  --core-radius-frac 0.2 \
  --core-radius-min-bins 6 \
  --time-feature-scale 1.0 \
  --select-metric weighted_orig \
  --val-detailed-every 5 \
  --val-detailed-samples 10 \
  --context-sensitivity-every 5 \
  --context-sensitivity-batches 2 \
  --context-sensitivity-samples 4 \
  --selection-pressure-weight 0.2 \
  --selection-temperature-weight 0.2 \
  --selection-pressure-core-weight 0.25 \
  --selection-temperature-core-weight 0.35 \
  --save-every-epochs 20 \
  --eval-samples 30 \
  --fewshot-contexts 1 2 5 10 \
  --training-stage anp_only \
  --mean-checkpoint-path "${MEAN_CKPT}" \
  --mean-hidden-dim 128 \
  --mean-n-hidden 2 \
  --mean-epochs 80 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144

echo "Training completed at $(date)"
