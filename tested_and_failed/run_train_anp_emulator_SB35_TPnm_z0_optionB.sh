#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=1000G
#SBATCH --time=48:00:00
#SBATCH -J anp-sb35-z0-optB
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_z0_optionB_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_z0_optionB_%j.err

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

# ─── Option B: Accuracy + Calibration Balance (z=0 only) ────────────
#
# Goal: close the gap between accuracy (RMSE 0.35 → 0.15 target) and
# calibration (z_std ~1.3 → 1.0 target) simultaneously.
#
# vs baseline (20260325_175639):
#
#  MEAN MODEL (bigger, to reduce residuals the ANP must learn):
#    mean_hidden_dim:   128 →  256
#    mean_n_hidden:     2   →  3
#    mean_epochs:       80  →  150
#    mean_loss:         mse    (keep; huber helps core but hurts overall)
#
#  ANP MODEL (slightly larger capacity, keep all regularization):
#    d_model:           192 →  224
#    d_latent:          96  →  112
#    dec_hidden:        384 →  448
#    (all reg: dropout=0.25, ctx_dropout=0.3, Student-t df=4, beta_nll=0.5,
#     free_bits=0.5, input_noise=0.02, weight_decay=1e-3 — UNCHANGED)
#
#  CALIBRATION:
#    var_cal_weight:    0.05 → 0.10  (stronger pull toward unit z_std)
#    NO stratified_var_cal (0.0)     (avoid noisy per-bin targets)
#
#  UNCHANGED from baseline:
#    training_stage:    full  (joint mean+ANP)
#    time_feature_scale: 0.1  (z=0 only; no benefit from higher)
#    snapnum:           90    (z=0 only)
#    n_heads:           8 (must divide d_model=224 → 8*28=224 ✓)
#
echo "=============================================="
echo "Option B: Accuracy + Calibration Balance (z=0)"
echo "  d_model=224, d_latent=112, dec_hidden=448"
echo "  mean: 256×3 layers, 150 epochs"
echo "  var_cal_weight=0.10, no stratified var cal"
echo "  training_stage=full, time_feature_scale=0.1"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs requested: ${GPUS_PER_NODE}"
echo "=============================================="

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
  --d-model 224 \
  --d-latent 112 \
  --radius-fourier-n-freq 16 \
  --radius-fourier-scale 2.0 \
  --n-heads 8 \
  --n-latent-layers 3 \
  --n-ctx-layers 3 \
  --max-latent-points 1024 \
  --dec-hidden 448 \
  --dec-layers 4 \
  --dropout 0.25 \
  --theta-film-scale 0.1 \
  --decoder-likelihood student_t \
  --student-t-df 4.0 \
  --smoothness-weight 0.005 \
  --var-cal-weight 0.10 \
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
  --time-feature-scale 0.1 \
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
  --mean-hidden-dim 256 \
  --mean-n-hidden 3 \
  --mean-epochs 150 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144

echo "Training completed at $(date)"
