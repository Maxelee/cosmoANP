#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-sb35-antioverfit
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_antioverfit_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_antioverfit_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

R500_PHYSICAL_FACTOR=${R500_PHYSICAL_FACTOR:-1.0}

cd /mnt/home/mlee1/ANP_tests

# ─── Anti-overfitting + Uncertainty Calibration Training ──────────────
#
# Key changes from the baseline (anp_all_profiles_20260317_034453):
#
#  REGULARIZATION (anti-overfitting):
#    dropout:              0.1  →  0.25    (2.5x stronger)
#    weight_decay:         1e-4 →  1e-3    (10x stronger)
#    context_dropout_rate: 0.0  →  0.3     (NEW: drop 30% of context points)
#    input_noise_std:      0.0  →  0.02    (NEW: Gaussian input noise)
#    dec_layers:           5    →  4       (reduce decoder capacity)
#    d_model:              256  →  192     (leaner encoder)
#    d_latent:             128  →  96      (leaner latent)
#    dec_hidden:           512  →  384     (leaner decoder)
#
#  UNCERTAINTY CALIBRATION:
#    decoder_likelihood:   gaussian → student_t (heavier tails)
#    student_t_df:         N/A  →  4.0     (generous tail weight)
#    var_cal_weight:       0.0  →  0.05    (explicit variance calibration)
#    beta_nll_weight:      0.0  →  0.5     (prevent sigma collapse)
#    free_bits:            0.0  →  0.5     (prevent posterior collapse)
#
#  TRAINING SCHEDULE:
#    kl_warmup_epochs:     80   →  120     (slower KL ramp)
#    patience:             120  →  80      (stop earlier before overfitting)
#    epochs:               500  →  500     (keep ceiling; early stop does the work)
#
echo "[INFO] Anti-overfitting + uncertainty-calibrated training."
echo "[INFO] Student-t(df=4), dropout=0.25, wd=1e-3, ctx_dropout=0.3, beta-NLL=0.5, free_bits=0.5"

python -u train_anp_emulator.py \
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
  --mean-hidden-dim 128 \
  --mean-epochs 80 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144
