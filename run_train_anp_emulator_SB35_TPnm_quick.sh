#!/bin/bash
set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

R500_PHYSICAL_FACTOR=${R500_PHYSICAL_FACTOR:-1.0}

cd /mnt/home/mlee1/ANP_tests

echo "[INFO] Quick debug all_profiles training (reduced data/compute)."
echo "[INFO] Single-GPU / no DDP."

python -u train_anp_emulator.py \
  --profiles-base /mnt/home/mlee1/ceph/Profiles_cy \
  --param-csv /mnt/home/mlee1/50Mpc_boxes/data/param_df.csv \
  --output-dir /mnt/home/mlee1/ANP_tests/anp_training_runs \
  --suite IllustrisTNG \
  --sim-set SB35 \
  --snapnum 90 \
  --target-name all_profiles \
  --all-profiles-subset temperature pressure gas_density metallicity \
  --max-runs 28 \
  --min-halos 2 \
  --max-halos-per-run 48 \
  --radial-stride 2 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --kl-warmup-epochs 10 \
  --patience 3 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 2 \
  --r500-physical-factor "${R500_PHYSICAL_FACTOR}" \
  --d-model 192 \
  --d-latent 96 \
  --radius-fourier-n-freq 8 \
  --radius-fourier-scale 2.0 \
  --n-heads 8 \
  --n-latent-layers 2 \
  --n-ctx-layers 2 \
  --max-latent-points 512 \
  --dec-hidden 320 \
  --dec-layers 3 \
  --dropout 0.1 \
  --theta-film-scale 0.1 \
  --smoothness-weight 0.003 \
  --var-cal-weight 0.0 \
  --task-uncertainty-l2-weight 1e-4 \
  --task-uncertainty-clip 5.0 \
  --channel-balance-loss \
  --channel-balance-alpha 1.0 \
  --channel-balance-eps 1e-6 \
  --core-radius-weight 1.5 \
  --core-radius-frac 0.2 \
  --core-radius-min-bins 4 \
  --select-metric weighted_orig \
  --val-detailed-every 1 \
  --val-detailed-samples 4 \
  --context-sensitivity-every 0 \
  --selection-pressure-weight 0.2 \
  --selection-temperature-weight 0.2 \
  --selection-pressure-core-weight 0.15 \
  --selection-temperature-core-weight 0.2 \
  --save-every-epochs 0 \
  --eval-samples 8 \
  --fewshot-contexts 1 2 \
  --fewshot-repeats 1 \
  --disable-mean-prior
