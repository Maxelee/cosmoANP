#!/bin/bash
#SBATCH --job-name=anp_repro_044659_t
#SBATCH --output=/mnt/home/mlee1/ceph/logs/anp_repro_044659_t_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/anp_repro_044659_t_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=40
#SBATCH --mem=1000G

# Reproduce training run anp_all_profiles_20260320_044659 with Student-t decoder
# Best multi-snapshot run (T_log10 RMSE = 1.0669 on external CV set).
# 7 channels (T, P, rho, Z, potential, DM_density, stellar_density),
# 4 snapshots (z=0, 0.5, 1.0, 2.0),
# core weights: T_core=0.35, P_core=0.25, core_radius_weight=2.0.

set -euo pipefail

unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

source /mnt/home/mlee1/venvs/torch3/bin/activate

cd /mnt/home/mlee1/ANP_tests

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
GPUS_PER_NODE=${GPUS_PER_NODE%%(*}

if [[ -z "${GPUS_PER_NODE}" ]]; then
  GPUS_PER_NODE=8
fi

echo "=============================================="
echo "ANP Reproduction: 044659 + Student-t (7-channel, 4-snap)"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs requested: ${GPUS_PER_NODE}"
echo "Repo: $PWD"
echo "=============================================="
nvidia-smi

USE_MEAN_PRIOR=${USE_MEAN_PRIOR:-0}
TRAIN_STAGE=${TRAIN_STAGE:-full}
MEAN_CKPT_PATH=${MEAN_CKPT_PATH:-/mnt/home/mlee1/ANP_tests/anp_training_runs/mean_all_profiles_20260323_115126/mean_model.pt}
MEAN_OUTPUT_PATH=${MEAN_OUTPUT_PATH:-}
DECODER_LIKELIHOOD=${DECODER_LIKELIHOOD:-student_t}
STUDENT_T_DF=${STUDENT_T_DF:-5.0}

echo "Training stage: ${TRAIN_STAGE}"
echo "Use mean prior: ${USE_MEAN_PRIOR}"
echo "Decoder likelihood: ${DECODER_LIKELIHOOD}"
echo "Student-t dof: ${STUDENT_T_DF}"
if [[ "${USE_MEAN_PRIOR}" == "1" && -n "${MEAN_CKPT_PATH}" ]]; then
  echo "Mean checkpoint path: ${MEAN_CKPT_PATH}"
fi

STAGE_ARGS=(--training-stage "${TRAIN_STAGE}")
if [[ "${USE_MEAN_PRIOR}" != "1" ]]; then
  STAGE_ARGS+=(--disable-mean-prior)
fi

case "${TRAIN_STAGE}" in
  mean_only)
    if [[ "${USE_MEAN_PRIOR}" != "1" ]]; then
      echo "ERROR: TRAIN_STAGE=mean_only requires USE_MEAN_PRIOR=1"
      exit 2
    fi
    if [[ -n "${MEAN_OUTPUT_PATH}" ]]; then
      STAGE_ARGS+=(--mean-output-path "${MEAN_OUTPUT_PATH}")
    fi
    ;;
  anp_only)
    if [[ "${USE_MEAN_PRIOR}" != "1" ]]; then
      echo "ERROR: TRAIN_STAGE=anp_only requires USE_MEAN_PRIOR=1"
      exit 2
    fi
    if [[ -z "${MEAN_CKPT_PATH}" ]]; then
      echo "ERROR: TRAIN_STAGE=anp_only requires MEAN_CKPT_PATH"
      exit 2
    fi
    STAGE_ARGS+=(--mean-checkpoint-path "${MEAN_CKPT_PATH}")
    ;;
  full)
    if [[ -n "${MEAN_OUTPUT_PATH}" ]]; then
      STAGE_ARGS+=(--mean-output-path "${MEAN_OUTPUT_PATH}")
    fi
    ;;
  *)
    echo "ERROR: Unsupported TRAIN_STAGE='${TRAIN_STAGE}'. Use one of: full, mean_only, anp_only"
    exit 2
    ;;
esac

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" train_anp_emulator.py \
  --enable-ddp \
  --ddp-timeout-sec 7200 \
  --ddp-num-workers 8 \
  --profiles-base /mnt/home/mlee1/ceph/Profiles_cy \
  --param-csv /mnt/home/mlee1/50Mpc_boxes/data/param_df.csv \
  --output-dir /mnt/home/mlee1/ANP_tests/anp_training_runs \
  --suite IllustrisTNG \
  --sim-set SB35 \
  --snapnums 90 74 60 44 \
  --snapshot-redshifts 90:0.0,74:0.5,60:1.0,44:2.0 \
  --min-snapshots-per-run 2 \
  --target-name all_profiles \
  --all-profiles-subset temperature pressure gas_density metallicity potential DM_density stellar_density \
  --max-runs 1024 \
  --min-halos 2 \
  --max-halos-per-run 150 \
  --radial-stride 1 \
  --r500-physical-factor 1.0 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --batch-size 2 \
  --num-workers 8 \
  --epochs 500 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --kl-warmup-epochs 80 \
  --patience 120 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 16 \
  --d-model 256 \
  --d-latent 128 \
  --radius-fourier-n-freq 16 \
  --radius-fourier-scale 2.0 \
  --n-heads 8 \
  --n-latent-layers 3 \
  --n-ctx-layers 3 \
  --max-latent-points 1024 \
  --dec-hidden 512 \
  --dec-layers 5 \
  --dropout 0.1 \
  --theta-film-scale 0.1 \
  --decoder-likelihood "${DECODER_LIKELIHOOD}" \
  --student-t-df "${STUDENT_T_DF}" \
  --smoothness-weight 0.005 \
  --var-cal-weight 0.0 \
  --task-uncertainty-l2-weight 5e-4 \
  --task-uncertainty-clip 5.0 \
  --channel-balance-loss \
  --channel-balance-alpha 1.0 \
  --channel-balance-eps 1e-6 \
  --core-radius-weight 2.0 \
  --core-radius-frac 0.2 \
  --core-radius-min-bins 6 \
  --max-aux-snapshots 2 \
  --aux-halo-frac 0.5 \
  --time-feature-scale 0.5 \
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
  --eval-samples 20 \
  --fewshot-contexts 1 2 5 10 \
  --mean-hidden-dim 128 \
  --mean-epochs 80 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144 \
  "${STAGE_ARGS[@]}"

echo "Training completed at $(date)"
