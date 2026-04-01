#!/bin/bash
#SBATCH --job-name=nf_repro_044659
#SBATCH --output=/mnt/home/mlee1/ceph/logs/nf_repro_044659_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/nf_repro_044659_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=40
#SBATCH --mem=1000G

# Conditional Normalizing Flow equivalent of ANP run anp_all_profiles_20260320_044659.
# Same data pipeline: 7 channels (T, P, rho, Z, potential, DM_density, stellar_density),
# 4 snapshots (z=0, 0.5, 1.0, 2.0), with mean-prior residual targets.
# Replaces the ANP (latent + context/target split) with a pointwise conditional NSF.

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
echo "NF emulator (conditional NSF, 7-channel, 4-snap)"
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

echo "Training stage: ${TRAIN_STAGE}"
echo "Use mean prior: ${USE_MEAN_PRIOR}"
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
  nf_only)
    if [[ "${USE_MEAN_PRIOR}" != "1" ]]; then
      echo "ERROR: TRAIN_STAGE=nf_only requires USE_MEAN_PRIOR=1"
      exit 2
    fi
    if [[ -z "${MEAN_CKPT_PATH}" ]]; then
      echo "ERROR: TRAIN_STAGE=nf_only requires MEAN_CKPT_PATH"
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
    echo "ERROR: Unsupported TRAIN_STAGE='${TRAIN_STAGE}'. Use one of: full, mean_only, nf_only"
    exit 2
    ;;
esac

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" train_nf_emulator.py \
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
  --batch-size 4096 \
  --num-workers 8 \
  --epochs 500 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --patience 120 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 1 \
  --nf-hidden 256 256 \
  --nf-transforms 8 \
  --nf-bins 8 \
  --dropout 0.0 \
  --save-every-epochs 20 \
  --val-detailed-every 5 \
  --eval-samples 50 \
  --mean-hidden-dim 128 \
  --mean-epochs 80 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144 \
  "${STAGE_ARGS[@]}"

echo "Training completed at $(date)"
