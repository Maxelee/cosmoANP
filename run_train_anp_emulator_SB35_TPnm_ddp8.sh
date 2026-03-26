#!/bin/bash
#SBATCH --job-name=anp_sb35_allprof_ddp8
#SBATCH --output=/mnt/home/mlee1/ceph/logs/anp_sb35_allprof_ddp8_%j.out
#SBATCH --error=/mnt/home/mlee1/ceph/logs/anp_sb35_allprof_ddp8_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=40
#SBATCH --mem=1000G

set -euo pipefail

# Keep launch semantics similar to the working diffusion script.
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

echo "=============================================="
echo "ANP Full Training (DDP)"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs requested: ${GPUS_PER_NODE}"
echo "Repo: $PWD"
echo "=============================================="
nvidia-smi

# One-process-per-GPU DDP launch.
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
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --kl-warmup-epochs 80 \
  --patience 120 \
  --early-stop-min-delta 1e-4 \
  --accum-steps 16 \
  --r500-physical-factor "${R500_PHYSICAL_FACTOR}" \
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
  --mean-hidden-dim 256 \
  --mean-epochs 150 \
  --mean-lr 1e-3 \
  --mean-weight-decay 1e-3 \
  --mean-batch-size 131072 \
  --mean-log-every 10 \
  --mean-predict-batch-size 262144

echo "Training completed at $(date)"
