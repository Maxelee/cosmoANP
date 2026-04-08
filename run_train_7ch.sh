#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-7ch
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_train_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_train_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# ─── 7-channel ANP: original 4 + DM_density, stellar_density, Mstar ─────
#
# z=0 (snap090 only), fiducial architecture, all other defaults.
# Mstar is a scalar per halo, broadcast to all radial bins.
#
python -u train_anp_emulator.py \
    --all-profiles-subset \
        temperature pressure gas_density metallicity \
        DM_density stellar_density Mstar \
    "$@"
