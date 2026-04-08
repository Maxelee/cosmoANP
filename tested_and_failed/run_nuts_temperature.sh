#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J anp-nuts-temp
#SBATCH -o /mnt/home/mlee1/ceph/logs/anp_nuts_temp_%j.out
#SBATCH -e /mnt/home/mlee1/ceph/logs/anp_nuts_temp_%j.err

set -euo pipefail

source ~/venvs/torch3/bin/activate

export OMP_NUM_THREADS=1

cd /mnt/home/mlee1/ANP_tests

# Temperature-only Pyro NUTS on observed kT profiles.
# Default behavior runs all three setups: 4, 6, and 35 free parameters,
# and saves full per-chain samples for each run.
#
# Example overrides:
#   sbatch run_nuts_temperature.sh --free-set 4 --n-halos 16 --warmup 1000 --samples 4000
#   sbatch run_nuts_temperature.sh --device cuda --target-accept 0.9

python -u run_nuts_temperature.py "$@"
