#!/bin/bash
# Submit 3 SLURM jobs, one per equal-count log10(M500c) quantile — halo level.
#
# Usage:
#   bash run_train_mass_quantiles.sh              # submit all 3
#   bash run_train_mass_quantiles.sh --epochs 50  # pass extra flags to each job
#
# Quantile boundaries are computed dynamically from all individual halo masses
# (~94k halos, log10 M ∈ [11.79, 14.97]) at training time.  Approximate edges
# for 5 equal-count bins (~18.9k halos each):
#   hq0: ~[11.79, 12.38)
#   hq1: ~[12.38, 12.51)
#   hq2: ~[12.51, 12.67)
#   hq3: ~[12.67, 12.90)
#   hq4: ~[12.90, 14.97]   — use for observed clusters (log10 M > 13.5)
#
# Each job trains on ONLY the halos in its mass slice (all families kept, but
# halos outside the slice are dropped from x/y arrays before training).
# Output dirs: anp_all_profiles_hq{0..4}of5_<timestamp>/

set -euo pipefail

N_QUANTILES=5

for IDX in 0 1 2 3 4; do
    echo "Submitting halo-quantile ${IDX}/${N_QUANTILES}..."
    sbatch \
        --job-name="anp-hq${IDX}" \
        --output="/mnt/home/mlee1/ceph/logs/anp_hq${IDX}_%j.out" \
        --error="/mnt/home/mlee1/ceph/logs/anp_hq${IDX}_%j.err" \
        --nodes=1 \
        --gpus-per-task=1 \
        --ntasks=1 \
        --cpus-per-task=10 \
        --partition=gpu \
        --constraint=h100 \
        --mem=100G \
        --time=24:00:00 \
        --wrap="
set -euo pipefail
source ~/venvs/torch3/bin/activate
export OMP_NUM_THREADS=1
cd /mnt/home/mlee1/ANP_tests
python -u train_anp_emulator.py \
    --halo-mass-quantile-idx ${IDX} \
    --n-halo-mass-quantiles ${N_QUANTILES} \
    $@
"
done

echo "All 5 halo-quantile jobs submitted."
echo "Monitor with: squeue -u \$USER"
