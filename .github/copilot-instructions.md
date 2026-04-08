# Project Guidelines

## Repository Layout
- `train_anp_emulator.py` — Main training script and model definition. Defaults reproduce the best model (`anp_all_profiles_20260325_175639`).
- `anp_emulator/` — Installable package: `api.py` (inference API), `diagnostics.py` (evaluation helpers), `__init__.py`.
- `run_train.sh` — Single SLURM submission script. Uses all defaults; pass overrides via `"$@"`.
- `notebooks/` — All Jupyter notebooks. Each has a `sys.path.insert(0, Path('..').resolve())` cell so imports work from the subdirectory.
- `tested_and_failed/` — Archived training scripts and the NF emulator from prior experiments. Reference only.
- `anp_training_runs/` — Training output directories, each named `anp_<target>_<timestamp>/`.

## Code Style
- Follow existing Python style in `train_anp_emulator.py` and `anp_emulator/api.py`: type hints on public functions, clear dataclasses, and explicit argument names.
- Prefer small, targeted edits. Do not reformat unrelated code.
- Keep numerical code vectorized with NumPy/PyTorch and preserve existing tensor shape conventions.

## Best Model & Defaults
- The reference model is `anp_all_profiles_20260325_175639` (z=0, 4-channel: temperature, pressure, gas_density, metallicity).
- All argparse defaults in `train_anp_emulator.py` are set to reproduce this model. Running `python train_anp_emulator.py` with no flags trains an equivalent run.
- Key architecture: `d_model=192`, `d_latent=96`, `dec_hidden=384`, `student_t(df=4)`, `dropout=0.25`, `context_dropout=0.3`, `beta_nll=0.5`, `free_bits=0.5`.
- Key training: `batch_size=2`, `accum_steps=16`, `epochs=500`, `patience=80`, `kl_warmup=120`, `weight_decay=1e-3`.

## Build and Test
- Use the user virtualenv:
  - `source ~/venvs/torch3/bin/activate`
- Lightweight checks preferred by default:
  - `python -c "import anp_emulator, torch, numpy; print('imports_ok')"`
  - `python -c "from anp_emulator import Emulator; print('api_ok')"`
- Full training is expensive and should not be run unless explicitly requested:
  - `sbatch run_train.sh`
- If a local debug run is explicitly requested, override defaults to reduce load: `python train_anp_emulator.py --epochs 5 --max-runs 16 --batch-size 2 --num-workers 2`.

## Conventions
- This repo models CAMELS profiles with ANP inputs shaped as: `[log_M500c, log(r/R500), theta_1..theta_35]`.
- The default target is `all_profiles` (joint 4-channel). Channel balancing is enabled by default.
- Positive-definite profile channels are modeled in log10 space and restored to physical units in inference utilities.
- Keep normalization based on the train split only; do not introduce validation/test leakage.

## Environment and Data Pitfalls
- Avoid launching heavy training on login nodes; prepare scripts there and run training on compute nodes.
- Script defaults include absolute paths (`--profiles-base`, `--param-csv`) that may not exist on every machine. Adjust before running.
- Checkpoint files in each run: `best_model.pt`, `mean_model.pt`, `epoch_*.pt`. Prefer loading by explicit path.
