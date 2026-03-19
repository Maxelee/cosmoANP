# Project Guidelines

## Code Style
- Follow existing Python style in `train_anp_emulator.py` and `anp_emulator/api.py`: type hints on public functions, clear dataclasses, and explicit argument names.
- Prefer small, targeted edits. Do not reformat unrelated code.
- Keep numerical code vectorized with NumPy/PyTorch and preserve existing tensor shape conventions.

## Architecture
- Main training and model definition live in `train_anp_emulator.py`.
- Inference-facing API lives in `anp_emulator/api.py` (`Emulator.from_checkpoint`, `Emulator.predict`).
- Evaluation helpers live in `anp_emulator/diagnostics.py`.
- Training outputs are versioned under `anp_training_runs/anp_<target>_<timestamp>/` with args and metrics snapshots.

## Build and Test
- Use the user virtualenv:
  - `source ~/venvs/torch3/bin/activate`
- Lightweight checks preferred by default:
  - `python -c "import anp_emulator, torch, numpy; print('imports_ok')"`
  - `python -c "from anp_emulator import Emulator; print('api_ok')"`
- Full training is expensive and should not be run unless explicitly requested:
  - `sbatch run_train_anp_emulator_SB35_TPnm.sh`
- If a local debug run is explicitly requested, start from the script flags but reduce load (`--epochs`, `--max-runs`, `--batch-size`, `--num-workers`).

## Conventions
- This repo models CAMELS profiles with ANP inputs shaped as: `[log_M500c, log(r/R500), theta_1..theta_35]`.
- `all_profiles` mode mixes channels with very different scales. Keep channel balancing enabled when training this mode.
- Positive-definite profile channels are modeled in log10 space and restored to physical units in inference utilities.
- Keep normalization based on the train split only; do not introduce validation/test leakage.

## Environment and Data Pitfalls
- Avoid launching heavy training on login nodes; prepare scripts there and run training on compute nodes.
- Script defaults include absolute paths that may not exist on every machine (for example `--profiles-base` and `--param-csv`). Adjust paths before running.
- Checkpoint naming can vary by run (`checkpoint_best.pt`, `best_model.pt`, or `epoch_*.pt`). Prefer loading by explicit checkpoint path when possible.
