from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from train_anp_emulator import MeanModel, StrongANP, add_mean_back, denorm_y, restore_all_profiles_physical_units

FieldArg = Union[str, Sequence[str]]
RadialBinsArg = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


@dataclass
class PredictionResult:
    mean: np.ndarray
    total_std: np.ndarray
    aleatoric_std: np.ndarray
    epistemic_std: np.ndarray
    field_names: List[str]


class Emulator:
    def __init__(
        self,
        model: StrongANP,
        mean_model: MeanModel | None,
        checkpoint_path: Path,
        target_names: List[str],
        args: Dict[str, Any],
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        device: torch.device,
    ):
        self.model = model
        self.mean_model = mean_model
        self.checkpoint_path = checkpoint_path
        self.target_names = target_names
        self.args = args
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device

        self.theta_dim = int(args.get("theta_dim", 35))
        self.theta_start_idx = int(args.get("theta_start_idx", 2))
        self.raw_x_dim = int(x_mean.shape[0])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str = "cpu") -> "Emulator":
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = dict(state.get("args", {}))
        norm = state.get("norm", None)
        if norm is None:
            raise KeyError("Checkpoint missing norm statistics")

        x_mean_np = np.asarray(norm["x_mean"], dtype=np.float32)
        x_std_np = np.asarray(norm["x_std"], dtype=np.float32)
        y_mean_np = np.asarray(norm["y_mean"], dtype=np.float32)
        y_std_np = np.asarray(norm["y_std"], dtype=np.float32)

        target_names = list(state.get("target_names", []))
        if not target_names:
            tname = state.get("target_name", None)
            if tname is None:
                raise KeyError("Checkpoint missing target_names/target_name")
            target_names = [str(tname)]

        raw_x_dim = int(x_mean_np.shape[0])
        y_dim = int(y_mean_np.shape[0])
        radius_fourier_n_freq = 0 if args.get("disable_radius_fourier", False) else int(args.get("radius_fourier_n_freq", 16))

        model_state = state["model_state_dict"]

        # Make loading robust across training configs: infer whether per-task
        # uncertainty parameters were part of this checkpoint.
        has_task_uncertainty_param = any(str(k).endswith("log_sigma_task") for k in model_state.keys())
        disable_task_uncertainty = bool(args.get("disable_task_uncertainty_weighting", False))
        target_name = str(state.get("target_name", args.get("target_name", "")))
        use_task_uncertainty_weighting = has_task_uncertainty_param or (
            (target_name == "all_profiles") and (not disable_task_uncertainty)
        )

        def _infer_model_x_dim(state_dict: Dict[str, Any], y_dim_val: int) -> Optional[int]:
            # Works for both bare and DataParallel-prefixed checkpoints.
            for key in (
                "latent.point.net.0.weight",
                "module.latent.point.net.0.weight",
                # Legacy fallback in case older checkpoints used a different name.
                "latent.embed.net.0.weight",
                "module.latent.embed.net.0.weight",
            ):
                w = state_dict.get(key, None)
                if torch.is_tensor(w) and w.ndim == 2:
                    in_features = int(w.shape[1])
                    x_dim_val = in_features - y_dim_val
                    if x_dim_val > 0:
                        return x_dim_val
            return None

        inferred_model_x_dim = _infer_model_x_dim(model_state, y_dim)

        candidate_include_raw: List[bool] = []
        if radius_fourier_n_freq <= 0:
            candidate_include_raw = [True]
        else:
            if inferred_model_x_dim is not None:
                expected_new = raw_x_dim + 2 * radius_fourier_n_freq
                expected_legacy = raw_x_dim + 2 * radius_fourier_n_freq - 1
                if inferred_model_x_dim == expected_new:
                    candidate_include_raw.append(True)
                elif inferred_model_x_dim == expected_legacy:
                    candidate_include_raw.append(False)
            # Keep deterministic fallback order for backwards compatibility.
            candidate_include_raw.extend([True, False])

        # Deduplicate while preserving order.
        seen = set()
        candidate_include_raw = [v for v in candidate_include_raw if not (v in seen or seen.add(v))]

        model = None
        last_err: Optional[Exception] = None
        for include_raw in candidate_include_raw:
            radius_fourier_extra_dim = 0
            if radius_fourier_n_freq > 0:
                radius_fourier_extra_dim = 2 * radius_fourier_n_freq
                if not include_raw:
                    radius_fourier_extra_dim -= 1
            model_x_dim = raw_x_dim + radius_fourier_extra_dim

            try:
                candidate_model = StrongANP(
                    x_dim=model_x_dim,
                    raw_x_dim=raw_x_dim,
                    y_dim=y_dim,
                    d_model=int(args.get("d_model", 256)),
                    d_latent=int(args.get("d_latent", 128)),
                    radial_feature_idx=1,
                    radius_fourier_n_freq=radius_fourier_n_freq,
                    radius_fourier_scale=float(args.get("radius_fourier_scale", 1.0)),
                    radius_fourier_include_raw_radius=include_raw,
                    n_heads=int(args.get("n_heads", 8)),
                    n_latent_layers=int(args.get("n_latent_layers", 2)),
                    n_ctx_layers=int(args.get("n_ctx_layers", 2)),
                    max_latent_points=int(args.get("max_latent_points", 4096)),
                    theta_start_idx=int(args.get("theta_start_idx", 2)),
                    dec_hidden=int(args.get("dec_hidden", 512)),
                    dec_layers=int(args.get("dec_layers", 4)),
                    dropout=float(args.get("dropout", 0.1)),
                    theta_film_scale=float(args.get("theta_film_scale", 0.1)),
                    smoothness_weight=float(args.get("smoothness_weight", 0.0)),
                    var_cal_weight=float(args.get("var_cal_weight", 0.0)),
                    use_task_uncertainty_weighting=use_task_uncertainty_weighting,
                    task_uncertainty_l2_weight=float(args.get("task_uncertainty_l2_weight", 0.0)),
                    task_uncertainty_clip=float(args.get("task_uncertainty_clip", 0.0)),
                    channel_balance_loss=bool(args.get("channel_balance_loss", False)),
                    channel_balance_alpha=float(args.get("channel_balance_alpha", 1.0)),
                    channel_balance_eps=float(args.get("channel_balance_eps", 1e-6)),
                )
                candidate_model.load_state_dict(model_state)
                candidate_model.eval()
                model = candidate_model
                args["radius_fourier_include_raw_radius"] = bool(include_raw)
                break
            except Exception as exc:  # noqa: BLE001 - aggregate candidate failures
                last_err = exc

        if model is None:
            raise RuntimeError(
                "Failed to load checkpoint with both Fourier-radius layouts "
                "(concat raw radius and legacy replace-radius)."
            ) from last_err

        mean_model = None
        if bool(state.get("mean_prior_enabled", False)) and state.get("mean_model_state_dict", None) is not None:
            mean_model = MeanModel(
                y_dim=int(y_mean_np.shape[0]),
                hidden_dim=int(args.get("mean_hidden_dim", 128)),
            )
            mean_model.load_state_dict(state["mean_model_state_dict"])
            mean_model.eval()

        dev = torch.device(device)
        model = model.to(dev)
        if mean_model is not None:
            mean_model = mean_model.to(dev)

        return cls(
            model=model,
            mean_model=mean_model,
            checkpoint_path=ckpt_path,
            target_names=target_names,
            args=args,
            x_mean=torch.tensor(x_mean_np, dtype=torch.float32, device=dev),
            x_std=torch.tensor(x_std_np, dtype=torch.float32, device=dev),
            y_mean=torch.tensor(y_mean_np, dtype=torch.float32, device=dev),
            y_std=torch.tensor(y_std_np, dtype=torch.float32, device=dev),
            device=dev,
        )

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: str = "cpu", checkpoint_name: str = "best_model.pt") -> "Emulator":
        run_dir_path = Path(run_dir)
        requested_ckpt = run_dir_path / checkpoint_name
        if requested_ckpt.exists():
            return cls.from_checkpoint(requested_ckpt, device=device)

        latest_epoch_ckpt: Optional[Path] = None
        latest_epoch_num = -1
        for ckpt_path in run_dir_path.glob("epoch_*.pt"):
            stem = ckpt_path.stem
            _, _, epoch_str = stem.partition("epoch_")
            if not epoch_str.isdigit():
                continue
            epoch_num = int(epoch_str)
            if epoch_num > latest_epoch_num:
                latest_epoch_num = epoch_num
                latest_epoch_ckpt = ckpt_path

        if latest_epoch_ckpt is not None:
            return cls.from_checkpoint(latest_epoch_ckpt, device=device)

        raise FileNotFoundError(
            f"No checkpoint found in run directory {run_dir_path}. "
            f"Tried requested checkpoint '{checkpoint_name}' and fallback pattern 'epoch_*.pt'."
        )

    def available_fields(self) -> List[str]:
        return list(self.target_names)

    def predict(
        self,
        theta: np.ndarray,
        M: np.ndarray,
        r_bins: RadialBinsArg,
        field: FieldArg,
        n_samples: int = 30,
    ) -> PredictionResult:
        batch, n_halo, n_r = self._build_zeroshot_batch(theta=theta, masses=M, r_bins=r_bins)

        with torch.no_grad():
            mu, total_std, aleatoric_std, epistemic_std = self.model.predict(batch, device=self.device, n_samples=n_samples)

            n_pts = n_halo * n_r
            mu = mu[:, :n_pts, :]
            total_std = total_std[:, :n_pts, :]
            aleatoric_std = aleatoric_std[:, :n_pts, :]
            epistemic_std = epistemic_std[:, :n_pts, :]

            mu_o = denorm_y(mu, self.y_mean, self.y_std)
            mu_o = add_mean_back(mu_o, batch["tgt_x"].to(self.device), self.x_mean, self.x_std, self.mean_model)
            mu_o_log = mu_o

            total_std_o = (total_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)
            aleatoric_std_o = (aleatoric_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)
            epistemic_std_o = (epistemic_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)

            if len(self.target_names) > 1:
                _, mu_o, total_std_o = restore_all_profiles_physical_units(
                    mu_o_log,
                    mu_o_log,
                    total_std_o,
                    self.target_names,
                )
                _, _, aleatoric_std_o = restore_all_profiles_physical_units(
                    mu_o_log,
                    mu_o_log,
                    aleatoric_std_o,
                    self.target_names,
                )
                _, _, epistemic_std_o = restore_all_profiles_physical_units(
                    mu_o_log,
                    mu_o_log,
                    epistemic_std_o,
                    self.target_names,
                )

        mu_np = mu_o[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()
        total_np = total_std_o[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()
        ale_np = aleatoric_std_o[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()
        epi_np = epistemic_std_o[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()

        idx, names, single = self._resolve_field_selection(field)

        mu_np = mu_np[..., idx]
        total_np = total_np[..., idx]
        ale_np = ale_np[..., idx]
        epi_np = epi_np[..., idx]

        if single:
            mu_np = mu_np[..., 0]
            total_np = total_np[..., 0]
            ale_np = ale_np[..., 0]
            epi_np = epi_np[..., 0]

        return PredictionResult(
            mean=mu_np,
            total_std=total_np,
            aleatoric_std=ale_np,
            epistemic_std=epi_np,
            field_names=names,
        )

    def _resolve_field_selection(self, field: FieldArg) -> Tuple[List[int], List[str], bool]:
        if isinstance(field, str):
            requested = [field]
            single = True
        else:
            requested = [str(x) for x in field]
            single = False

        if len(requested) == 0:
            raise ValueError("field must be a non-empty string or list of strings")

        out_idx = []
        for name in requested:
            if name not in self.target_names:
                raise ValueError(
                    f"Unknown field '{name}'. Available fields: {self.target_names}"
                )
            out_idx.append(self.target_names.index(name))

        return out_idx, requested, single

    def _build_zeroshot_batch(self, theta: np.ndarray, masses: np.ndarray, r_bins: RadialBinsArg):
        masses = np.asarray(masses, dtype=np.float32)
        theta = np.asarray(theta, dtype=np.float32)

        if masses.ndim == 0:
            masses = masses[None]

        if masses.ndim != 1:
            raise ValueError(f"M must be 1D (or scalar). Got shape {masses.shape}")
        if np.any(masses <= 0):
            raise ValueError("All halo masses in M must be > 0")

        n_halo = int(masses.shape[0])
        r_bins_arr = np.asarray(r_bins, dtype=np.float32)
        if r_bins_arr.ndim == 0:
            r_bins_arr = r_bins_arr[None]

        if r_bins_arr.ndim == 1:
            r_bins_use = np.repeat(r_bins_arr[None, :], n_halo, axis=0)
        elif r_bins_arr.ndim == 2:
            if r_bins_arr.shape[0] != n_halo:
                raise ValueError(
                    f"2D r_bins must have first dimension len(M)={n_halo}, got {r_bins_arr.shape[0]}"
                )
            r_bins_use = r_bins_arr
        else:
            raise ValueError(
                "r_bins must be scalar, 1D, or 2D with shape (len(M), n_r). "
                f"Got shape {r_bins_arr.shape}"
            )

        if np.any(r_bins_use <= 0):
            raise ValueError("All radius bins in r_bins must be > 0")

        n_r = int(r_bins_use.shape[1])

        if theta.ndim == 1:
            if theta.shape[0] != self.theta_dim:
                raise ValueError(f"theta must have length {self.theta_dim}, got {theta.shape[0]}")
            theta_use = np.repeat(theta[None, :], n_halo, axis=0)
        elif theta.ndim == 2:
            if theta.shape[1] != self.theta_dim:
                raise ValueError(f"theta second dimension must be {self.theta_dim}, got {theta.shape[1]}")
            if theta.shape[0] not in (1, n_halo):
                raise ValueError(f"theta first dimension must be 1 or len(M)={n_halo}, got {theta.shape[0]}")
            theta_use = np.repeat(theta, n_halo, axis=0) if theta.shape[0] == 1 else theta
        else:
            raise ValueError(f"theta must be shape ({self.theta_dim},) or (n_halo, {self.theta_dim}). Got {theta.shape}")

        log_m = np.log10(np.clip(masses, 1e10, None))
        log_r = np.log10(np.clip(r_bins_use, 1e-4, None))

        x_raw = np.zeros((n_halo, n_r, self.raw_x_dim), dtype=np.float32)
        x_raw[..., 0] = log_m[:, None]
        x_raw[..., 1] = log_r
        x_raw[..., self.theta_start_idx : self.theta_start_idx + self.theta_dim] = theta_use[:, None, :]

        x_norm = (x_raw - self.x_mean.detach().cpu().numpy()[None, None, :]) / self.x_std.detach().cpu().numpy()[None, None, :]
        x_norm = x_norm.reshape(1, n_halo * n_r, self.raw_x_dim)

        tgt_x = torch.tensor(x_norm, dtype=torch.float32)
        tgt_mask = torch.ones((1, n_halo * n_r), dtype=torch.bool)

        # Use a neutral zero-shot context token so we do not leak any target x
        # back into the context path. Keep it unmasked to avoid all-masked
        # attention edge cases in the current model implementation.
        ctx_x = torch.zeros((1, 1, self.raw_x_dim), dtype=torch.float32)
        ctx_mask = torch.ones((1, 1), dtype=torch.bool)

        y_dim = len(self.target_names)
        batch = {
            "ctx_x": ctx_x,
            "ctx_y": torch.zeros((1, 1, y_dim), dtype=torch.float32),
            "tgt_x": tgt_x,
            "tgt_y": torch.zeros((1, n_halo * n_r, y_dim), dtype=torch.float32),
            "ctx_mask": ctx_mask,
            "tgt_mask": tgt_mask,
            "meta": [{"run_id": -1, "n_halo": n_halo, "n_r": n_r, "n_c": 1}],
        }
        return batch, n_halo, n_r
