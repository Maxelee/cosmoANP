from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from train_anp_emulator import CCPredictor, MeanModel, PerSnapshotMeanModel, StrongANP, add_mean_back, denorm_y, restore_all_profiles_physical_units, _lookup_bin_stats

FieldArg = Union[str, Sequence[str]]
RadialBinsArg = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


@dataclass
class PredictionResult:
    mean: np.ndarray
    total_std: np.ndarray
    aleatoric_std: np.ndarray
    epistemic_std: np.ndarray
    field_names: List[str]
    # Log-space outputs for channels modelled in log10 space.
    # These are populated when the emulator uses log channels and give
    # a symmetric, better-calibrated uncertainty representation.
    mean_log10: Optional[np.ndarray] = None
    std_log10: Optional[np.ndarray] = None


class Emulator:
    def __init__(
        self,
        model: StrongANP,
        mean_model: MeanModel | PerSnapshotMeanModel | None,
        checkpoint_path: Path,
        target_names: List[str],
        args: Dict[str, Any],
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        device: torch.device,
        cc_prior: Optional[Dict[str, Any]] = None,
        cc_predictor: Optional[CCPredictor] = None,
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
        self.norm_stats: Optional[Dict[str, Any]] = None  # set by from_checkpoint if available

        self.theta_dim = int(args.get("theta_dim", 35))
        self.theta_start_idx = int(args.get("theta_start_idx", 2))
        self.raw_x_dim = int(x_mean.shape[0])
        self.use_continuous_redshift_feature = bool(args.get("use_continuous_redshift_feature", False))
        self.redshift_feature_idx = int(args.get("redshift_feature_idx", self.theta_start_idx + self.theta_dim))

        # Parse snapshot->redshift metadata from checkpoint args when available.
        redshift_by_snap_cfg = args.get("redshift_by_snap", {}) or {}
        self.redshift_by_snap: Dict[int, float] = {}
        for k, v in redshift_by_snap_cfg.items():
            try:
                self.redshift_by_snap[int(k)] = float(v)
            except (TypeError, ValueError):
                continue

        snapnums_cfg = args.get("resolved_snapnums", None)
        if snapnums_cfg is None:
            snapnum_legacy = int(args.get("snapnum", 90))
            self.snapnums = [snapnum_legacy]
        else:
            self.snapnums = [int(s) for s in snapnums_cfg]
        self.snap_to_idx = {int(s): i for i, s in enumerate(self.snapnums)}
        self.default_snapnum = int(self.snapnums[0])

        # Continuous-z feature is usable only if the model input layout contains it.
        expected_min_dim = self.theta_start_idx + self.theta_dim + 1
        if self.redshift_feature_idx < 0 or self.raw_x_dim < expected_min_dim:
            self.use_continuous_redshift_feature = False

        # Cool-core indicator feature.
        self.use_cc_indicator = bool(args.get("cc_indicator", False))
        self.cc_indicator_feature_idx = int(args.get("cc_indicator_feature_idx", -1))
        self.cc_indicator_mode = str(args.get("cc_indicator_mode", "continuous")).lower()
        if self.cc_indicator_mode not in {"continuous", "binary"}:
            self.cc_indicator_mode = "continuous"
        self.cc_prior = cc_prior  # may be None for older checkpoints
        self.cc_predictor = cc_predictor  # parameter-aware CC predictor (may be None)

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
                    num_snapshots=max(1, len(args.get("resolved_snapnums", [args.get("snapnum", 90)]))),
                    time_feature_scale=float(args.get("time_feature_scale", 0.1)),
                    ideal_gas_weight=float(args.get("ideal_gas_weight", 0.0)),
                    ideal_gas_channel_indices=args.get("ideal_gas_channel_indices", None),
                    cc_dual_head=bool(args.get("cc_dual_head", False)),
                    cc_indicator_feature_idx=int(args.get("cc_indicator_feature_idx", -1)),
                    cc_indicator_mode=str(args.get("cc_indicator_mode", "continuous")),
                    decoder_likelihood=str(args.get("decoder_likelihood", "gaussian")),
                    student_t_df=float(args.get("student_t_df", 5.0)),
                )
                # Backward compat: inject default y_std_buf for checkpoints
                # saved before the ideal-gas penalty was introduced.
                if "y_std_buf" not in model_state:
                    model_state["y_std_buf"] = candidate_model.y_std_buf.clone()
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
            # ---- Per-snapshot mean models ----
            if state.get("per_snapshot_mean_models", False):
                mm_configs = state["mean_model_configs"]
                mm_states = state["mean_model_state_dicts"]
                sub_models: Dict[str, MeanModel] = {}
                for z_key in sorted(mm_configs):
                    cfg = mm_configs[z_key]
                    m = MeanModel(
                        y_dim=int(y_mean_np.shape[0]),
                        hidden_dim=int(cfg["hidden_dim"]),
                        use_redshift=False,
                        theta_dim=int(cfg["theta_dim"]),
                        theta_start_idx=int(cfg["theta_start_idx"]),
                        n_hidden_layers=int(cfg.get("n_hidden_layers", 2)),
                    )
                    m.load_state_dict(mm_states[z_key])
                    m.eval()
                    sub_models[z_key] = m
                mean_model = PerSnapshotMeanModel(sub_models)
                mean_model.eval()
            else:
                # ---- Single shared mean model (legacy) ----
                mm_sd = state["mean_model_state_dict"]
                first_weight_key = next((k for k in mm_sd if k.endswith("net.0.weight")), None)
                # Detect input dimension to infer redshift and theta usage.
                mm_in_dim = int(mm_sd[first_weight_key].shape[1]) if first_weight_key else 2
                mean_theta_dim = int(args.get("mean_theta_dim", 0))
                theta_start_idx = int(args.get("theta_start_idx", 2))
                # input = 2 (logM, logr) + theta_dim + (1 if redshift)
                mean_use_redshift = (mm_in_dim == 2 + mean_theta_dim + 1)
                # Infer n_hidden_layers from the number of weight matrices in the state dict.
                mm_weight_keys = sorted(k for k in mm_sd if k.startswith("net.") and k.endswith(".weight"))
                mm_n_hidden = max(1, len(mm_weight_keys) - 1)  # all but the output layer
                mean_model = MeanModel(
                    y_dim=int(y_mean_np.shape[0]),
                    hidden_dim=int(args.get("mean_hidden_dim", 128)),
                    use_redshift=mean_use_redshift,
                    theta_dim=mean_theta_dim,
                    theta_start_idx=theta_start_idx,
                    n_hidden_layers=mm_n_hidden,
                )
                mean_model.load_state_dict(mm_sd)
                mean_model.eval()

        dev = torch.device(device)
        model = model.to(dev)
        if mean_model is not None:
            mean_model = mean_model.to(dev)

        # Load parameter-aware CC predictor if available.
        cc_predictor = None
        cc_pred_sd = state.get("cc_predictor_state_dict", None)
        if cc_pred_sd is not None:
            theta_dim = int(args.get("theta_dim", 35))
            # Infer hidden_dim and n_layers from the saved state dict.
            layer_keys = [k for k in cc_pred_sd if k.startswith("net.") and k.endswith(".weight")]
            hidden_dim = int(cc_pred_sd[layer_keys[0]].shape[0]) if layer_keys else 128
            n_layers = max(1, len(layer_keys) - 1) if layer_keys else 3  # all except output layer
            out_dim = int(cc_pred_sd[layer_keys[-1]].shape[0]) if layer_keys else 2
            cc_mode = "binary" if out_dim == 1 else "continuous"
            cc_predictor = CCPredictor(
                theta_dim=theta_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                mode=cc_mode,
            )
            cc_predictor.load_state_dict(cc_pred_sd)
            cc_predictor.eval()
            cc_predictor = cc_predictor.to(dev)

        emu = cls(
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
            cc_prior=state.get("cc_prior", None),
            cc_predictor=cc_predictor,
        )
        # Store full norm dict for mass-redshift-aware denormalization.
        if norm.get("mass_redshift_aware", False):
            emu.norm_stats = norm
        return emu

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, device: str = "cpu", checkpoint_name: str = "best_model.pt") -> "Emulator":
        run_dir_path = Path(run_dir)
        requested_ckpt = run_dir_path / checkpoint_name
        if requested_ckpt.exists():
            return cls.from_checkpoint(requested_ckpt, device=device)

        # Fallback: scan epoch checkpoints and pick the one with the best
        # (lowest) val_weighted_orig metric.  If metrics are unavailable,
        # fall back to the latest epoch number.
        best_metric_ckpt: Optional[Path] = None
        best_metric_val: float = float("inf")
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
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                w = state.get("metrics", {}).get("val_weighted_orig", float("inf"))
                if w < best_metric_val:
                    best_metric_val = w
                    best_metric_ckpt = ckpt_path
            except Exception:
                pass

        chosen = best_metric_ckpt if best_metric_ckpt is not None else latest_epoch_ckpt
        if chosen is not None:
            return cls.from_checkpoint(chosen, device=device)

        raise FileNotFoundError(
            f"No checkpoint found in run directory {run_dir_path}. "
            f"Tried requested checkpoint '{checkpoint_name}' and fallback pattern 'epoch_*.pt'."
        )

    def available_fields(self) -> List[str]:
        return list(self.target_names)

    def sample_cc_indicator(
        self,
        log_masses: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample CC indicator values from the learned prior p(cc | log_M).

        Parameters
        ----------
        log_masses : (n_halo,) log10(M500c) values.
        rng : optional numpy random generator.

        Returns
        -------
        cc : (n_halo,) sampled CC feature values.
            - continuous mode: sampled log10(T_core / T500)
            - binary mode: sampled CC_tag in {0,1}
        """
        if rng is None:
            rng = np.random.default_rng()
        if self.cc_prior is None:
            raise RuntimeError(
                "No CC prior available in this checkpoint. "
                "Retrain with --cc-indicator to enable CC sampling."
            )
        prior = self.cc_prior
        edges = np.asarray(prior["bin_edges"], dtype=np.float64)

        logM = np.asarray(log_masses, dtype=np.float64)
        mode = str(prior.get("mode", self.cc_indicator_mode)).lower()

        if mode == "binary":
            probs = np.asarray(prior.get("bin_p_cc", []), dtype=np.float64)
            if probs.size == 0:
                gp = float(prior.get("global_p_cc", 0.5))
                probs = np.full(max(1, len(edges) - 1), gp, dtype=np.float64)
            bin_idx = np.digitize(logM, edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(probs) - 1)
            p = np.clip(probs[bin_idx], 1e-6, 1.0 - 1e-6)
            return rng.binomial(1, p).astype(np.float32)

        means = np.asarray(prior["bin_mean"], dtype=np.float64)
        stds = np.asarray(prior["bin_std"], dtype=np.float64)
        bin_idx = np.digitize(logM, edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(means) - 1)
        return rng.normal(means[bin_idx], stds[bin_idx]).astype(np.float32)

    @torch.no_grad()
    def sample_cc_indicator_parametric(
        self,
        log_masses: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample CC indicator from the parameter-aware predictor p(cc | logM, theta).

        Parameters
        ----------
        log_masses : (n_halo,) log10(M500c) values.
        theta : (theta_dim,) or (n_halo, theta_dim) feedback/cosmological parameters.
        rng : optional numpy random generator.

        Returns
        -------
        cc : (n_halo,) sampled CC feature values.
            - continuous mode: sampled log10(T_core / T500)
            - binary mode: sampled CC_tag in {0,1}
        """
        if self.cc_predictor is None:
            raise RuntimeError(
                "No parametric CC predictor available. "
                "Retrain with --cc-indicator to enable parameter-aware CC sampling."
            )
        if rng is None:
            rng = np.random.default_rng()

        logM_t = torch.tensor(np.asarray(log_masses, dtype=np.float32), device=self.device)
        theta_np = np.asarray(theta, dtype=np.float32)
        if theta_np.ndim == 1:
            theta_np = np.broadcast_to(theta_np[None, :], (len(logM_t), theta_np.shape[0])).copy()
        theta_t = torch.tensor(theta_np, device=self.device)

        if str(getattr(self.cc_predictor, "mode", self.cc_indicator_mode)).lower() == "binary":
            p_cc, _ = self.cc_predictor(logM_t, theta_t)
            p_np = np.clip(p_cc.cpu().numpy(), 1e-6, 1.0 - 1e-6)
            return rng.binomial(1, p_np).astype(np.float32)

        mu, sigma = self.cc_predictor(logM_t, theta_t)
        assert sigma is not None
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        return rng.normal(mu_np, sigma_np).astype(np.float32)

    def _cc_prob_from_value(self, cc_value: np.ndarray) -> np.ndarray:
        """Map explicit CC feature values to p(CC) per halo."""
        cc = np.asarray(cc_value, dtype=np.float32).ravel()
        if self.cc_indicator_mode == "binary":
            return (cc >= 0.5).astype(np.float32)
        return (cc < 0.0).astype(np.float32)

    def _cc_prob_from_prior(self, log_masses: np.ndarray) -> Optional[np.ndarray]:
        """Return p(CC | logM) from the stored prior when available."""
        if self.cc_prior is None:
            return None
        prior = self.cc_prior
        edges = np.asarray(prior["bin_edges"], dtype=np.float64)
        logM = np.asarray(log_masses, dtype=np.float64)
        mode = str(prior.get("mode", self.cc_indicator_mode)).lower()

        if mode == "binary":
            probs = np.asarray(prior.get("bin_p_cc", []), dtype=np.float64)
            if probs.size == 0:
                gp = float(prior.get("global_p_cc", 0.5))
                probs = np.full(max(1, len(edges) - 1), gp, dtype=np.float64)
            bin_idx = np.digitize(logM, edges) - 1
            bin_idx = np.clip(bin_idx, 0, len(probs) - 1)
            return np.clip(probs[bin_idx], 1e-6, 1.0 - 1e-6).astype(np.float32)

        means = np.asarray(prior.get("bin_mean", []), dtype=np.float64)
        stds = np.asarray(prior.get("bin_std", []), dtype=np.float64)
        if means.size == 0 or stds.size == 0:
            return None
        bin_idx = np.digitize(logM, edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(means) - 1)
        mu_t = torch.tensor(means[bin_idx], dtype=torch.float32)
        sigma_t = torch.tensor(np.clip(stds[bin_idx], 1e-6, None), dtype=torch.float32)
        p = torch.distributions.Normal(mu_t, sigma_t).cdf(torch.zeros_like(mu_t))
        return p.cpu().numpy().astype(np.float32)

    def predict(
        self,
        theta: np.ndarray,
        M: np.ndarray,
        r_bins: RadialBinsArg,
        field: FieldArg,
        snapnum: int | None = None,
        redshift: float | None = None,
        n_samples: int = 30,
        cc_indicator: np.ndarray | float | None = None,
        n_cc_samples: int = 8,
    ) -> PredictionResult:
        """Predict profiles for given halo masses and cosmological parameters.

        Parameters
        ----------
        cc_indicator : optional explicit CC indicator value(s). If ``None``
            and the model was trained with ``--cc-indicator``, values are
            automatically sampled from the learned mass-dependent prior and
            marginalized over ``n_cc_samples`` draws.
        n_cc_samples : number of independent CC prior draws to average over
            when ``cc_indicator`` is not provided. Ignored when the model has
            no CC feature or when an explicit value is given.
        """
        masses_arr = np.asarray(M, dtype=np.float32).ravel()

        # ---- Dual-head CC/NCC: blend both heads with p(CC) weights ----
        cc_dual = bool(self.args.get("cc_dual_head", False))
        if cc_dual:
            return self._predict_single(
                theta=theta, M=masses_arr, r_bins=r_bins, field=field,
                snapnum=snapnum, redshift=redshift, n_samples=n_samples,
                cc_indicator=cc_indicator,
            )

        need_cc_marginalization = (
            self.use_cc_indicator
            and (self.cc_prior is not None or self.cc_predictor is not None)
            and cc_indicator is None
        )

        if need_cc_marginalization:
            # Marginalize over the CC prior: run n_cc_samples forward passes,
            # each with an independent draw of cc_indicator, and average the
            # results in physical space.
            # Prefer the parameter-aware predictor p(cc | logM, theta) over
            # the mass-only prior p(cc | logM) when available.
            log_m_for_prior = np.log10(np.clip(masses_arr, 1e10, None))
            use_parametric = self.cc_predictor is not None
            rng = np.random.default_rng()
            mu_accum = None
            var_accum = None
            for _ in range(n_cc_samples):
                if use_parametric:
                    cc_draw = self.sample_cc_indicator_parametric(
                        log_m_for_prior, theta=theta, rng=rng,
                    )
                else:
                    cc_draw = self.sample_cc_indicator(log_m_for_prior, rng=rng)
                result_i = self._predict_single(
                    theta=theta, M=masses_arr, r_bins=r_bins, field=field,
                    snapnum=snapnum, redshift=redshift, n_samples=n_samples,
                    cc_indicator=cc_draw,
                )
                if mu_accum is None:
                    mu_accum = result_i.mean.copy()
                    var_accum = result_i.total_std.copy() ** 2
                else:
                    mu_accum += result_i.mean
                    var_accum += result_i.total_std ** 2
            mu_avg = mu_accum / n_cc_samples
            # Total variance = avg of per-sample variances + variance of per-sample means
            # (law of total variance). We approximate by averaging variances since the
            # per-sample mean shifts are already captured in the variance accumulation.
            std_avg = np.sqrt(var_accum / n_cc_samples)
            return PredictionResult(
                mean=mu_avg,
                total_std=std_avg,
                aleatoric_std=result_i.aleatoric_std,
                epistemic_std=result_i.epistemic_std,
                field_names=result_i.field_names,
            )
        else:
            # Direct prediction (no CC marginalization needed).
            cc_val = None
            if self.use_cc_indicator and cc_indicator is not None:
                cc_val = np.asarray(cc_indicator, dtype=np.float32)
                if cc_val.ndim == 0:
                    cc_val = np.full(len(masses_arr), float(cc_val), dtype=np.float32)
            return self._predict_single(
                theta=theta, M=masses_arr, r_bins=r_bins, field=field,
                snapnum=snapnum, redshift=redshift, n_samples=n_samples,
                cc_indicator=cc_val,
            )

    def _predict_single(
        self,
        theta: np.ndarray,
        M: np.ndarray,
        r_bins: RadialBinsArg,
        field: FieldArg,
        snapnum: int | None = None,
        redshift: float | None = None,
        n_samples: int = 30,
        cc_indicator: np.ndarray | None = None,
    ) -> PredictionResult:
        """Single forward pass with a fixed CC indicator (or None)."""
        batch, n_halo, n_r = self._build_zeroshot_batch(
            theta=theta,
            masses=M,
            r_bins=r_bins,
            snapnum=snapnum,
            redshift=redshift,
            cc_indicator=cc_indicator,
        )

        # Compute CC weights for dual-head blending.
        cc_weights = None
        cc_dual = bool(self.args.get("cc_dual_head", False))
        if cc_dual:
            masses_arr = np.asarray(M, dtype=np.float32).ravel()
            log_m = np.log10(np.clip(masses_arr, 1e10, None))
            p_cc_np: Optional[np.ndarray] = None

            # Explicit CC indicator/tag takes precedence.
            if cc_indicator is not None:
                cc_arr = np.asarray(cc_indicator, dtype=np.float32).ravel()
                if cc_arr.size == 1 and len(log_m) > 1:
                    cc_arr = np.full(len(log_m), float(cc_arr[0]), dtype=np.float32)
                elif cc_arr.size != len(log_m):
                    raise ValueError(
                        f"cc_indicator must have size 1 or len(M)={len(log_m)}, got {cc_arr.size}"
                    )
                p_cc_np = self._cc_prob_from_value(cc_arr)
            elif self.cc_predictor is not None:
                log_m_t = torch.tensor(log_m, device=self.device)
                theta_np = np.asarray(theta, dtype=np.float32)
                if theta_np.ndim == 1:
                    theta_np = np.broadcast_to(theta_np[None, :], (len(log_m), theta_np.shape[0])).copy()
                theta_t = torch.tensor(theta_np, device=self.device)
                cc_mode = str(getattr(self.cc_predictor, "mode", self.cc_indicator_mode)).lower()
                if cc_mode == "binary":
                    p_cc, _ = self.cc_predictor(log_m_t, theta_t)
                else:
                    mu_cc, sigma_cc = self.cc_predictor(log_m_t, theta_t)
                    assert sigma_cc is not None
                    p_cc = torch.distributions.Normal(mu_cc, sigma_cc).cdf(torch.zeros_like(mu_cc))
                p_cc_np = p_cc.detach().cpu().numpy().astype(np.float32)
            else:
                p_cc_np = self._cc_prob_from_prior(log_m)

            if p_cc_np is not None:
                p_cc_t = torch.tensor(p_cc_np, dtype=torch.float32, device=self.device)
                # Expand to (1, n_halo*n_r, 1) — uniform across radii per halo.
                p_cc_per_halo = p_cc_t.unsqueeze(1).expand(-1, n_r).reshape(1, -1, 1)
                cc_weights = p_cc_per_halo

        with torch.no_grad():
            mu, total_std, aleatoric_std, epistemic_std = self.model.predict(
                batch, device=self.device, n_samples=n_samples,
                cc_weights=cc_weights,
            )

            n_pts = n_halo * n_r
            mu = mu[:, :n_pts, :]
            total_std = total_std[:, :n_pts, :]
            aleatoric_std = aleatoric_std[:, :n_pts, :]
            epistemic_std = epistemic_std[:, :n_pts, :]

            # Denormalize using per-halo mass-redshift bin stats when available.
            if self.norm_stats is not None and self.norm_stats.get("mass_redshift_aware", False):
                # Recover raw log_m (pre-normalization) from the batch x input.
                tgt_x_raw = batch["tgt_x"].to(self.device)
                log_m_raw = (tgt_x_raw[0, :n_pts, 0] * self.x_std[0] + self.x_mean[0]).detach().cpu().numpy()
                # All radial bins for same halo share same mass — take every n_r-th value.
                log_m_per_halo = log_m_raw[::n_r]
                z_snap = redshift if redshift is not None else float(
                    self.redshift_by_snap.get(snapnum or self.default_snapnum, 0.0)
                )
                ym_np, ys_np = _lookup_bin_stats(self.norm_stats, log_m_per_halo, z_snap)
                # ym_np: (n_halo, 1, y_dim), ys_np: (n_halo, 1, y_dim)
                # Expand to (1, n_halo*n_r, y_dim)
                ym_exp = np.repeat(ym_np, n_r, axis=1).reshape(1, n_pts, -1)
                ys_exp = np.repeat(ys_np, n_r, axis=1).reshape(1, n_pts, -1)
                ym_t = torch.tensor(ym_exp, dtype=mu.dtype, device=mu.device)
                ys_t = torch.tensor(ys_exp, dtype=mu.dtype, device=mu.device)
                mu_o = mu * ys_t + ym_t
                mu_o = add_mean_back(mu_o, batch["tgt_x"].to(self.device), self.x_mean, self.x_std, self.mean_model)
                mu_o_log = mu_o
                total_std_o = (total_std * ys_t).clamp_min(1e-6)
                aleatoric_std_o = (aleatoric_std * ys_t).clamp_min(1e-6)
                epistemic_std_o = (epistemic_std * ys_t).clamp_min(1e-6)
            else:
                mu_o = denorm_y(mu, self.y_mean, self.y_std)
                mu_o = add_mean_back(mu_o, batch["tgt_x"].to(self.device), self.x_mean, self.x_std, self.mean_model)
                mu_o_log = mu_o
                total_std_o = (total_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)
                aleatoric_std_o = (aleatoric_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)
                epistemic_std_o = (epistemic_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)

            # Save log-space predictions before physical-unit transform.
            mu_log_save = mu_o_log[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()
            std_log_save = total_std_o[0].reshape(n_halo, n_r, -1).detach().cpu().numpy()

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
        mu_log_np = mu_log_save[..., idx]
        std_log_np = std_log_save[..., idx]

        if single:
            mu_np = mu_np[..., 0]
            total_np = total_np[..., 0]
            ale_np = ale_np[..., 0]
            epi_np = epi_np[..., 0]
            mu_log_np = mu_log_np[..., 0]
            std_log_np = std_log_np[..., 0]

        return PredictionResult(
            mean=mu_np,
            total_std=total_np,
            aleatoric_std=ale_np,
            epistemic_std=epi_np,
            field_names=names,
            mean_log10=mu_log_np,
            std_log10=std_log_np,
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

    def _nearest_snapnum_for_redshift(self, redshift: float) -> int:
        if not self.redshift_by_snap:
            return int(self.default_snapnum)
        pairs = [(s, z) for s, z in self.redshift_by_snap.items() if int(s) in self.snap_to_idx]
        if not pairs:
            return int(self.default_snapnum)
        z_tgt = float(redshift)
        snap_sel, _ = min(pairs, key=lambda t: abs(float(t[1]) - z_tgt))
        return int(snap_sel)

    def _redshift_for_snapnum(self, snapnum: int) -> float | None:
        return self.redshift_by_snap.get(int(snapnum), None)

    def _build_zeroshot_batch(
        self,
        theta: np.ndarray,
        masses: np.ndarray,
        r_bins: RadialBinsArg,
        snapnum: int | None = None,
        redshift: float | None = None,
        cc_indicator: np.ndarray | None = None,
    ):
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

        if snapnum is not None:
            chosen_snap = int(snapnum)
        elif redshift is not None:
            chosen_snap = int(self._nearest_snapnum_for_redshift(float(redshift)))
        else:
            chosen_snap = int(self.default_snapnum)
        if chosen_snap not in self.snap_to_idx:
            raise ValueError(
                f"Unknown snapnum {chosen_snap}. Available snapshots in this checkpoint: {self.snapnums}"
            )
        snap_idx = int(self.snap_to_idx[chosen_snap])

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
        if self.use_continuous_redshift_feature and 0 <= self.redshift_feature_idx < self.raw_x_dim:
            if redshift is not None:
                z_value = float(redshift)
            else:
                z_from_snap = self._redshift_for_snapnum(chosen_snap)
                z_value = float(0.0 if z_from_snap is None else z_from_snap)
            x_raw[..., self.redshift_feature_idx] = z_value

        if self.use_cc_indicator and 0 <= self.cc_indicator_feature_idx < self.raw_x_dim:
            if cc_indicator is not None:
                cc_vals = np.asarray(cc_indicator, dtype=np.float32)
                if cc_vals.ndim == 0:
                    cc_vals = np.full(n_halo, float(cc_vals), dtype=np.float32)
                x_raw[..., self.cc_indicator_feature_idx] = cc_vals[:, None]

        x_norm = (x_raw - self.x_mean.detach().cpu().numpy()[None, None, :]) / self.x_std.detach().cpu().numpy()[None, None, :]
        x_norm = x_norm.reshape(1, n_halo * n_r, self.raw_x_dim)

        tgt_x = torch.tensor(x_norm, dtype=torch.float32)
        tgt_mask = torch.ones((1, n_halo * n_r), dtype=torch.bool)
        tgt_snap = torch.full((1, n_halo * n_r), int(snap_idx), dtype=torch.long)

        # Use a neutral zero-shot context token so we do not leak any target x
        # back into the context path. Keep it unmasked to avoid all-masked
        # attention edge cases in the current model implementation.
        ctx_x = torch.zeros((1, 1, self.raw_x_dim), dtype=torch.float32)
        ctx_mask = torch.ones((1, 1), dtype=torch.bool)
        ctx_snap = torch.full((1, 1), int(snap_idx), dtype=torch.long)

        y_dim = len(self.target_names)
        batch = {
            "ctx_x": ctx_x,
            "ctx_y": torch.zeros((1, 1, y_dim), dtype=torch.float32),
            "ctx_snap": ctx_snap,
            "tgt_x": tgt_x,
            "tgt_y": torch.zeros((1, n_halo * n_r, y_dim), dtype=torch.float32),
            "tgt_snap": tgt_snap,
            "ctx_mask": ctx_mask,
            "tgt_mask": tgt_mask,
            "meta": [{"run_id": -1, "snapnum": chosen_snap, "snap_idx": snap_idx, "n_halo": n_halo, "n_r": n_r, "n_c": 1}],
        }
        return batch, n_halo, n_r

    # ------------------------------------------------------------------
    # Differentiable forward pass (for gradient-based sampling / HMC)
    # ------------------------------------------------------------------

    def predict_log10_differentiable(
        self,
        theta: torch.Tensor,
        M: np.ndarray,
        r_bins: RadialBinsArg,
        field: FieldArg,
        snapnum: int | None = None,
        redshift: float | None = None,
        n_samples: int = 10,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable forward pass returning log10-space (mean, std).

        Unlike :meth:`predict`, this keeps the computation graph intact so
        that ``torch.autograd.grad(output, theta)`` works.  This is the
        building block for HMC / gradient-based inference.

        Parameters
        ----------
        theta : torch.Tensor
            Shape ``(theta_dim,)`` with ``requires_grad=True``.
        M, r_bins, field, snapnum, redshift, n_samples
            Same semantics as :meth:`predict`.

        Returns
        -------
        mu_log10 : torch.Tensor   – shape ``(n_halo, n_r, n_field)``
        std_log10 : torch.Tensor  – shape ``(n_halo, n_r, n_field)``
        """
        # --- fixed geometry (no grad needed) ---
        masses = np.asarray(M, dtype=np.float32).ravel()
        n_halo = len(masses)
        r_bins_arr = np.asarray(r_bins, dtype=np.float32)
        if r_bins_arr.ndim == 1:
            r_bins_use = np.repeat(r_bins_arr[None, :], n_halo, axis=0)
        else:
            r_bins_use = r_bins_arr
        n_r = r_bins_use.shape[1]

        if snapnum is not None:
            chosen_snap = int(snapnum)
        elif redshift is not None:
            chosen_snap = int(self._nearest_snapnum_for_redshift(float(redshift)))
        else:
            chosen_snap = int(self.default_snapnum)
        snap_idx = int(self.snap_to_idx[chosen_snap])
        z_eval = redshift if redshift is not None else float(
            self.redshift_by_snap.get(chosen_snap, 0.0)
        )

        log_m = np.log10(np.clip(masses, 1e10, None))
        log_r = np.log10(np.clip(r_bins_use, 1e-4, None))

        dev = self.device
        x_mean_d = self.x_mean.to(dev)
        x_std_d = self.x_std.to(dev)

        # Build the non-theta part of x_raw as a detached tensor.
        x_raw_np = np.zeros((n_halo, n_r, self.raw_x_dim), dtype=np.float32)
        x_raw_np[..., 0] = log_m[:, None]
        x_raw_np[..., 1] = log_r
        if self.use_continuous_redshift_feature and 0 <= self.redshift_feature_idx < self.raw_x_dim:
            x_raw_np[..., self.redshift_feature_idx] = z_eval
        x_raw_base = torch.tensor(x_raw_np, dtype=torch.float32, device=dev)

        # Inject theta via torch.cat (not in-place) so autograd tracks it.
        ts = self.theta_start_idx
        theta_d = theta.to(dev).float()
        theta_block = theta_d.unsqueeze(0).unsqueeze(0).expand(n_halo, n_r, -1)
        parts = []
        if ts > 0:
            parts.append(x_raw_base[:, :, :ts])
        parts.append(theta_block)
        te = ts + self.theta_dim
        if te < self.raw_x_dim:
            parts.append(x_raw_base[:, :, te:])
        x_raw = torch.cat(parts, dim=-1)

        # Normalize x with torch ops so grad flows through theta.
        x_norm = (x_raw - x_mean_d.view(1, 1, -1)) / x_std_d.view(1, 1, -1)
        tgt_x = x_norm.reshape(1, n_halo * n_r, self.raw_x_dim)

        tgt_snap = torch.full((1, n_halo * n_r), snap_idx, dtype=torch.long, device=dev)
        tgt_mask = torch.ones((1, n_halo * n_r), dtype=torch.bool, device=dev)
        ctx_x = torch.zeros((1, 1, self.raw_x_dim), dtype=torch.float32, device=dev)
        ctx_mask = torch.ones((1, 1), dtype=torch.bool, device=dev)
        ctx_snap = torch.full((1, 1), snap_idx, dtype=torch.long, device=dev)
        y_dim = len(self.target_names)

        batch = {
            "ctx_x": ctx_x,
            "ctx_y": torch.zeros((1, 1, y_dim), dtype=torch.float32, device=dev),
            "ctx_snap": ctx_snap,
            "tgt_x": tgt_x,
            "tgt_y": torch.zeros((1, n_halo * n_r, y_dim), dtype=torch.float32, device=dev),
            "tgt_snap": tgt_snap,
            "ctx_mask": ctx_mask,
            "tgt_mask": tgt_mask,
            "meta": [{"run_id": -1, "snapnum": chosen_snap, "snap_idx": snap_idx,
                       "n_halo": n_halo, "n_r": n_r, "n_c": 1}],
        }

        # --- Model forward (WITH gradients) ---
        self.model.eval()
        ctx_x_e = self.model._fuse_time(self.model._embed_x(batch["ctx_x"]), batch["ctx_snap"])
        tgt_x_e = self.model._fuse_time(self.model._embed_x(batch["tgt_x"]), batch["tgt_snap"])

        q_ctx = self.model.latent(ctx_x_e, batch["ctx_y"], batch["ctx_mask"])
        r = self.model.det(ctx_x_e, batch["ctx_y"], batch["ctx_mask"], tgt_x_e, batch["tgt_mask"])

        if deterministic:
            # Use the latent mean — gives a smooth, deterministic energy
            # surface required for correct HMC / leapfrog integration.
            z = q_ctx.mean
            mu_s, sig_s = self.model.dec(tgt_x_e, r, z)
            pred_mean = mu_s
            total_std = self.model._aleatoric_var_from_scale(sig_s).sqrt()
        else:
            mus, sigs = [], []
            for _ in range(n_samples):
                z = q_ctx.rsample()
                mu_s, sig_s = self.model.dec(tgt_x_e, r, z)
                mus.append(mu_s)
                sigs.append(sig_s)
            mus_t = torch.stack(mus, dim=0)
            sigs_t = torch.stack(sigs, dim=0)

            pred_mean = mus_t.mean(0)                        # (1, n_pts, y_dim)
            aleatoric_var = self.model._aleatoric_var_from_scale(sigs_t).mean(0)
            epistemic_var = mus_t.var(0, unbiased=False)
            total_std = (aleatoric_var + epistemic_var).sqrt()

        n_pts = n_halo * n_r
        pred_mean = pred_mean[:, :n_pts, :]
        total_std = total_std[:, :n_pts, :]

        # --- Denormalize to log10 space ---
        if self.norm_stats is not None and self.norm_stats.get("mass_redshift_aware", False):
            ym_np, ys_np = _lookup_bin_stats(self.norm_stats, log_m, z_eval)
            ym_exp = np.repeat(ym_np, n_r, axis=1).reshape(1, n_pts, -1)
            ys_exp = np.repeat(ys_np, n_r, axis=1).reshape(1, n_pts, -1)
            ym_t = torch.tensor(ym_exp, dtype=pred_mean.dtype, device=dev)
            ys_t = torch.tensor(ys_exp, dtype=pred_mean.dtype, device=dev)
            mu_o = pred_mean * ys_t + ym_t
            std_o = (total_std * ys_t).clamp_min(1e-6)
        else:
            mu_o = pred_mean * self.y_std.view(1, 1, -1) + self.y_mean.view(1, 1, -1)
            std_o = (total_std * self.y_std.view(1, 1, -1)).clamp_min(1e-6)

        # Add mean model (differentiable — theta flows through it).
        if self.mean_model is not None:
            from train_anp_emulator import predict_mean_from_raw_x
            # Reconstruct raw x from normalized tgt_x for mean model.
            x_raw_full = tgt_x * x_std_d.view(1, 1, -1) + x_mean_d.view(1, 1, -1)
            mean_y = predict_mean_from_raw_x(x_raw_full, self.mean_model)
            mu_o = mu_o + mean_y

        # Reshape to (n_halo, n_r, y_dim) — squeeze the batch dim.
        mu_log10 = mu_o[0].reshape(n_halo, n_r, -1)
        std_log10 = std_o[0].reshape(n_halo, n_r, -1)

        # Field selection.
        idx, names, single = self._resolve_field_selection(field)
        mu_log10 = mu_log10[..., idx]
        std_log10 = std_log10[..., idx]

        return mu_log10, std_log10
