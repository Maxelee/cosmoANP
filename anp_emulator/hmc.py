"""
Hamiltonian Monte Carlo (HMC) sampler built on the differentiable ANP emulator.

The emulator is a PyTorch neural network, so we can backpropagate through
the likelihood to get exact gradients of log p(data | theta) with respect to
the 35 cosmological + astrophysical parameters — enabling efficient HMC
sampling that would be infeasible with finite-difference gradients.

Usage
-----
>>> from anp_emulator import Emulator
>>> from anp_emulator.hmc import HMCSampler
>>> emu = Emulator.from_run_dir("anp_training_runs/anp_all_profiles_20260325_175639")
>>> sampler = HMCSampler(emu, y_obs=..., sigma_obs=..., M=..., r_bins=...,
...                       prior_lo=..., prior_hi=...)
>>> chains = sampler.run(n_samples=2000, n_warmup=500, n_chains=4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .api import Emulator

TARGET_FIELDS = ["temperature", "pressure", "gas_density", "metallicity"]


@dataclass
class HMCResult:
    """Container for HMC sampling results."""
    samples: np.ndarray          # (n_chains, n_samples, theta_dim)
    log_prob: np.ndarray         # (n_chains, n_samples)
    accept_rate: np.ndarray      # (n_chains,)
    param_names: List[str]
    n_warmup: int
    step_size: np.ndarray        # (n_chains,) — adapted step sizes
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def flat_samples(self) -> np.ndarray:
        """Post-warmup samples flattened across chains: (n_effective, theta_dim)."""
        return self.samples[:, self.n_warmup:, :].reshape(-1, self.samples.shape[-1])

    @property
    def flat_log_prob(self) -> np.ndarray:
        return self.log_prob[:, self.n_warmup:].ravel()


class HMCSampler:
    """
    NUTS-like HMC sampler using the differentiable ANP emulator likelihood.

    The sampler implements a simplified No-U-Turn variant with dual averaging
    for step-size adaptation during warmup.

    Parameters
    ----------
    emu : Emulator
        Trained ANP emulator with ``predict_log10_differentiable`` available.
    y_obs : np.ndarray
        Observed profile in **log10** space, shape ``(n_halo, n_r, n_field)``
        or ``(n_r, n_field)`` for a single halo.
    sigma_obs : np.ndarray or float
        Observational noise in **log10 (dex)** space.  Scalar, per-field
        ``(n_field,)``, or full ``(n_halo, n_r, n_field)``.
    M : np.ndarray
        Halo masses, shape ``(n_halo,)``.
    r_bins : np.ndarray
        Radial bins, shape ``(n_halo, n_r)`` or ``(n_r,)``.
    prior_lo, prior_hi : np.ndarray
        Flat (uniform) prior bounds, shape ``(theta_dim,)``.
    field : str or list[str]
        Which target fields to include in the likelihood.
    snapnum, redshift : optional
        Snapshot / redshift for the emulator prediction.
    n_leapfrog : int
        Number of leapfrog steps per proposal (default: 25).
    """

    def __init__(
        self,
        emu: Emulator,
        y_obs: np.ndarray,
        sigma_obs,
        M: np.ndarray,
        r_bins: np.ndarray,
        prior_lo: np.ndarray,
        prior_hi: np.ndarray,
        field: Sequence[str] = TARGET_FIELDS,
        snapnum: int | None = None,
        redshift: float | None = None,
        n_leapfrog: int = 25,
        n_samples_per_eval: int = 5,
        include_model_error: bool = False,
        include_log_det: bool = True,
        stack_profiles: bool = False,
        stack_weights: np.ndarray | None = None,
    ):
        self.emu = emu
        self.field = [field] if isinstance(field, str) else list(field)
        self.snapnum = snapnum
        self.redshift = redshift
        self.n_leapfrog = n_leapfrog
        self.n_samples_per_eval = n_samples_per_eval
        self.include_model_error = bool(include_model_error)
        self.include_log_det = bool(include_log_det)
        self.stack_profiles = bool(stack_profiles)

        self.M = np.asarray(M, dtype=np.float32).ravel()
        self.r_bins = np.asarray(r_bins, dtype=np.float32)

        n_field = len(self.field)
        y = np.asarray(y_obs, dtype=np.float64)
        # Normalize to (n_halo, n_r, n_field)
        if y.ndim == 1:
            y = y.reshape(1, -1, 1)
        elif y.ndim == 2:
            if n_field == 1:
                # (n_halo, n_r) with squeezed field dim
                y = y[..., None]
            else:
                # (n_r, n_field) — single halo
                y = y[None, ...]
        assert y.ndim == 3 and y.shape[-1] == n_field, (
            f"y_obs shape {y.shape} incompatible with {n_field} fields"
        )
        self.y_obs = torch.tensor(y, dtype=torch.float64, device=emu.device)

        s = np.asarray(sigma_obs, dtype=np.float64)
        if s.ndim == 0:
            s = np.broadcast_to(s, y.shape)
        elif s.ndim == 1:
            s = np.broadcast_to(s[None, None, :], y.shape)
        elif s.ndim == 2:
            if n_field == 1:
                s = s[..., None]
            else:
                s = s[None, ...]
        s = np.broadcast_to(s, y.shape)
        self.sigma_obs = torch.tensor(s, dtype=torch.float64, device=emu.device)

        self.prior_lo = torch.tensor(prior_lo, dtype=torch.float64, device=emu.device)
        self.prior_hi = torch.tensor(prior_hi, dtype=torch.float64, device=emu.device)
        self.theta_dim = int(emu.theta_dim)
        self.prior_lo_np = np.asarray(prior_lo, dtype=np.float64).copy()
        self.prior_hi_np = np.asarray(prior_hi, dtype=np.float64).copy()
        self.prior_center = 0.5 * (self.prior_lo_np + self.prior_hi_np)
        self.prior_half_width = np.maximum(0.5 * (self.prior_hi_np - self.prior_lo_np), 1e-12)

        # Whitened coordinates: z in [-1, 1]^d maps to theta = center + half_width * z.
        self.z_lo = -np.ones(self.theta_dim, dtype=np.float64)
        self.z_hi = np.ones(self.theta_dim, dtype=np.float64)

        if self.stack_profiles:
            if stack_weights is None:
                w = np.ones_like(self.M, dtype=np.float64)
            else:
                w = np.asarray(stack_weights, dtype=np.float64).ravel()
                if w.shape[0] != self.M.shape[0]:
                    raise ValueError(
                        f"stack_weights length ({w.shape[0]}) must match len(M) ({self.M.shape[0]})"
                    )
            if np.any(w < 0):
                raise ValueError("stack_weights must be non-negative")
            if np.sum(w) <= 0:
                raise ValueError("stack_weights must have positive sum")
            w = w / np.sum(w)
            self.stack_weights = torch.tensor(w, dtype=torch.float64, device=emu.device).view(-1, 1, 1)
        else:
            self.stack_weights = None

        # Valid mask: only include bins where y_obs is finite and non-extreme
        self.valid = torch.isfinite(self.y_obs) & (self.y_obs > -30.0)

        # Precompute constant: -0.5 / sigma^2
        self.inv_var = -0.5 / (self.sigma_obs ** 2).clamp_min(1e-30)

    def _log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Flat prior: 0 inside bounds, -inf outside."""
        in_bounds = ((theta >= self.prior_lo) & (theta <= self.prior_hi)).all()
        return torch.tensor(0.0, dtype=torch.float64, device=theta.device) if in_bounds else torch.tensor(-float("inf"), dtype=torch.float64, device=theta.device)

    def _theta_to_z(self, theta: np.ndarray) -> np.ndarray:
        return (np.asarray(theta, dtype=np.float64) - self.prior_center) / self.prior_half_width

    def _z_to_theta(self, z: np.ndarray) -> np.ndarray:
        return self.prior_center + self.prior_half_width * np.asarray(z, dtype=np.float64)

    def _log_likelihood_and_grad(
        self, theta_np: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute log-likelihood and its gradient w.r.t. theta.

        Uses deterministic=True so the latent mean (not random samples) is
        used — this gives a smooth energy surface required for correct
        leapfrog integration.
        """
        theta = torch.tensor(theta_np, dtype=torch.float32, device=self.emu.device, requires_grad=True)

        mu_log10, std_log10 = self.emu.predict_log10_differentiable(
            theta=theta,
            M=self.M,
            r_bins=self.r_bins,
            field=self.field,
            snapnum=self.snapnum,
            redshift=self.redshift,
            n_samples=self.n_samples_per_eval,
            deterministic=True,
        )

        # Log-likelihood: Gaussian in log10 space.
        mu_d = mu_log10.to(torch.float64)
        std_model = std_log10.to(torch.float64).clamp_min(1e-12)

        if self.stack_profiles:
            # Build stacked prediction and stacked model variance with
            # Var(sum w_i x_i) = sum (w_i^2 Var(x_i)) for independent terms.
            mu_d = (self.stack_weights * mu_d).sum(dim=0, keepdim=True)
            model_var = (self.stack_weights.pow(2) * (std_model ** 2)).sum(dim=0, keepdim=True)
        else:
            model_var = std_model ** 2

        resid = self.y_obs - mu_d
        if self.include_model_error:
            total_var = (self.sigma_obs ** 2 + model_var).clamp_min(1e-30)
        else:
            total_var = (self.sigma_obs ** 2).clamp_min(1e-30)

        # Dynamic finite mask: keep only entries that are valid in data space
        # and also finite in the model prediction and variance terms.
        finite_mask = (
            self.valid
            & torch.isfinite(mu_d)
            & torch.isfinite(total_var)
            & torch.isfinite(resid)
        )
        if not bool(torch.any(finite_mask)):
            # No usable points for this theta; return a very bad state.
            return -float("inf"), np.zeros_like(theta_np, dtype=np.float64)

        # Build numerically safe tensors so masked-out NaNs/Infs cannot leak
        # into autograd and corrupt gradients.
        safe_resid = torch.where(finite_mask, resid, torch.zeros_like(resid))
        safe_total_var = torch.where(finite_mask, total_var, torch.ones_like(total_var))

        ll_elementwise = -0.5 * (safe_resid ** 2) / safe_total_var
        if self.include_log_det:
            ll_elementwise = ll_elementwise - 0.5 * torch.log(2.0 * np.pi * safe_total_var)

        ll = ll_elementwise[finite_mask].sum()

        if not bool(torch.isfinite(ll)):
            return -float("inf"), np.zeros_like(theta_np, dtype=np.float64)

        ll.backward()
        grad = theta.grad.detach().cpu().to(torch.float64).numpy().copy()
        if not np.all(np.isfinite(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        return float(ll.detach().cpu()), grad

    def _log_likelihood_and_grad_z(
        self, z_np: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute log-likelihood and gradient in whitened z-space."""
        theta_np = self._z_to_theta(z_np)
        ll, grad_theta = self._log_likelihood_and_grad(theta_np)
        grad_z = grad_theta * self.prior_half_width
        return ll, grad_z

    def _leapfrog(
        self,
        theta: np.ndarray,
        momentum: np.ndarray,
        step_size: float,
        n_steps: int,
        lo: np.ndarray,
        hi: np.ndarray,
        inv_mass: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Leapfrog integration with optional diagonal mass matrix.

        Parameters
        ----------
        inv_mass : optional (d,) array
            Diagonal of the inverse mass matrix.  When *None*, identity is used.
        """
        theta = theta.copy()
        momentum = momentum.copy()
        if inv_mass is None:
            inv_mass = np.ones_like(theta)

        log_prob, grad = self._log_likelihood_and_grad_z(theta)

        # Half step for momentum
        momentum += 0.5 * step_size * grad

        for i in range(n_steps - 1):
            # Full step for position  (v = M^{-1} p)
            theta += step_size * inv_mass * momentum

            # Reflect off prior bounds
            theta, momentum = self._reflect(theta, momentum, lo, hi)

            log_prob, grad = self._log_likelihood_and_grad_z(theta)

            # Full step for momentum
            momentum += step_size * grad

        # Last full step for position
        theta += step_size * inv_mass * momentum
        theta, momentum = self._reflect(theta, momentum, lo, hi)

        log_prob, grad = self._log_likelihood_and_grad_z(theta)

        # Half step for momentum
        momentum += 0.5 * step_size * grad

        return theta, momentum, log_prob

    def _reflect(
        self, theta: np.ndarray, momentum: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reflect theta off prior boundaries to keep it in bounds."""
        for d in range(len(theta)):
            width = hi[d] - lo[d]
            if width <= 0:
                theta[d] = lo[d]
                momentum[d] = 0.0
                continue

            # Fast mirrored folding into [lo, hi] without unbounded while-loops.
            x = theta[d] - lo[d]
            y = np.mod(x, 2.0 * width)
            if y <= width:
                theta[d] = lo[d] + y
                n_reflect = int(np.floor(x / width))
            else:
                theta[d] = hi[d] - (y - width)
                n_reflect = int(np.floor(x / width)) + 1

            # Odd number of wall crossings flips momentum sign.
            if n_reflect % 2 != 0:
                momentum[d] = -momentum[d]
        return theta, momentum

    def _hamiltonian(
        self, log_prob: float, momentum: np.ndarray,
        inv_mass: np.ndarray | None = None,
    ) -> float:
        """H = -log_prob + 0.5 * p^T M^{-1} p."""
        if inv_mass is None:
            ke = 0.5 * np.dot(momentum, momentum)
        else:
            ke = 0.5 * np.dot(momentum, inv_mass * momentum)
        return -log_prob + ke

    def run(
        self,
        n_samples: int = 2000,
        n_warmup: int = 500,
        n_chains: int = 4,
        init_theta: np.ndarray | None = None,
        init_step_size: float = 0.01,
        target_accept: float = 0.65,
        param_names: List[str] | None = None,
        verbose: bool = True,
        seed: int = 42,
    ) -> HMCResult:
        """
        Run HMC sampling.

        Parameters
        ----------
        n_samples : int
            Total samples per chain (including warmup).
        n_warmup : int
            Warmup samples for step-size adaptation.
        n_chains : int
            Number of independent chains.
        init_theta : optional
            Starting point(s), shape ``(theta_dim,)`` or ``(n_chains, theta_dim)``.
        init_step_size : float
            Initial leapfrog step size.
        target_accept : float
            Target MH acceptance rate for dual-averaging adaptation.
        param_names : optional
            Names for the theta dimensions.
        verbose : bool
            Print progress.
        seed : int
            Random seed.
        """
        rng = np.random.default_rng(seed)
        lo = self.z_lo.copy()
        hi = self.z_hi.copy()
        min_step_size = 1e-6
        max_step_size = 1.0

        # Initialize chains.
        if init_theta is None:
            starts = np.array([
                rng.uniform(lo, hi) for _ in range(n_chains)
            ])
        elif init_theta.ndim == 1:
            start_z = self._theta_to_z(np.asarray(init_theta, dtype=np.float64))
            starts = np.tile(start_z, (n_chains, 1))
            # Small jitter (0.3% of prior width) so chains start slightly
            # different but within each other's reachable exploration radius.
            # In high-D parameter spaces, larger jitter creates inter-chain
            # separation that exceeds what short chains can traverse, leading
            # to artificially inflated R-hat.
            for c in range(n_chains):
                jitter = rng.normal(0, 0.003 * (hi - lo), size=self.theta_dim)
                starts[c] = np.clip(starts[c] + jitter, lo, hi)
        else:
            starts = np.asarray([self._theta_to_z(v) for v in init_theta[:n_chains]], dtype=np.float64)

        all_samples = np.zeros((n_chains, n_samples, self.theta_dim))
        all_states = np.zeros((n_chains, n_samples, self.theta_dim))
        all_log_prob = np.zeros((n_chains, n_samples))
        all_accept = np.zeros(n_chains)
        all_step_size = np.zeros(n_chains)

        for chain_idx in range(n_chains):
            if verbose:
                print(f"\n  [HMC] Chain {chain_idx + 1}/{n_chains}")

            theta_current = starts[chain_idx].astype(np.float64)
            log_prob_current, _ = self._log_likelihood_and_grad_z(theta_current)

            # --- Windowed warmup (inspired by Stan) ---
            # Window 1: first half of warmup — identity mass, adapt step size.
            # Window 2: second half of warmup — estimate diagonal mass from
            #           window-1 samples, reset step-size adaptation.
            # This lets the sampler learn dimensional scales before sampling.
            mass_switch = n_warmup // 2 if n_warmup >= 20 else n_warmup
            inv_mass = np.ones(self.theta_dim)  # M^{-1} = I initially
            mass_diag = np.ones(self.theta_dim)  # M = I initially

            # Dual averaging for step-size adaptation (Hoffman & Gelman 2014).
            log_eps = np.log(init_step_size)
            log_eps_bar = 0.0
            H_bar = 0.0
            gamma = 0.05
            t0 = 10
            kappa = 0.75
            mu = np.log(10.0 * init_step_size)

            n_accept = 0
            step_size = init_step_size

            for i in range(n_samples):
                # At the mass-matrix switch point, estimate diagonal mass
                # from warmup samples and reset step-size adaptation.
                if i == mass_switch and mass_switch > 10:
                    warmup_samples = all_states[chain_idx, :mass_switch, :]
                    var_est = np.var(warmup_samples, axis=0, ddof=1)
                    raw_var = var_est.copy()
                    # Clamp to avoid degenerate scales that can freeze dynamics
                    # or trigger extremely large effective jumps.
                    var_est = np.clip(var_est, 1e-4, 1e4)
                    # If variance is essentially at floor in all dimensions,
                    # there is no trustworthy scale information yet.
                    # Keep identity mass to avoid pathological preconditioning.
                    degenerate_mass = np.all(raw_var <= 1e-6)
                    if not degenerate_mass:
                        # Use posterior scale estimate as the momentum covariance M,
                        # so the integrator uses M^{-1} in position updates.
                        mass_diag = var_est.copy()
                        inv_mass = 1.0 / mass_diag
                        # Reset dual averaging with new mass landscape.
                        log_eps = np.log(step_size)
                        log_eps_bar = 0.0
                        H_bar = 0.0
                        mu = np.log(10.0 * step_size)
                        if verbose:
                            scale_range = np.sqrt(var_est)
                            print(f"    [mass update at step {i}] "
                                  f"scale range: [{scale_range.min():.4g}, "
                                  f"{scale_range.max():.4g}]")
                    elif verbose:
                        print(f"    [mass update at step {i}] skipped (degenerate warmup variance)")

                # Draw momentum from N(0, M) where M = diag(mass_diag).
                momentum = rng.standard_normal(self.theta_dim) * np.sqrt(mass_diag)

                H_current = self._hamiltonian(log_prob_current, momentum, inv_mass)

                # Leapfrog integration.
                # Randomize number of leapfrog steps slightly for ergodicity.
                n_lf = max(1, int(rng.integers(
                    max(1, self.n_leapfrog // 2),
                    self.n_leapfrog + 1,
                )))
                theta_prop, momentum_prop, log_prob_prop = self._leapfrog(
                    theta_current, momentum, step_size, n_lf, lo, hi, inv_mass,
                )

                H_proposed = self._hamiltonian(log_prob_prop, momentum_prop, inv_mass)

                # Metropolis accept/reject.
                log_alpha = -(H_proposed - H_current)
                # If the proposal is non-finite, force rejection and adapt as
                # a failed proposal (accept_prob = 0).
                if (not np.isfinite(H_proposed)) or (not np.isfinite(log_alpha)):
                    accept_prob = 0.0
                else:
                    # Clamp for numerical stability.
                    log_alpha = min(0.0, log_alpha)
                    accept_prob = float(np.exp(log_alpha))

                if rng.random() < accept_prob and np.isfinite(H_proposed):
                    theta_current = theta_prop
                    log_prob_current = log_prob_prop
                    n_accept += 1

                all_states[chain_idx, i] = theta_current
                all_samples[chain_idx, i] = self._z_to_theta(theta_current)
                all_log_prob[chain_idx, i] = log_prob_current

                # Dual-averaging step-size adaptation during warmup.
                if i < n_warmup:
                    adapt_i = i - mass_switch if i >= mass_switch else i
                    adapt_i = max(adapt_i, 0)
                    w = 1.0 / (adapt_i + 1 + t0)
                    H_bar = (1 - w) * H_bar + w * (target_accept - accept_prob)
                    log_eps = mu - (np.sqrt(adapt_i + 1) / gamma) * H_bar
                    step_size = np.exp(log_eps)
                    m = (adapt_i + 1) ** (-kappa)
                    log_eps_bar = m * log_eps + (1 - m) * log_eps_bar

                    # Clamp step size to prevent extreme values.
                    step_size = np.clip(step_size, min_step_size, max_step_size)

                if verbose and (i + 1) % max(1, n_samples // 10) == 0:
                    rate = n_accept / (i + 1)
                    phase = "warmup" if i < n_warmup else "sample"
                    print(f"    [{phase}] {i+1}/{n_samples}  "
                          f"accept={rate:.2f}  eps={step_size:.4g}  "
                          f"log_p={log_prob_current:.1f}")

            # Use adapted step size for post-warmup.
            # Clamp here as well so dual-averaging history cannot yield
            # unstable values after warmup.
            if n_warmup > 0:
                final_step = float(np.clip(np.exp(log_eps_bar), min_step_size, max_step_size))
            else:
                final_step = float(np.clip(step_size, min_step_size, max_step_size))
            all_accept[chain_idx] = n_accept / n_samples
            all_step_size[chain_idx] = final_step

            if verbose:
                print(f"    Chain {chain_idx + 1} done: "
                      f"accept={all_accept[chain_idx]:.2f}, eps={final_step:.4g}")

        # Compute basic diagnostics.
        diagnostics = _compute_diagnostics(all_samples, all_log_prob, n_warmup)

        if param_names is None:
            param_names = [f"theta_{i}" for i in range(self.theta_dim)]

        return HMCResult(
            samples=all_samples,
            log_prob=all_log_prob,
            accept_rate=all_accept,
            param_names=param_names,
            n_warmup=n_warmup,
            step_size=all_step_size,
            diagnostics=diagnostics,
        )


def _compute_diagnostics(
    samples: np.ndarray,
    log_prob: np.ndarray,
    n_warmup: int,
) -> Dict[str, Any]:
    """Basic MCMC diagnostics: R-hat and effective sample size."""
    post = samples[:, n_warmup:, :]  # (n_chains, n_post, d)
    n_chains, n_post, d = post.shape
    diag: Dict[str, Any] = {}

    if n_chains < 2 or n_post < 10:
        return diag

    # Split R-hat (Gelman-Rubin) per parameter.
    rhat = np.zeros(d)
    for j in range(d):
        chain_means = post[:, :, j].mean(axis=1)
        chain_vars = post[:, :, j].var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = n_post * chain_means.var(ddof=1)
        var_hat = ((n_post - 1) / n_post) * W + (1.0 / n_post) * B
        # Guard: if W is near zero the chains haven't moved (stuck).
        # Report R-hat as NaN rather than an astronomically large number.
        if W < 1e-20:
            rhat[j] = float("nan")
        else:
            rhat[j] = np.sqrt(var_hat / W)
    diag["rhat"] = rhat
    diag["rhat_max"] = float(np.nanmax(rhat)) if np.any(np.isfinite(rhat)) else float("nan")

    # Rough effective sample size (per chain, autocorrelation-based).
    n_eff = np.zeros(d)
    for j in range(d):
        for c in range(n_chains):
            chain = post[c, :, j]
            chain = chain - chain.mean()
            var = np.var(chain)
            if var < 1e-30:
                continue
            # Simple initial positive sequence estimator.
            max_lag = min(n_post // 2, 500)
            acf_sum = 0.0
            for lag in range(1, max_lag):
                rho = np.mean(chain[lag:] * chain[:-lag]) / var
                if rho < 0.05:
                    break
                acf_sum += rho
            ess = n_post / (1 + 2 * acf_sum)
            n_eff[j] += ess
    diag["n_eff"] = n_eff
    diag["n_eff_min"] = float(n_eff.min())

    return diag
