# ANPcosmo Emulator Validation: A Rigorous Testing Suite for 35-Parameter Feedback Inference

## Executive Summary

Constraining 35 cosmological and astrophysical parameters from observed thermodynamic profiles of galaxy clusters requires the ANP emulator to be more than just "accurate on the test set." The emulator must (1) reproduce profiles faithfully enough that posterior constraints from mock data are unbiased, (2) have well-calibrated uncertainties so that credible intervals are trustworthy, and (3) remain robust to the noise levels and radial coverage of actual observations from eROSITA, ACT, SPT, and future experiments. This report proposes a hierarchical, ten-category test suite grounded in the current emulator literature, tailored to the ANP architecture and the CAMELS-zoomGZ simulation suite, and designed to answer the single most important question: *is this emulator good enough to do lossless, unbiased Bayesian inference across the 35-dimensional parameter space?*

***

## 1. Context: What the Emulator Must Achieve

### 1.1 The Downstream Task

The ultimate goal is to run a Bayesian likelihood pipeline of the form

\[
\mathcal{L}(\theta) = -\frac{1}{2}\left(\mathbf{d}_\mathrm{obs} - \boldsymbol{\mu}_\mathrm{ANP}(\theta)\right)^T \Sigma^{-1}_\mathrm{obs}\left(\mathbf{d}_\mathrm{obs} - \boldsymbol{\mu}_\mathrm{ANP}(\theta)\right)
\]

where \(\boldsymbol{\mu}_\mathrm{ANP}(\theta)\) is the emulated profile vector (stacked over profile types and radial bins), and \(\Sigma_\mathrm{obs}\) encodes observational noise (photon noise, background, systematics). The emulator replaces a hydrodynamical simulation inside the likelihood loop, evaluated millions of times during MCMC or nested sampling. Any systematic error in \(\boldsymbol{\mu}_\mathrm{ANP}\) biases the posterior; miscalibrated predictive uncertainties corrupt any scheme that propagates emulator error into the likelihood.

### 1.2 The Theoretical Accuracy Criterion

Bevins, Gessey-Jones & Handley (2025) provide a theoretically motivated upper bound on the Kullback–Leibler divergence between the emulated and true posteriors:[^1]

\[
\mathcal{D}_\mathrm{KL}\left(P_\epsilon \,\|\, P\right) \leq \frac{N_d}{2}\left(\frac{\mathrm{RMSE}}{\sigma_\mathrm{obs}}\right)^2
\]

For \(\mathcal{D}_\mathrm{KL} < 1\) (less than one nat of false information), the requirement is:

\[
\frac{\mathrm{RMSE}}{\sigma_\mathrm{obs}} \leq \sqrt{\frac{2}{N_d}}
\]

For a typical stacked cluster profile with \(N_d \sim 30\) radial bins, this gives RMSE ≤ 26% of the observational noise per bin. A more conservative operational target (consistent with the 21-cm emulator literature) is RMSE ≤ 10–15% of \(\sigma_\mathrm{obs}\). This is the single most important design criterion for the test suite: every accuracy metric should be quoted relative to the *observational* noise, not just the signal amplitude.[^2]

### 1.3 Current Testing Gaps

The [cosmoANP repository](https://github.com/Maxelee/cosmoANP) contains `diagnostics.py` with `rmse_by_field`, `residual_radius_summary`, `coverage_curve`, `pit_values`, and `uncertainty_error_rank_correlation`. Notebooks test the 1P set, CV set, and SB35 set individually. What is missing is a set of **inference-level** tests: mock parameter recovery, TARP coverage, KL divergence between emulated and direct posteriors, Fisher forecast consistency, and a direct comparison against observational data uncertainties. The tests proposed here fill those gaps.

***

## 2. Observational Noise and Accuracy Requirements

Before designing tests, it is necessary to anchor accuracy thresholds to real observational noise levels. The table below summarizes the relevant instruments and expected noise regimes for stacked cluster profiles.

| Observable | Instrument | Noise Regime | Relevant Scale |
|---|---|---|---:|
| X-ray surface brightness (0.5–2 keV) | eROSITA eRASS | ~10–30% per radial bin (stacked) | 10–400 kpc |
| Compton-y (tSZ) | Simons Observatory / ACT | ~5–20% per radial bin (stacked) | 0.1–1 R₂₀₀ |
| Gas temperature (spectroscopic) | Chandra/XMM-Newton | ~15–40% per annulus | 10–500 kpc |
| Gas density (X-ray deprojected) | Chandra/XMM-Newton | ~5–20% per bin | 10–500 kpc |
| Metallicity | Chandra/XMM-Newton | ~20–50% per bin | 50–300 kpc |

The Moser et al. (2022) CAMELS-SZ study demonstrated that for a DESI-like sample observed with the Simons Observatory, constraints of ~7–31% are achievable on the four primary CAMELS feedback parameters. The newer CAMELS-zoomGZ + eROSITA study found that X-ray emulators can constrain feedback parameters when compared to eRASS observations, with reduced χ² of the best-fit model at 0.83 for IllustrisTNG. These observational noise levels define the denominator in the RMSE/σ criterion.[^3][^4]

***

## 3. The Test Suite

### 3.1 Test Category 1: Point Accuracy on a Held-Out Test Set

**What it tests:** Whether the emulated *mean* profile is close to the simulation truth across the full 35D parameter space.

**Tests to run:**

- **RMSE per field, per radial bin.** For each profile type \(f \in \{T, \rho_g, Z, S_{Bx}, y\}\) and each radial bin \(r_i\), compute RMSE over a held-out Latin-hypercube test set of ≥200 parameter configurations (already partly implemented in `diagnostics.py`).

- **Fractional RMSE relative to observational noise.** Compute \(\epsilon_f(r_i) = \mathrm{RMSE}_f(r_i) / \sigma^\mathrm{obs}_f(r_i)\), where \(\sigma^\mathrm{obs}\) is the expected noise from eROSITA / SO at the corresponding radial bin. The criterion is \(\epsilon_f(r_i) \leq 0.15\) for all bins in any profile type to be used in inference.[^1]

- **Outlier fraction \(F_\mathrm{out}\).** For an observation-noise–weighted \(\chi^2\):
\[
\Delta\chi^2_j = \sum_{i} \frac{\left[\hat{\mu}_{j,i} - y_{j,i}^\mathrm{true}\right]^2}{\sigma^2_{\mathrm{obs},i}}
\]
compute the fraction of test points with \(\Delta\chi^2 > \Delta\chi^2_\mathrm{thresh}\). The threshold from attention-based CMB emulators and the DES science requirement document is \(\Delta\chi^2 = 0.2\); a more permissive but still scientifically motivated threshold for cluster profiles is \(\Delta\chi^2 \leq 1\). The target is \(F_\mathrm{out}(\Delta\chi^2 > 0.2) < 10\%\).[^5]

- **Pearson correlation coefficient \(r\) per parameter.** For each of the 35 parameters, train a simple inference network (as in Hernandez-Martinez et al. 2024) on the emulated profiles and compute the correlation between predicted and true parameter values on the test set. Target: \(r \geq 0.97\) for cosmological parameters, \(r \geq 0.90\) for astrophysical parameters, consistent with CAMELS-zoomGZ results.[^6]

**Implementation notes:** Use the SB35 test split (15% of total data) as the primary evaluation set. Report metrics stratified by halo mass bin (10¹³–10¹³·⁵, 10¹³·⁵–10¹⁴, 10¹⁴–10¹⁴·⁵ M☉) to check for mass-dependent failure modes.

***

### 3.2 Test Category 2: Uncertainty Calibration Tests

**What it tests:** Whether the ANP's predictive standard deviation \(\hat{\sigma}_\mathrm{ANP}\) is a reliable uncertainty estimate—not over- or under-confident.

**Tests to run:**

- **Coverage curves (PP plot / calibration curve).** For each profile type, at each nominal confidence level \(p \in [0.05, 0.95]\), compute the fraction of test-set residuals that fall within the corresponding Gaussian interval. Plot empirical coverage vs. nominal coverage. Ideal: the curve lies on the diagonal. A curve lying above the diagonal indicates over-dispersion (conservative); below indicates under-dispersion (overconfident). Already implemented in `coverage_curve` in `diagnostics.py` — but this must be done separately for **epistemic** \(\hat{\sigma}_\mathrm{epi}\) and **aleatoric** \(\hat{\sigma}_\mathrm{alea}\) components if the ANP decomposes them.

- **PIT histograms.** Compute Probability Integral Transform values:
\[
u_{j,i,f} = \Phi\left(\frac{y_{j,i,f}^\mathrm{true} - \hat{\mu}_{j,i,f}}{\hat{\sigma}_{j,i,f}}\right)
\]
and plot histograms over the test set for each field \(f\). A flat histogram indicates correct calibration. A U-shaped histogram indicates over-dispersion; hump-shaped indicates under-confidence. Apply a Kolmogorov–Smirnov test for uniformity and report the p-value.

- **Reduced chi-squared.** Compute the scalar:
\[
\chi^2_\mathrm{red} = \frac{1}{N_\mathrm{test} N_\mathrm{bins}} \sum_{j,i} \frac{(y_{j,i}^\mathrm{true} - \hat{\mu}_{j,i})^2}{\hat{\sigma}_{j,i}^2}
\]
Target: \(\chi^2_\mathrm{red} \approx 1\). Values significantly greater than 1 indicate the ANP is under-estimating uncertainty; significantly less than 1 indicates over-estimation (and wasted constraining power). Already implemented in `diagnostics.py` for the Hernandez-Martinez formulation.[^6]

- **Spearman rank correlation \(\rho(|\epsilon|, \hat{\sigma})\).** Already in `uncertainty_error_rank_correlation`. Target: \(\rho > 0.4\) for uncertainty to carry useful information about where the emulator is unreliable.

- **TARP (Tests of Accuracy with Random Points).** This is the most rigorous high-dimensional calibration test and is currently *absent* from the repo. TARP estimates the coverage probability of posterior credible regions without requiring explicit posterior evaluations[^7][^8]. For each test point \((\theta_j, x_j)\), draw a random reference point \(\theta_\mathrm{ref}\) from the prior, compute the distance \(\|(\hat{\theta} - \theta_\mathrm{ref})\|^2\) in the emulator's predictive space, and check if \(\theta_j^\mathrm{true}\) falls within the corresponding credible region. The TARP coverage plot should follow the diagonal[^9]. Deviations diagnose overconfident (below diagonal) or underconfident (above diagonal) posteriors at the inference level, and can detect failures invisible to bin-by-bin RMSE metrics[^10].

  ```python
  # Minimal TARP sketch using the `tarp` package
  # pip install tarp
  import tarp
  # theta_samples: (N_test, N_params, N_posterior_samples) from the emulator posterior
  # theta_true: (N_test, N_params)
  ecp, alpha = tarp.get_tarp_coverage(theta_samples, theta_true,
                                      norm=True, bootstrap=True)
  ```

***

### 3.3 Test Category 3: Single-Parameter (1P) Variation and Sensitivity Tests

**What it tests:** Whether the emulator correctly captures the qualitative and quantitative effect of each of the 35 parameters on the profiles.

**Tests to run:**

- **1P monotonicity test.** For each parameter \(\theta_k \in \{$\Omega_m$, $\sigma_8$, ASN1, AAGN1, ASN2, AAGN2, ...\}\), vary it across its prior range while holding all 34 others at fiducial values (matching the CAMELS 1P set structure). Generate emulated profiles at 10+ parameter values and check monotonicity of integrated quantities (e.g., \(Y_{200c}\) should increase with ASN1, metallicity should increase with AAGN2, etc.) against known physical expectations from IllustrisTNG single-parameter variation studies. The ANP already has a 1P test notebook; this should be formalized.[^11]

- **Derivative/sensitivity consistency.** Compute finite-difference emulator derivatives:
\[
\frac{\partial \mu_f(r_i)}{\partial \theta_k} \approx \frac{\hat{\mu}_f(r_i, \theta^\mathrm{fid} + \delta_k) - \hat{\mu}_f(r_i, \theta^\mathrm{fid} - \delta_k)}{2\delta_k}
\]
and compare against direct simulation derivatives from the CAMELS 1P set. The ratio (emulator derivative / simulation derivative) should be within 20% for all radial bins and parameter types. This test directly validates the Jacobian used in any Fisher forecast. Plot derivative profiles analogously to Figure 4 of Moser et al. (2022).[^12][^3]

- **Non-degeneracy test.** Following Hernandez-Martinez et al. (2024), verify that varying each of the 35 parameters produces a *unique* feature in the profile space—i.e., no two parameters produce profiles that are indistinguishable within the emulator's predictive uncertainty. This can be checked by computing the pairwise Bhattacharyya distance between 1P variation profiles. Any pair with distance < 1.0 indicates a potential degeneracy that will inflate posterior widths.[^6]

- **Sobol sensitivity indices.** For each profile type and radial bin, compute first-order and total-order Sobol sensitivity indices using the ANP's fast emulations (~10⁵ evaluations are cheap). This reveals which parameters dominate profile variance at which radii, and serves as an independent cross-check of the derivative test. Parameters identified as non-influential by Sobol indices but inferred to have tight posteriors are a red flag.[^11]

***

### 3.4 Test Category 4: Mock Parameter Recovery Tests

**What it tests:** Whether the emulator, used inside a Bayesian likelihood, recovers the true parameter values from mock observations. This is the most direct test of whether the emulator is good enough for the downstream inference task.

**Protocol:**

1. Select 20–50 test parameter configurations from the SB35 held-out set (spanning the full prior volume, not just near the fiducial).
2. Generate a mock observation by taking the emulated (or simulation) profile at \(\theta_j^\mathrm{true}\) and adding a synthetic noise realization \(\mathbf{n} \sim \mathcal{N}(0, \Sigma_\mathrm{obs})\) at realistic eROSITA / SO noise levels. Repeat with S/N levels corresponding to 10%, 20%, 40% Gaussian noise (matching the Hernandez-Martinez et al. noise suite).[^6]
3. Run nested sampling (e.g., `nautilus` or `dynesty`) with the emulator likelihood over the full 35D space, using the mock data as the observation.
4. Compute the following diagnostics:

- **Emulator bias** (from Bevins+ 2025):[^1]
\[
b_k = \frac{|\mu_k^{(\epsilon)} - \theta_k^\mathrm{true}|}{\sigma_k^\mathrm{posterior}}
\]
where \(\mu_k^{(\epsilon)}\) is the posterior mean from the emulator and \(\sigma_k\) is the posterior standard deviation. Target: \(b_k < 0.5\) for all 35 parameters, across all mock observations.

- **Coverage fraction.** Check that the 68% (1σ) confidence interval contains \(\theta_k^\mathrm{true}\) for approximately 68% of the 20–50 mock runs for each parameter. A systematic shortfall (say, 68% CI contains truth only 45% of the time) indicates bias or miscalibration.

- **KL divergence between emulated and direct simulation posteriors.** For 5–10 test points where direct simulation evaluation is feasible (e.g., using CAMELS 1P or CV simulations evaluated at known parameters), run inference with both the emulator and the direct simulation. Compute \(\mathcal{D}_\mathrm{KL}(P_\epsilon \| P_\mathrm{true})\) using the analytical bound in Eq. (32) of Bevins et al. (2025)[^1]. Target: \(\mathcal{D}_\mathrm{KL} < 1\) nat.

- **Posterior width ratio.** Compute \(\sigma_k^\mathrm{emulator} / \sigma_k^\mathrm{Fisher}\) for each parameter. This should be within a factor of ~2 of the Fisher forecast prediction. Large deviations indicate parameter space structure the Fisher approximation misses.

***

### 3.5 Test Category 5: Fisher Information Consistency Test

**What it tests:** Whether the emulator's derivatives are accurate enough to produce reliable Fisher matrix forecasts, and whether those forecasts are consistent with actual posterior widths from MCMC.

**Protocol:**

1. Compute the Fisher matrix at a fiducial cosmology \(\theta^\mathrm{fid}\):
\[
F_{kl} = \sum_{ij} \frac{\partial \mu_i(\theta^\mathrm{fid})}{\partial \theta_k} \left(\Sigma^{-1}_\mathrm{obs}\right)_{ij} \frac{\partial \mu_j(\theta^\mathrm{fid})}{\partial \theta_l}
\]
using emulator-based finite-difference derivatives as in Test Category 3. This follows the methodology of Moser et al. (2022) for the SZ profiles.[^3]

2. Invert to obtain predicted \(1\sigma\) marginalized parameter errors: \(\sigma_k^\mathrm{Fisher} = \sqrt{(F^{-1})_{kk}}\).

3. Compare to actual posterior widths from mock recovery (Category 4). The ratio \(\sigma_k^\mathrm{mock} / \sigma_k^\mathrm{Fisher}\) should be in [0.7, 2.0] for well-behaved parameters (Gaussian posteriors). Ratios outside this range signal non-Gaussianity, degeneracy, or emulator inaccuracy.

4. Compute the **condition number** of the Fisher matrix. A very high condition number (>10⁵) indicates near-degenerate parameters—the emulator may be amplifying numerical noise in derivatives. Cross-check by repeating with CARPoolGP derivatives from direct simulations.

5. Compare to the Moser+ 2022 CAMELS-TNG SZ forecasts for ASN1, AAGN1, ASN2, AAGN2 as an external sanity check: your emulator should reproduce (or improve on) their forecasted constraints for those four parameters.[^3]

***

### 3.6 Test Category 6: Context Set Size and ANP-Specific Tests

**What it tests:** Properties unique to the ANP architecture—context sensitivity, latent space behavior, and the effect of varying the context set composition.

**Tests to run:**

- **Context size convergence test.** Run predictions for fixed target parameters \(\theta^\mathrm{test}\) while varying the number of context points \(N_c \in \{1, 2, 5, 10, 20, 50, N_\mathrm{max}\}\). Plot the mean prediction and the epistemic uncertainty \(\hat{\sigma}_\mathrm{epi}(r)\) as a function of \(N_c\). The epistemic uncertainty should decrease monotonically with \(N_c\) and saturate at large \(N_c\); the mean prediction should stabilize. A minimum context size \(N_c^\mathrm{min}\) can be defined as the point where \(\hat{\sigma}_\mathrm{epi}(r) / \hat{\sigma}_\mathrm{epi}^\mathrm{max}(r) < 0.1\) for all radial bins. This is critical for deployment: you need to know how many context observations a given observation strategy requires.

- **Context diversity test.** Compare predictions when the context consists of: (a) profiles near the test point in parameter space, (b) profiles uniformly spread across the prior, (c) profiles from a single halo mass, vs. (d) profiles spanning multiple mass bins. Quantify the impact on \(\hat{\sigma}_\mathrm{epi}\) and prediction bias. This informs optimal context set design for actual observations.

- **Latent space interpolation test.** For pairs of parameter configurations \(\theta_A, \theta_B\) that span the prior, generate predictions at intermediate \(\theta_\mathrm{interp} = \lambda\theta_A + (1-\lambda)\theta_B\) for \(\lambda \in [0,1]\) and compare to direct simulation profiles. Smooth interpolation (profiles vary continuously) is expected from a well-trained ANP; discontinuities or sudden uncertainty inflation signal poor coverage in the training set.

- **Out-of-training-distribution epistemic inflation test.** Perturb a test parameter configuration to lie slightly outside the training prior bounds (by 5–10%). The ANP's epistemic uncertainty should inflate noticeably. If \(\hat{\sigma}_\mathrm{epi}\) does not increase, the model is not detecting that it is extrapolating, which would be dangerous for inference pipelines.[^13]

***

### 3.7 Test Category 7: Noise Robustness and Observational Realism Tests

**What it tests:** Whether the emulator's inference performance degrades gracefully as observational noise increases, and whether it remains informative at realistic noise levels from eROSITA and SO.

**Protocol (following Hernandez-Martinez et al. 2024):**[^6]

- Add Gaussian noise at levels of 0%, 10%, 20%, 30%, 40% of the bin signal.
- Track the posterior widths and coverage statistics as a function of noise level.
- Identify which of the 35 parameters remain constrainable at observationally realistic noise levels. Based on CAMELS-zoomGZ results, cosmological parameters (\(\Omega_m, H_0, \Omega_b\)) and certain astrophysical parameters (IMF slope, ASN2) should remain robust to ≥40% noise. Wind parameters (W1–W8) and BH parameters (BH1–BH5) are expected to degrade first.[^6]
- Compute the **information gain** per profile type (ratio of posterior volume to prior volume in each parameter direction) as a function of profile combination. This directly addresses the question: *which thermodynamic profiles are worth observing for constraining each parameter?*

**Radial cut tests:**
- Progressively remove outer radial bins from \(R_{200c}\) inward to \(0.1 R_{200c}\). Track which parameters lose constraining power soonest.
- Progressively remove inner bins from \(0.1 R_{200c}\) outward. The inner region tends to carry the most information for feedback parameters; quantify how much is lost.[^3]
- Reproduce the Hernandez-Martinez et al. (2024) Figure on radial cuts for the ANP emulator as a direct comparison.[^6]

***

### 3.8 Test Category 8: Comparison to Observational Data

**What it tests:** Whether the best-fit emulator model is consistent with real data from eROSITA, ACT, and SPT—the ultimate ground-truth test.

**Protocol:**

- **eROSITA comparison (X-ray profiles).** Following Shreeram et al. (2024/2025), stack eRASS observed X-ray surface brightness profiles in bins of stellar mass or halo mass, and compute the emulator likelihood at a grid of parameter values. The best-fit reduced \(\chi^2\) should be of order 1; a value of \(\chi^2_\mathrm{red} > 3\) indicates model failure (as found for SIMBA and Astrid in that study).[^4]
- **SZ comparison (ACT/SPT profiles).** Following Moser et al. (2022), stack ACT DR4 or SPT-3G profiles for CMASS-like galaxy samples and compute the emulator-predicted profile vs. observed profile. Check if the best-fit parameter combination is consistent with independent eROSITA constraints.[^3]
- **Cross-observable consistency.** Run joint inference on X-ray + SZ profiles simultaneously. Parameters constrained by both probes should agree within their combined posterior. Inconsistency is a sign of either systematic errors in the observations or a missing physical ingredient in the model.

***

### 3.9 Test Category 9: Cross-Validation and Consistency Stress Tests

**What it tests:** Robustness of the emulator and diagnostic integrity.

**Tests to run:**

- **CV set test (cosmic variance).** Evaluate the emulator on the CV set (multiple realizations at the fiducial cosmology). The predicted profile uncertainty should be comparable to the cosmic variance measured from the CV simulations. Systematic bias of the emulator mean relative to the CV mean across realizations is a red flag.

- **Corrupted input test.** Randomly permute the radial bin order of test profiles before feeding them as context to the ANP. The emulator should produce much worse predictions (higher RMSE, \(\hat{\sigma}_\mathrm{epi}\) inflation). This validates that the emulator is using spatial structure, not just aggregate statistics.[^6]

- **Label permutation test.** Randomly permute the parameter labels \(\theta\) relative to profiles in the training set and retrain. Performance should collapse to near-prior levels. If it does not, the architecture may be picking up spurious correlations.

- **Half-training-set robustness.** Train on 50% of the training data and compare the held-out test set RMSE and coverage metrics. The ratio of test-set RMSE at full vs. half training should reveal whether the model is in a data-limited regime or has reached capacity. For CAMELS-zoomGZ results, performance saturates around 30,000 emulated profiles when using all five profile types.[^6]

- **Ensemble disagreement test.** Train 5 independent ANP models with different random seeds. Compute pairwise KL divergences between their predictions on a shared test set. High variance across ensemble members indicates insufficient training data or architecture instability, and the ensemble mean is a more reliable predictor.[^14]

***

### 3.10 Test Category 10: Posterior Predictive Checks and End-to-End Validation

**What it tests:** The full inference pipeline, treating the emulator as a black box and asking whether the posterior is statistically consistent with the data.

**Protocol:**

- **Posterior predictive check (PPC).** After running inference on mock data and obtaining posterior samples \(\{\theta^{(s)}\}_{s=1}^S\), generate emulated profiles for each posterior sample and compare the distribution of predicted profiles against the mock data. If the data vector lies within the 68% predictive interval for most radial bins and profile types, the model is self-consistent.

- **Expected coverage (PP-plot) over many mock runs.** For each of the 35 parameters and 20–50 mock runs, compute the fraction of runs where \(\theta_k^\mathrm{true}\) lies within the \(p\)-credible interval, for \(p \in [0.1, 0.9]\). This is the global coverage diagnostic. Target: the PP-curve should lie on the diagonal to within the scatter expected from the finite number of mock runs.[^15]

- **Multi-fidelity consistency.** If direct simulation profiles are available at 10–20 parameter configurations (from CAMELS-zoomGZ or other zoom simulations), run inference with (a) the ANP emulator and (b) the direct simulation profiles as the forward model. Compare posteriors. The KL divergence between emulated and simulation-based posteriors should satisfy \(\mathcal{D}_\mathrm{KL} < 1\). This is the direct answer to the question "is the emulator good enough for inference?"[^1]

***

## 4. Summary Table of Tests and Pass/Fail Criteria

| Test | Category | Key Metric | Pass Criterion | Priority |
|---|---|---|---|---|
| RMSE per field vs. obs. noise | 1 | RMSE/σ_obs | ≤ 0.15 per bin | Critical |
| Outlier fraction | 1 | F_out(Δχ² > 0.2) | < 10% | Critical |
| Pearson r per parameter | 1 | r(θ_true, θ_pred) | ≥ 0.97 cosmo, ≥ 0.90 astro | Critical |
| Coverage curve | 2 | Empirical vs. nominal coverage | Diagonal within 5% | Critical |
| PIT histogram KS test | 2 | KS p-value | > 0.05 | High |
| Reduced χ² of residuals | 2 | χ²_red | 0.8–1.2 | High |
| TARP test | 2 | TARP coverage curve | Diagonal within 2σ | Critical |
| 1P monotonicity | 3 | Qualitative sign check | All pass | High |
| Derivative accuracy vs. sims | 3 | Derivative ratio | Within 20% | High |
| Non-degeneracy Bhattacharyya | 3 | Pairwise distance | > 1.0 for all pairs | Medium |
| Mock parameter recovery bias | 4 | b_k = |Δμ|/σ | < 0.5σ for all 35 params | Critical |
| Mock coverage fraction | 4 | 68% CI coverage | 60–75% | Critical |
| KL div. emulator vs. sim posterior | 4 | D_KL | < 1 nat | Critical |
| Fisher vs. posterior width ratio | 5 | σ_mock / σ_Fisher | 0.7–2.0 | High |
| Fisher condition number | 5 | κ(F) | < 10⁵ | Medium |
| Context size convergence | 6 | σ_epi vs. N_context | Saturates monotonically | High |
| OOD epistemic inflation | 6 | σ_epi beyond prior | Inflates > 50% | High |
| Noise robustness r at 40% noise | 7 | r(θ_cosmo) at 40% noise | > 0.80 for Ω_m, H_0 | High |
| eROSITA χ² best-fit | 8 | χ²_red vs. eRASS | ≈ 1 (< 3) | Critical |
| ACT/SPT cross-check | 8 | Posterior consistency | Overlap > 1σ | High |
| CV set bias | 9 | Mean emulator bias | < 0.05 × signal | High |
| Corrupted input degradation | 9 | RMSE ratio | > 2× nominal RMSE | Medium |
| Ensemble KL divergence | 9 | Pairwise KL across 5 models | < 0.5 nat | High |
| Posterior predictive check | 10 | 68% predictive interval | Contains data ≥ 60% of bins | Critical |
| Multi-fidelity posterior KL | 10 | D_KL(emulator || sim) | < 1 nat | Critical |

***

## 5. Implementation Roadmap

### Phase 1 (Immediate): Formalize Existing Tests
- Standardize the RMSE/σ_obs metric in `diagnostics.py` — add a `noise_model` argument that maps each profile type to an eROSITA/SO noise level.
- Add the TARP test as a new function in `diagnostics.py`. The `tarp` Python package (from Lemos et al. 2023) provides a clean API.[^8]
- Add a `reduced_chi_squared` function.
- Formalize the 1P sensitivity test as a standalone notebook with pass/fail checks.

### Phase 2 (Core): Inference-Level Validation
- Implement the mock parameter recovery test: generate 30 mock observations at random parameter configurations, run `dynesty` nested sampling with the ANP likelihood, compute emulator bias and coverage fraction for all 35 parameters.
- Add the Fisher matrix computation using emulator finite differences and compare to Moser+ 2022 as a cross-check.[^3]
- Compute the KL divergence bound using Eq. (32) from Bevins+ 2025 after mock recovery.[^1]

### Phase 3 (Advanced): Observational Comparison
- Compile stacked eROSITA X-ray profiles from eRASS public data (or existing stacks from Shreeram+ 2025) and run the emulator likelihood to obtain observational constraints.[^4]
- Compute joint X-ray + SZ posteriors.
- Report the tension between best-fit emulator parameters and the TNG fiducial, following the methodology of the eROSITA CAMELS paper.[^16]

***

## 6. Connection to Literature and Benchmarks

The proposed test suite draws directly from and can be validated against several key benchmarks:

- **Hernandez-Martinez et al. (2024):** The CAMELS-zoomGZ neural network study achieves r > 0.97 for cosmological and r > 0.90 for astrophysical parameters on noiseless stacked profiles. These are the target benchmarks for Test Categories 1 and 7. Reproducing their Table 2 results with the ANP emulator serves as an external cross-validation.[^6]

- **Moser et al. (2022):** Fisher forecasts for ASN1, AAGN1, ASN2, AAGN2 from SZ profiles provide a reference sensitivity analysis. The ANP's Fisher matrix from Test Category 5 should recover the ~7–31% constraints on those four parameters, and extend them to the full 35D space.[^3]

- **Bevins et al. (2025):** The KL divergence bound RMSE/σ_obs ≤ √(2/N_d) provides the theoretical minimum accuracy requirement. For N_d = 30 radial bins this gives RMSE ≤ 26% of the noise; the operational target of ≤ 15% provides a safety margin.[^1]

- **Lemos et al. (2023):** TARP is the state-of-the-art posterior calibration test, now used by the Learning the Universe collaboration and in SBI pipelines for 21-cm inference. Adding TARP to the cosmoANP test suite brings it to the same standard.[^17][^9][^8]

- **Shreeram et al. (2025):** X-ray CAMELS emulator comparison against eRASS demonstrates that a reduced χ² ≈ 0.83 is achievable for IllustrisTNG at the best-fit parameters, while SIMBA and Astrid fail (χ² = 4.0 and 3.0). This is the target for Test Category 8.[^4]

---

## References

1. [On the accuracy of posterior recovery with neural network emulators](https://academic.oup.com/mnras/article/544/1/375/8257485) - In this paper, we define an upper bound on the Kullback–Leibler (KL) divergence between the true and...

2. [On the accuracy of posterior recovery with neural network emulators](https://arxiv.org/html/2503.13263v2)

3. [The Circumgalactic Medium from the CAMELS Simulations: Forecasting Constraints on Feedback Processes from Future Sunyaev-Zeldovich Observations](https://ar5iv.labs.arxiv.org/html/2201.02708) - The cycle of baryons through the circumgalactic medium (CGM) is important to understand in the conte...

4. [X-Raying CAMELS: Constraining Baryonic Feedback in the ...](https://inspirehep.net/files/5801a1a7926de3c633fd35a125e4428a)

5. [Attention-based Neural Network Emulators for Multi-Probe Data Vectors Part III: Modeling The Next Generation Surveys](https://arxiv.org/html/2505.22574v2)

6. [[PDF] arXiv:2410.10942v1 [astro-ph.CO] 14 Oct 2024](https://arxiv.org/pdf/2410.10942.pdf) - generated a set of noisy profiles by adding random Gaus- sian noise to each bin, with noise levels o...

7. [Sampling-Based Accuracy Testing of Posterior Estimators for ...](https://proceedings.mlr.press/v202/lemos23a.html) - In this paper, we introduce "Tests of Accuracy with Random Points" (TARP) coverage testing as a meth...

8. [Sampling-Based Accuracy Testing of Posterior Estimators for ... - arXiv](https://arxiv.org/abs/2302.03026) - In this paper, we introduce `Tests of Accuracy with Random Points' (TARP) coverage testing as a meth...

9. [TARP | Learning the Universe](https://learning-the-universe.github.io/projects/Robustness_TARP/) - We have developed a new technique, Tests of Accuracy with Random Points (TARP), to estimate coverage...

10. [Sampling-Based Accuracy Testing of Posterior Estimators ...](https://icml.cc/virtual/2023/poster/24029)

11. [The CAMELS project: Expanding the galaxy formation model space with new
  ASTRID and 28-parameter TNG and SIMBA suites](https://arxiv.org/pdf/2304.02096.pdf) - We present CAMELS-ASTRID, the third suite of hydrodynamical simulations in
the Cosmology and Astroph...

12. [How to estimate Fisher information matrices from simulations - ar5iv](https://ar5iv.labs.arxiv.org/html/2305.08994) - The Fisher information matrix is a quantity of fundamental importance for information geometry and a...

13. [Interpolation of GEDI Biomass Estimates with Calibrated Uncertainty ...](https://arxiv.org/html/2601.16834v3) - Then, we show that Attentive Neural Processes achieve parity or better accuracy while maintaining ne...

14. [[2507.13495] Simulation-based inference with deep ensembles - arXiv](https://arxiv.org/abs/2507.13495) - Abstract page for arXiv paper 2507.13495: Simulation-based inference with deep ensembles: Evaluating...

15. [[PDF] A Trust Crisis In Simulation-Based Inference? Your Posterior ...](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_6.pdf) - This work directly assesses the quality of credible regions through the notion of expected coverage,...

16. [X-raying CAMELS: Constraining Baryonic Feedback in the Circum-Galactic
  Medium with the CAMELS simulations and eRASS X-ray Observations](http://arxiv.org/pdf/2412.04559.pdf) - ...a crucial role
in regulating star formation and feedback. Using the CAMELS simulation suite,
we d...

17. [Simulation based inference of the ionization history from the 2D 21 cm power spectrum](https://www.semanticscholar.org/paper/e9cb51feef785d5c00ad7522b688fdeb6640c6e3) - The 21 cm signal contains a wealth of information about the formation of the first stars and the rei...

