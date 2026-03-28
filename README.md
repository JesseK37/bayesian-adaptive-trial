# Bayesian Adaptive Clinical Trial Simulation
### Stochastic Disease Progression · Sequential Bayesian Inference · Response-Adaptive Randomisation

> **Portfolio context.** This is the third project in a data science portfolio.
> Projects 1–2 cover causal inference (IV/2SLS) and MLOps (LightGBM on GCP).
> This project is the mathematical differentiator — research-level methodology over engineering.

---

## Abstract

We build a three-layer simulation framework for a Phase II oncology adaptive clinical trial. Tumour burden evolves as a Gompertz stochastic differential equation (SDE) with multiplicative noise, calibrated to published tumour-growth-inhibition (TGI) parameters. A Bayesian Beta-Binomial model with conjugate posteriors updates the posterior probability of treatment superiority at each interim look. Response-adaptive randomisation (RAR) via Thompson sampling shifts patient allocation toward the better-performing arm in real time. Operating characteristics — power, Type I error, expected sample size, and early stop rate — are estimated by Monte Carlo simulation over a grid of true treatment effects. The adaptive design assigns meaningfully more patients to the superior treatment arm while preserving statistical validity, at comparable power to a fixed 1:1 design.

---

## 1. Motivation

Standard randomised controlled trials fix the allocation ratio and sample size at design time. This is statistically clean but ethically inefficient: if interim data suggest one arm is clearly superior, a fixed design continues enrolling patients onto the inferior arm regardless. Adaptive designs address this by allowing the trial to learn and respond — stopping early when evidence is conclusive, and shifting allocation toward the better arm while the trial is ongoing.

Bayesian adaptive designs are increasingly prominent in oncology and rare disease trials (I-SPY2, BATTLE, GBM AGILE). The FDA issued dedicated guidance in 2019. Yet their operating characteristics are poorly understood by practitioners without a simulation-based grounding. This project builds that grounding from first principles.

The three-layer architecture separates concerns cleanly:

| Layer | Responsibility | Key method |
|---|---|---|
| 1 — Disease simulator | Generate patient tumour trajectories | Gompertz SDE, Milstein discretisation |
| 2 — Inference engine | Update beliefs about response rates | Beta-Binomial conjugate posterior, PyMC NUTS |
| 3 — Decision engine | Allocate patients, apply stopping rules | Thompson sampling RAR, Monte Carlo OC |

---

## 2. Statistical Model

### 2.1 Tumour dynamics (Layer 1)

Tumour volume $V(t)$ evolves under the Gompertz SDE:

$$dV(t) = \bigl[\alpha - \beta \ln V(t) - \delta\bigr]\, V(t)\, dt \;+\; \sigma\, V(t)\, dW(t)$$

where $\alpha$ is the intrinsic proliferation rate, $\beta$ is the Gompertz deceleration coefficient capturing vascular and nutrient limitations, $\delta$ is the drug-induced elimination rate (zero in the control arm), $\sigma$ is the diffusion coefficient encoding biological noise, and $W(t)$ is standard Brownian motion.

The multiplicative noise term $\sigma V(t)\, dW(t)$ is the biologically motivated choice: it preserves $V(t) > 0$ for all $t$ and implies proportional variability — a 50 cm³ tumour fluctuates more in absolute terms than a 5 cm³ tumour, but with the same coefficient of variation. This is consistent with the TGI model class validated by Stein et al. (2011) against RECIST clinical data.

**Discretisation.** The Milstein scheme is used in preference to plain Euler–Maruyama. For multiplicative noise $g(V) = \sigma V$ the correction term is $\frac{1}{2}\sigma^2 V\bigl(\Delta W^2 - \Delta t\bigr)$, which improves strong convergence from order $\tfrac{1}{2}$ to order $1$. With step size $\Delta t = 0.5$ days the two schemes agree closely for this parameter regime; Milstein is the default for publication-quality results.

**Baseline heterogeneity.** Patient baseline volumes $V_0$ are drawn from a log-normal distribution with mean 25 cm³ and CV 0.50, reflecting the wide variation in tumour burden at trial entry in an unselected Phase II population.

**RECIST classification.** At the primary assessment time $T_\text{assess} = 84$ days (12 weeks), each patient is classified by percentage change from baseline:

| Category | Threshold | Interpretation |
|---|---|---|
| CR — complete response | $V / V_0 \leq 0\%$ | Tumour below detection |
| PR — partial response | $-30\% \leq \Delta \leq 0\%$ | Meaningful shrinkage |
| SD — stable disease | $0\% < \Delta < +20\%$ | Neither growing nor shrinking |
| PD — progressive disease | $\Delta \geq +20\%$ | Tumour growing |

The binary endpoint for the Bayesian model is objective response rate (ORR = CR + PR).

**Calibrated parameters.** The default parameters are anchored to Stein et al. (2011):

| Parameter | Value | Interpretation |
|---|---|---|
| $\alpha$ | 0.012 day⁻¹ | Intrinsic growth rate |
| $\beta$ | 0.0018 day⁻¹ | Gompertz deceleration |
| $\sigma$ | 0.06 day⁻½ | Biological noise |
| $\delta_\text{treatment}$ | 0.005–0.011 day⁻¹ | Drug elimination (grid) |
| $\delta_\text{control}$ | 0.001 day⁻¹ | Residual decay (BSC) |

At $\delta = 0.009$ the model produces treatment ORR ≈ 44% vs control ORR ≈ 14%, consistent with a positive Phase II result in a solid tumour setting.

---

### 2.2 Bayesian response model (Layer 2)

For each arm $a \in \{\text{treatment, control}\}$, the response rate $\theta_a$ is inferred from the observed responders $y_a$ out of $n_a$ patients:

$$y_a \mid \theta_a \;\sim\; \text{Binomial}(n_a,\, \theta_a)$$
$$\theta_a \;\sim\; \text{Beta}(\alpha_a,\, \beta_a)$$

yielding the conjugate posterior:

$$\theta_a \mid y_a \;\sim\; \text{Beta}\!\bigl(\alpha_a + y_a,\;\; \beta_a + n_a - y_a\bigr)$$

The treatment effect is $\Delta = \theta_T - \theta_C$. The key decision quantity at each interim look is:

$$p_\text{sup} = P(\theta_T > \theta_C \mid \text{data}) = \int_0^1 F_C(\theta)\, p_T(\theta)\, d\theta$$

estimated by Monte Carlo over posterior samples (50,000 draws per evaluation).

**Two inference paths.** The conjugate path is used inside the Layer 3 Monte Carlo loop — it evaluates in microseconds, which matters when running 5,000 trial replications. The full PyMC path with NUTS sampling (4 chains × 2,000 draws, target acceptance 0.90) is reserved for the final analysis of a single trial realisation, where full posterior diagnostics (R-hat, ESS, posterior predictive checks) are required.

**Prior specification.** Three prior regimes encode different states of prior knowledge:

| Regime | $\theta_T$ prior | $\theta_C$ prior | E[$\theta_T$] | ESS |
|---|---|---|---|---|
| Sceptical | Beta(2, 18) | Beta(3, 17) | 0.10 | 20 |
| Neutral | Beta(2, 8) | Beta(3, 17) | 0.20 | 10 |
| Optimistic | Beta(5, 10) | Beta(3, 17) | 0.33 | 15 |

The control prior is fixed across all regimes, anchored to the historical control ORR of 15%. The treatment prior is deliberately vaguer — less is known about the investigational drug. Prior sensitivity is most visible at early interim looks (n ≈ 20–30 per arm) where the effective sample size of the prior is non-negligible relative to the data. At the final analysis the data dominate all three regimes.

---

### 2.3 Adaptive design and stopping rules (Layer 3)

**Response-adaptive randomisation.** After each interim look the allocation probability for the next cohort is updated via Thompson sampling:

$$\rho_{k+1} = \text{clip}\!\Bigl(P(\theta_T > \theta_C \mid \text{data}_{1:k}),\; \rho_\min,\; \rho_\max\Bigr)$$

with $[\rho_\min, \rho_\max] = [0.20, 0.80]$. The clip bounds follow the recommendation of Thall and Wathen (2007) and ensure both arms retain sufficient patients for valid inference. Without clipping, a strong early signal would allocate almost all subsequent patients to treatment, destroying the statistical comparison.

Thompson sampling is a multi-armed bandit rule that naturally balances exploration and exploitation. Here it means: if the posterior strongly favours treatment, allocate more patients there — but never so many that the control arm becomes uninformative.

**Stopping rules.** At each look the trial stops if:

$$\text{Efficacy:} \quad p_\text{sup} > \eta_E = 0.975$$
$$\text{Futility:} \quad p_\text{sup} < \eta_F = 0.10$$

These thresholds are consistent with Thall and Simon (1994) guidelines for Phase IIB Bayesian designs. The trial proceeds to the final analysis if neither rule fires at any interim.

**Interim schedule.** With $n_\text{total} = 120$ and three interims, looks occur at $n \approx 30, 60, 90, 120$ patients (total enrolled). This is a relatively aggressive interim schedule — it maximises the opportunity for early stopping but increases the multiplicity burden, reflected in the empirical Type I error reported below.

---

## 3. Operating Characteristics

OC surfaces are estimated over a grid of true treatment effects $\delta \in \{0.001, 0.003, 0.005, 0.007, 0.009, 0.011\}$ with 3,500 Monte Carlo replications per point.

### 3.1 Power and Type I error

| $\delta$ | ORR (treat) | ORR (ctrl) | Power (adaptive) | Power (fixed) | Type I proxy |
|---|---|---|---|---|---|
| 0.001 | ≈ 15% | ≈ 14% | 0.004 | 0.024 | ✓ (near null) |
| 0.003 | ≈ 20% | ≈ 14% | 0.160 | 0.196 | — |
| 0.005 | ≈ 28% | ≈ 14% | 0.648 | 0.694 | — |
| 0.007 | ≈ 36% | ≈ 14% | 0.962 | 0.970 | — |
| 0.009 | ≈ 44% | ≈ 14% | 0.996 | 1.000 | — |
| 0.011 | ≈ 52% | ≈ 14% | 1.000 | 1.000 | — |

At the near-null point ($\delta = 0.001$, ORR difference ≈ 1%) the empirical Type I error is 0.004–0.024, well below the nominal $\eta_E = 0.975$ threshold. The power curves are monotone and approach 1.0 rapidly — consistent with the 14% vs 44% ORR separation being a large effect for a trial of this size.

### 3.2 Expected sample size

The adaptive design's primary benefit over fixed 1:1 is ethical rather than statistical: RAR shifts patients toward the superior arm during the trial.

| $\delta$ | E[N] adaptive | E[N] fixed | E[$N_\text{treat}$] adaptive | E[$N_\text{treat}$] fixed |
|---|---|---|---|---|
| 0.005 | 88.1 | 89.5 | ~59 | ~45 |
| 0.007 | 58.3 | 59.0 | ~47 | ~30 |
| 0.009 | 39.6 | 40.6 | ~35 | ~20 |

Total expected sample sizes are similar (RAR is not a sample-size reduction tool). The treatment arm allocation is meaningfully higher under RAR — at $\delta = 0.005$, approximately 67% of enrolled patients receive the superior treatment vs 50% under fixed allocation. This is the clinical argument for adaptive designs in oncology where the treatment arm offers the chance of meaningful tumour response.

---

## 4. Results

**Power curve** rises from near-zero at the null (validating the Type I error control) through a steep transition around $\delta = 0.005$ to near-perfect power at $\delta \geq 0.009$. The adaptive and fixed designs are almost identical in power — consistent with the theoretical result that RAR trades a small power cost for ethical benefits.

**Thompson sampling dynamics.** The allocation trace shows $\rho$ rising quickly toward the 0.80 clip for strong treatment effects, then remaining pinned. For weaker effects ($\delta = 0.003$–$0.005$) the allocation drifts upward more gradually across looks, reflecting genuine posterior uncertainty. The clip bound is reached in the majority of trials by look 2 when $\delta \geq 0.007$.

**Sample size distribution** is discrete, with mass concentrated at the four look exit points (n ≈ 30, 60, 90, 120). At $\delta = 0.005$ roughly 50% of trials stop early; at $\delta = 0.009$ over 99% stop at or before look 2. The fixed design distribution is near-identical in shape, confirming that stopping behaviour is driven by the posterior probability threshold rather than the allocation mechanism.

**Prior sensitivity.** At the final analysis (n = 60 per arm) the three prior regimes produce near-identical posteriors — the data overwhelm priors with ESS 10–20 against 60–120 observations. Sensitivity is most visible at look 1 (n ≈ 15 per arm): the sceptical prior can suppress $p_\text{sup}$ below $\eta_E$ even when the data weakly favour treatment, delaying efficacy stops by one look. This represents approximately 20–30 additional patients in expectation — a quantifiable cost of prior scepticism.

---

## 5. Discussion

**On the choice of Gompertz over logistic growth.** The Gompertz model has a steeper early deceleration than logistic: tumours slow down sooner relative to their carrying capacity. Empirically this fits observed clinical RECIST trajectories better (Stein et al., 2011; Claret et al., 2013). The Gompertz carrying capacity under the calibrated parameters is $\exp(\alpha/\beta) \approx 790\text{ cm}^3$ — a tumour that would be fatal untreated, which is the correct regime for a solid tumour Phase II.

**On Thompson sampling versus other RAR rules.** Several RAR rules appear in the literature: doubly-adaptive biased coin (Hu and Zhang, 2004), optimal allocation targeting minimum variance, and fixed power-boosting rules. Thompson sampling is chosen here for three reasons: it has a principled Bayesian derivation (the allocation probability *is* the posterior probability of superiority), it requires no additional tuning parameters beyond the clip bounds, and its operating characteristics are well-characterised in the recent adaptive trial literature.

**On the conjugate vs MCMC path.** The Beta-Binomial model has an exact conjugate posterior — MCMC is not needed for inference. The PyMC NUTS path is included for two reasons. First, it produces ArviZ diagnostics (R-hat, ESS, MCSE) that serve as a correctness check on the conjugate implementation. Second, for extensions to hierarchical models (e.g., modelling patient subgroups or borrowing strength across historical cohorts) the conjugate form no longer applies and MCMC becomes necessary. The two-path architecture makes that extension straightforward.

**Limitations.**
- The SDE parameters are fixed across patients. A natural extension is a hierarchical SDE where $\alpha$, $\beta$, $\delta$ themselves have patient-level distributions — this is the PK/PD modelling direction.
- RECIST response is measured at a single fixed time point. Longitudinal modelling of the full trajectory (as in the Claret et al. landmark model) would allow earlier and more powerful interim analyses.
- The simulation does not model dropout, delayed assessment, or non-compliance — standard simplifications for a Phase II design study.
- Operating characteristics are estimated at fixed $n_\text{sim} = 3500$. Monte Carlo standard errors on power estimates are approximately $\sqrt{p(1-p)/3500} \approx 0.008$ at $p = 0.5$.

---

## 6. Repository Structure

```
bayesian-adaptive-trial/
│
├── sde_simulator.py        # Layer 1: Gompertz SDE, patient simulation, RECIST
├── bayesian_model.py       # Layer 2: Beta-Binomial posterior, PyMC NUTS, priors
├── adaptive_trial.py       # Layer 3: Thompson sampling RAR, OC simulation
│
├── figures/
│   ├── trajectories.png        # Tumour volume trajectories, both arms
│   ├── waterfall.png           # RECIST waterfall, % change from baseline
│   ├── post_evol.png           # Posterior Beta densities across interim looks
│   ├── prior_sens.png          # P(superiority) across prior regimes
│   ├── oc_surface.png          # Power / E[N] / early stop rate curves
│   ├── allocation_drift.png    # Thompson sampling allocation trace
│   └── sample_size_dist.png    # N enrolled distribution, adaptive vs fixed
│
└── README.md
```

---

## 7. Reproduction

```bash
pip install numpy scipy matplotlib pymc arviz

# Layer 1 — disease simulator
python sde_simulator.py

# Layer 2 — Bayesian inference (conjugate + PyMC NUTS)
python bayesian_model.py

# Layer 3 — adaptive trial OC surface (increase n_sim for production)
python adaptive_trial.py
```

For a faster smoke test set `n_sim=100` in `adaptive_trial.py`. For publication-quality OC curves use `n_sim=5000` and `n_jobs=-1` to parallelise across all available cores.

---

## 8. References

1. Stein WD et al. (2011). Tumor regression and growth rates determined in five intramural NCI prostate cancer trials. *J Clin Oncol* 29:2090–2096.
2. Claret L et al. (2013). Model-based prediction of phase III overall survival in colorectal cancer. *J Clin Oncol* 31:2198–2204.
3. Berry SM et al. (2010). *Bayesian Adaptive Methods for Clinical Trials*. CRC Press.
4. Thall PF, Simon R (1994). Practical Bayesian guidelines for Phase IIB clinical trials. *Biometrics* 50:337–349.
5. Thall PF, Wathen JK (2007). Practical Bayesian adaptive randomisation in clinical trials. *Eur J Cancer* 43:859–866.
6. FDA (2019). Adaptive Designs for Clinical Trials of Drugs and Biologics: Guidance for Industry. US FDA.
7. Kloeden PE, Platen E (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
8. Kumar R et al. (2019). ArviZ: A unified library for exploratory analysis of Bayesian models. *JOSS* 4(33):1143.

---

## 8. Author
Jesse Koivu — Pure mathematics PhD, transitioning to data science. This project is part of a data science portfolio. The project demonstrates Stochastic Simulation, Bayesian Modeling and applications to a real world problem.

[LinkedIn] · [GitHub]
