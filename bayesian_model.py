"""
bayesian_model.py
=================
Layer 2 – Bayesian Hierarchical Response Model

Model overview
--------------
The binary endpoint is Objective Response (OR = CR + PR under RECIST 1.1).
For each arm a ∈ {treatment, control} we model:

    yₐ | θₐ  ~  Binomial(nₐ, θₐ)         (likelihood)
    θₐ        ~  Beta(αₐ, βₐ)             (prior)

giving a conjugate posterior:

    θₐ | yₐ   ~  Beta(αₐ + yₐ,  βₐ + nₐ − yₐ)

This is the closed-form (analytical) path, which runs without PyMC and is
used for interim updates at each adaptive look.

For the full posterior over the treatment effect Δ = θ_T − θ_C (or the
odds ratio), we use PyMC with NUTS (No-U-Turn Sampler, a variant of HMC).
The NUTS path produces:
    - Full posterior traces for θ_T, θ_C, Δ, OR
    - Posterior predictive checks
    - ArviZ diagnostics (R-hat, ESS, MCSE)

Prior specification
-------------------
We use a weakly informative Beta prior anchored to historical control data.
A reasonable historical control ORR of ~15% (consistent with Layer 1
calibration) gives:

    θ_C ~ Beta(3, 17)    E[θ_C] = 0.15,  effective sample size = 20
    θ_T ~ Beta(2, 8)     E[θ_T] = 0.20,  weak optimism, ESS = 10

The treatment prior is deliberately vaguer — we know less about the new drug.
Prior sensitivity is analysed via three prior regimes (sceptical, neutral,
optimistic) whose impact on posterior decisions is quantified in the output.

Decision thresholds (used by Layer 3)
--------------------------------------
At each interim look the model computes:

    P(θ_T > θ_C | data)   — posterior probability of superiority
    P(θ_T > θ_C + MID)    — probability of clinically meaningful difference
                             where MID = minimum important difference (default 0.10)

Stopping rules (enforced by Layer 3, computed here):
    Efficacy  : P(θ_T > θ_C | data) > η_E   (default 0.975)
    Futility  : P(θ_T > θ_C | data) < η_F   (default 0.10)

References
----------
[1] Berry SM et al. (2010). Bayesian Adaptive Methods for Clinical Trials.
    CRC Press.
[2] Thall PF, Simon R (1994). Practical Bayesian guidelines for Phase IIB
    clinical trials. Biometrics 50:337–349.
[3] Gelman A et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.
[4] Kumar R, Carroll C, Hartikainen A, Martin O (2019). ArviZ: A unified
    library for exploratory analysis of Bayesian models. JOSS 4(33):1143.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Literal

# PyMC / ArviZ are optional — imported lazily inside functions that need them
# so the module is importable even if they are not installed.


# ---------------------------------------------------------------------------
# Prior specification
# ---------------------------------------------------------------------------

@dataclass
class BetaPrior:
    """
    Beta(alpha, beta) prior for a response rate θ ∈ (0, 1).

    Attributes
    ----------
    alpha, beta : float
        Shape parameters.  E[θ] = alpha / (alpha + beta).
        Effective sample size (ESS) = alpha + beta.
    label : str
        Human-readable name for plots and output tables.
    """
    alpha: float
    beta:  float
    label: str = ""

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def ess(self) -> float:
        """Effective sample size of the prior."""
        return self.alpha + self.beta

    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta
        return np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))

    def __repr__(self) -> str:
        return (f"Beta({self.alpha:.1f}, {self.beta:.1f})  "
                f"[E={self.mean:.2f}, ESS={self.ess:.0f}]  {self.label}")


# Three prior regimes for sensitivity analysis
PRIORS: dict[str, dict[str, BetaPrior]] = {
    "sceptical": {
        "treatment": BetaPrior(2, 18, "sceptical treatment"),   # E=0.10
        "control":   BetaPrior(3, 17, "sceptical control"),     # E=0.15
    },
    "neutral": {
        "treatment": BetaPrior(2, 8,  "neutral treatment"),     # E=0.20
        "control":   BetaPrior(3, 17, "neutral control"),       # E=0.15
    },
    "optimistic": {
        "treatment": BetaPrior(5, 10, "optimistic treatment"),  # E=0.33
        "control":   BetaPrior(3, 17, "optimistic control"),    # E=0.15
    },
}


# ---------------------------------------------------------------------------
# Conjugate (analytical) posterior — used for interim adaptive updates
# ---------------------------------------------------------------------------

@dataclass
class ConjugatePosterior:
    """
    Closed-form Beta posterior after observing y responses in n patients.

    posterior: θ | y ~ Beta(α + y, β + n − y)
    """
    prior:        BetaPrior
    n_observed:   int
    y_observed:   int          # number of responders

    @property
    def alpha_post(self) -> float:
        return self.prior.alpha + self.y_observed

    @property
    def beta_post(self) -> float:
        return self.prior.beta + (self.n_observed - self.y_observed)

    @property
    def posterior_mean(self) -> float:
        return self.alpha_post / (self.alpha_post + self.beta_post)

    @property
    def posterior_std(self) -> float:
        a, b = self.alpha_post, self.beta_post
        return np.sqrt(a * b / ((a + b)**2 * (a + b + 1)))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1 - level) / 2
        dist = stats.beta(self.alpha_post, self.beta_post)
        return dist.ppf(lo), dist.ppf(1 - lo)

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw posterior samples (for Monte Carlo decision quantities)."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.beta(self.alpha_post, self.beta_post, size=size)

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (f"Beta({self.alpha_post:.1f}, {self.beta_post:.1f})  "
                f"mean={self.posterior_mean:.3f}  "
                f"95% CI=[{lo:.3f}, {hi:.3f}]  "
                f"(n={self.n_observed}, y={self.y_observed})")


# ---------------------------------------------------------------------------
# Interim analysis engine (conjugate path)
# ---------------------------------------------------------------------------

@dataclass
class DecisionThresholds:
    eta_efficacy:  float = 0.975   # stop for efficacy if P(θ_T > θ_C) > η_E
    eta_futility:  float = 0.100   # stop for futility if P(θ_T > θ_C) < η_F
    mid:           float = 0.100   # minimum important difference


def posterior_probability_superiority(
    post_treat: ConjugatePosterior,
    post_ctrl:  ConjugatePosterior,
    n_mc:       int = 50_000,
    rng:        np.random.Generator | None = None,
) -> float:
    """
    Estimate P(θ_T > θ_C | data) by Monte Carlo over conjugate posteriors.

    Using MC rather than the closed-form integral because:
    (a) it generalises to the PyMC path without code duplication, and
    (b) the closed-form integral requires a regularised incomplete beta
        function that adds complexity for marginal accuracy gain at n≥50k.
    """
    if rng is None:
        rng = np.random.default_rng()
    s_treat = post_treat.sample(n_mc, rng)
    s_ctrl  = post_ctrl.sample(n_mc, rng)
    return float(np.mean(s_treat > s_ctrl))


def posterior_probability_mid(
    post_treat: ConjugatePosterior,
    post_ctrl:  ConjugatePosterior,
    mid:        float = 0.10,
    n_mc:       int   = 50_000,
    rng:        np.random.Generator | None = None,
) -> float:
    """P(θ_T > θ_C + MID | data) — clinically meaningful superiority."""
    if rng is None:
        rng = np.random.default_rng()
    s_treat = post_treat.sample(n_mc, rng)
    s_ctrl  = post_ctrl.sample(n_mc, rng)
    return float(np.mean(s_treat > s_ctrl + mid))


@dataclass
class InterimResult:
    """
    Results from a single interim analysis.
    """
    look:               int
    n_treat:            int
    n_ctrl:             int
    y_treat:            int
    y_ctrl:             int
    post_treat:         ConjugatePosterior
    post_ctrl:          ConjugatePosterior
    p_superiority:      float
    p_mid:              float
    decision:           Literal["continue", "stop_efficacy", "stop_futility"]
    thresholds:         DecisionThresholds

    def summary_row(self) -> dict:
        lo_t, hi_t = self.post_treat.credible_interval()
        lo_c, hi_c = self.post_ctrl.credible_interval()
        return {
            "look":          self.look,
            "n_treat":       self.n_treat,
            "n_ctrl":        self.n_ctrl,
            "orr_treat":     self.y_treat / self.n_treat if self.n_treat else 0,
            "orr_ctrl":      self.y_ctrl  / self.n_ctrl  if self.n_ctrl  else 0,
            "post_mean_T":   self.post_treat.posterior_mean,
            "post_mean_C":   self.post_ctrl.posterior_mean,
            "ci95_T":        f"[{lo_t:.3f}, {hi_t:.3f}]",
            "ci95_C":        f"[{lo_c:.3f}, {hi_c:.3f}]",
            "p_superior":    self.p_superiority,
            "p_mid":         self.p_mid,
            "decision":      self.decision,
        }


def run_interim_analysis(
    look:         int,
    responses:    dict,                # cohort dict from Layer 1
    prior_regime: str = "neutral",
    thresholds:   DecisionThresholds | None = None,
    n_mc:         int = 50_000,
    rng:          np.random.Generator | None = None,
) -> InterimResult:
    """
    Run a single interim analysis using conjugate posteriors.

    Parameters
    ----------
    look : int
        Interim look number (1-indexed).
    responses : dict
        Output of `simulate_trial_cohort` from Layer 1, or a slice of it
        representing patients observed so far.
    prior_regime : str
        One of "sceptical", "neutral", "optimistic".
    thresholds : DecisionThresholds
        Stopping rule parameters.
    """
    if thresholds is None:
        thresholds = DecisionThresholds()
    if rng is None:
        rng = np.random.default_rng()

    priors = PRIORS[prior_regime]

    # Count responders (CR + PR = ORR)
    def _count(arm_key: str) -> tuple[int, int]:
        resp = responses[arm_key]["responses"]
        n    = len(resp)
        y    = int(np.sum((resp == "CR") | (resp == "PR")))
        return n, y

    n_t, y_t = _count("treatment")
    n_c, y_c = _count("control")

    post_treat = ConjugatePosterior(priors["treatment"], n_t, y_t)
    post_ctrl  = ConjugatePosterior(priors["control"],   n_c, y_c)

    p_sup = posterior_probability_superiority(post_treat, post_ctrl, n_mc, rng)
    p_mid = posterior_probability_mid(post_treat, post_ctrl, thresholds.mid, n_mc, rng)

    if p_sup > thresholds.eta_efficacy:
        decision = "stop_efficacy"
    elif p_sup < thresholds.eta_futility:
        decision = "stop_futility"
    else:
        decision = "continue"

    return InterimResult(
        look=look, n_treat=n_t, n_ctrl=n_c, y_treat=y_t, y_ctrl=y_c,
        post_treat=post_treat, post_ctrl=post_ctrl,
        p_superiority=p_sup, p_mid=p_mid,
        decision=decision, thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Full MCMC model via PyMC (NUTS sampler)
# ---------------------------------------------------------------------------

def build_pymc_model(
    y_treat:      int,
    n_treat:      int,
    y_ctrl:       int,
    n_ctrl:       int,
    prior_regime: str = "neutral",
):
    """
    Build a PyMC model for the two-arm Binomial-Beta problem.

    Model
    -----
        θ_T ~ Beta(α_T, β_T)
        θ_C ~ Beta(α_C, β_C)
        y_T ~ Binomial(n_T, θ_T)
        y_C ~ Binomial(n_C, θ_C)
        Δ   = θ_T − θ_C                 (derived quantity)
        OR  = [θ_T/(1−θ_T)] / [θ_C/(1−θ_C)]  (odds ratio)

    Returns
    -------
    model : pymc.Model
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "PyMC is required for the MCMC path.\n"
            "Install with:  pip install pymc"
        )

    priors = PRIORS[prior_regime]
    p_t = priors["treatment"]
    p_c = priors["control"]

    with pm.Model() as model:
        # Priors
        theta_T = pm.Beta("theta_T", alpha=p_t.alpha, beta=p_t.beta)
        theta_C = pm.Beta("theta_C", alpha=p_c.alpha, beta=p_c.beta)

        # Likelihood
        pm.Binomial("y_T", n=n_treat, p=theta_T, observed=y_treat)
        pm.Binomial("y_C", n=n_ctrl,  p=theta_C, observed=y_ctrl)

        # Derived quantities — tracked in the trace
        pm.Deterministic("delta",  theta_T - theta_C)
        pm.Deterministic(
            "odds_ratio",
            (theta_T / (1 - theta_T)) / (theta_C / (1 - theta_C))
        )

    return model


def run_mcmc(
    y_treat:      int,
    n_treat:      int,
    y_ctrl:       int,
    n_ctrl:       int,
    prior_regime: str  = "neutral",
    draws:        int  = 2000,
    tune:         int  = 1000,
    chains:       int  = 4,
    target_accept: float = 0.90,
    random_seed:  int  = 42,
) -> tuple:
    """
    Sample the posterior via NUTS and return (model, idata).

    Parameters
    ----------
    draws, tune, chains : int
        Standard PyMC sampling parameters.
        defaults are conservative for publication quality:
        4 chains × 2000 draws = 8000 posterior samples after warmup.
    target_accept : float
        NUTS step-size target acceptance rate.  0.90 is appropriate for
        this smooth Beta posterior; increase to 0.95 for more complex models.

    Returns
    -------
    model : pymc.Model
    idata : arviz.InferenceData
        Contains posterior, posterior_predictive, log_likelihood.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        raise ImportError(
            "PyMC and ArviZ are required.\n"
            "Install with:  pip install pymc arviz"
        )

    model = build_pymc_model(y_treat, n_treat, y_ctrl, n_ctrl, prior_regime)

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )
        # Posterior predictive for PPC
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    return model, idata


def mcmc_decision_quantities(idata) -> dict:
    """
    Extract key decision quantities from an ArviZ InferenceData object.

    Returns
    -------
    dict with:
        p_superiority   : P(θ_T > θ_C | data)
        p_mid           : P(Δ > 0.10 | data)
        delta_mean      : E[Δ | data]
        delta_hdi_95    : 95% HDI for Δ
        or_median       : median odds ratio
        or_hdi_95       : 95% HDI for odds ratio
        rhat_max        : max R-hat across all parameters (convergence)
        ess_min         : min bulk ESS across all parameters
    """
    import arviz as az
    import numpy as np

    post = idata.posterior

    theta_T = post["theta_T"].values.flatten()
    theta_C = post["theta_C"].values.flatten()
    delta   = post["delta"].values.flatten()
    OR      = post["odds_ratio"].values.flatten()

    p_sup = float(np.mean(theta_T > theta_C))
    p_mid = float(np.mean(delta > 0.10))

    delta_hdi = az.hdi(idata, var_names=["delta"],  hdi_prob=0.95)["delta"].values
    or_hdi    = az.hdi(idata, var_names=["odds_ratio"], hdi_prob=0.95)["odds_ratio"].values

    summary   = az.summary(idata, var_names=["theta_T", "theta_C", "delta", "odds_ratio"])
    rhat_max  = float(summary["r_hat"].max())
    ess_min   = float(summary["ess_bulk"].min())

    return {
        "p_superiority": p_sup,
        "p_mid":         p_mid,
        "delta_mean":    float(np.mean(delta)),
        "delta_hdi_95":  tuple(delta_hdi),
        "or_median":     float(np.median(OR)),
        "or_hdi_95":     tuple(or_hdi),
        "rhat_max":      rhat_max,
        "ess_min":        ess_min,
    }


# ---------------------------------------------------------------------------
# Prior sensitivity analysis
# ---------------------------------------------------------------------------

def prior_sensitivity_analysis(
    y_treat: int,
    n_treat: int,
    y_ctrl:  int,
    n_ctrl:  int,
    n_mc:    int = 100_000,
    rng:     np.random.Generator | None = None,
) -> list[dict]:
    """
    Run the conjugate model under all three prior regimes and return a
    table of posterior summaries for comparison.

    This is the analytical complement to the MCMC sensitivity analysis —
    it runs instantly and is used to motivate prior choice in the write-up.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    rows = []
    for regime in ("sceptical", "neutral", "optimistic"):
        priors = PRIORS[regime]
        post_t = ConjugatePosterior(priors["treatment"], n_treat, y_treat)
        post_c = ConjugatePosterior(priors["control"],   n_ctrl,  y_ctrl)

        p_sup = posterior_probability_superiority(post_t, post_c, n_mc, rng)
        p_mid = posterior_probability_mid(post_t, post_c, 0.10, n_mc, rng)

        lo_t, hi_t = post_t.credible_interval()
        lo_c, hi_c = post_c.credible_interval()

        rows.append({
            "prior_regime":  regime,
            "prior_T":       str(priors["treatment"]),
            "prior_C":       str(priors["control"]),
            "post_mean_T":   post_t.posterior_mean,
            "post_mean_C":   post_c.posterior_mean,
            "ci95_T":        (lo_t, hi_t),
            "ci95_C":        (lo_c, hi_c),
            "p_superiority": p_sup,
            "p_mid":         p_mid,
        })
    return rows


# ---------------------------------------------------------------------------
# Diagnostic plots (conjugate path — no PyMC required)
# ---------------------------------------------------------------------------

def plot_posterior_evolution(
    interim_results: list[InterimResult],
    figsize:         tuple = (14, 5),
    save_path:       str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot how the posterior distributions for θ_T and θ_C evolve across
    interim looks.  One panel per interim, prior overlaid in dashed line.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_looks = len(interim_results)
    fig, axes = plt.subplots(1, n_looks, figsize=figsize, sharey=True)
    if n_looks == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117")

    theta_grid = np.linspace(0, 1, 400)

    for ax, res in zip(axes, interim_results):
        ax.set_facecolor("#0d1117")

        # Treatment posterior
        post_t = stats.beta(res.post_treat.alpha_post, res.post_treat.beta_post)
        ax.fill_between(theta_grid, post_t.pdf(theta_grid),
                        alpha=0.30, color="#4fa3e0")
        ax.plot(theta_grid, post_t.pdf(theta_grid), color="#4fa3e0", lw=1.8,
                label=f"θ_T  μ={res.post_treat.posterior_mean:.2f}")

        # Control posterior
        post_c = stats.beta(res.post_ctrl.alpha_post, res.post_ctrl.beta_post)
        ax.fill_between(theta_grid, post_c.pdf(theta_grid),
                        alpha=0.25, color="#e07a4f")
        ax.plot(theta_grid, post_c.pdf(theta_grid), color="#e07a4f", lw=1.8,
                label=f"θ_C  μ={res.post_ctrl.posterior_mean:.2f}")

        # Prior overlay (treatment only, neutral regime for reference)
        prior_t = PRIORS["neutral"]["treatment"]
        prior_pdf = stats.beta(prior_t.alpha, prior_t.beta).pdf(theta_grid)
        ax.plot(theta_grid, prior_pdf, color="#4fa3e0", lw=0.8,
                ls="--", alpha=0.45, label="prior θ_T")

        # Annotation
        decision_colors = {
            "continue":       "#888",
            "stop_efficacy":  "#50d890",
            "stop_futility":  "#e05555",
        }
        dcol = decision_colors[res.decision]
        ax.set_title(
            f"Look {res.look}  (n={res.n_treat + res.n_ctrl})",
            color="white", fontsize=9, pad=6,
        )
        ax.text(0.97, 0.96,
                f"P(sup)={res.p_superiority:.3f}\n"
                f"P(MID)={res.p_mid:.3f}\n"
                f"{res.decision.replace('_', ' ')}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                color=dcol, linespacing=1.6)

        ax.set_xlabel("Response rate θ", color="#aaa", fontsize=8)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=7, framealpha=0.15, labelcolor="white")

    axes[0].set_ylabel("Posterior density", color="#aaa", fontsize=8)
    fig.suptitle(
        "Posterior Evolution — θ_T vs θ_C across Interim Looks",
        color="white", fontsize=11, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_prior_sensitivity(
    sensitivity_rows: list[dict],
    figsize:          tuple = (10, 4),
    save_path:        str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Bar chart comparing P(superiority) and P(MID) across prior regimes.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    regimes = [r["prior_regime"] for r in sensitivity_rows]
    p_sup   = [r["p_superiority"] for r in sensitivity_rows]
    p_mid   = [r["p_mid"]         for r in sensitivity_rows]
    colors  = ["#e05555", "#4fa3e0", "#50d890"]

    for ax, vals, title, threshold in zip(
        axes,
        [p_sup, p_mid],
        ["P(θ_T > θ_C | data)", "P(θ_T > θ_C + 0.10 | data)"],
        [0.975, 0.80],
    ):
        ax.set_facecolor("#0d1117")
        bars = ax.bar(regimes, vals, color=colors, width=0.5, linewidth=0)
        ax.axhline(threshold, color="white", lw=0.8, ls="--", alpha=0.5,
                   label=f"threshold = {threshold}")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#888", labelsize=8)
        ax.set_ylabel("Probability", color="#aaa", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=7, framealpha=0.15, labelcolor="white")

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=8)

    fig.suptitle("Prior Sensitivity Analysis", color="white", fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_mcmc_trace(
    idata,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    ArviZ trace plot for the four key parameters.
    Requires PyMC + ArviZ.
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError("pip install arviz")

    import matplotlib.pyplot as plt
    axes = az.plot_trace(
        idata,
        var_names=["theta_T", "theta_C", "delta", "odds_ratio"],
        figsize=(12, 7),
        combined=False,
    )
    fig = axes.ravel()[0].get_figure()
    fig.suptitle("MCMC Trace — NUTS sampler", fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_posterior_delta(
    idata,
    mid:       float = 0.10,
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Posterior distribution of Δ = θ_T − θ_C with HDI and MID annotated.
    Requires ArviZ.
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError("pip install arviz")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    delta_samples = idata.posterior["delta"].values.flatten()

    az.plot_posterior(
        idata,
        var_names=["delta"],
        hdi_prob=0.95,
        ax=ax,
        color="#4fa3e0",
    )
    ax.axvline(0,   color="#e05555", lw=1.0, ls="--", alpha=0.7, label="Δ = 0 (null)")
    ax.axvline(mid, color="#50d890", lw=1.0, ls="--", alpha=0.7, label=f"Δ = {mid} (MID)")

    p_pos = np.mean(delta_samples > 0)
    p_mid_val = np.mean(delta_samples > mid)
    ax.text(0.02, 0.92,
            f"P(Δ>0)   = {p_pos:.3f}\nP(Δ>{mid}) = {p_mid_val:.3f}",
            transform=ax.transAxes, fontsize=9,
            color="white", va="top", linespacing=1.8)

    ax.set_title("Posterior of treatment effect  Δ = θ_T − θ_C",
                 color="white", fontsize=10)
    ax.set_xlabel("Δ", color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(fontsize=8, framealpha=0.15, labelcolor="white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Entry point — smoke test (conjugate path only, no PyMC required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from sde_simulator import TumourSDEParams, TrialConfig, simulate_trial_cohort

    print("=" * 60)
    print("Layer 2 — Bayesian Response Model  (smoke test)")
    print("=" * 60)

    # --- Simulate a full cohort from Layer 1 ---
    params = TumourSDEParams()
    cfg    = TrialConfig(n_patients=200, seed=42)
    cohort = simulate_trial_cohort(cfg, params)

    t_resp = cohort["treatment"]["responses"]
    c_resp = cohort["control"]["responses"]
    n_t, y_t = len(t_resp), int(np.sum((t_resp == "CR") | (t_resp == "PR")))
    n_c, y_c = len(c_resp), int(np.sum((c_resp == "CR") | (c_resp == "PR")))
    print(f"\nObserved data: treat n={n_t} y={y_t}  ctrl n={n_c} y={y_c}")

    # --- Single interim analysis (conjugate) ---
    print("\n--- Interim analysis (conjugate, neutral prior) ---")
    result = run_interim_analysis(look=1, responses=cohort, prior_regime="neutral")
    for k, v in result.summary_row().items():
        print(f"  {k:<18}: {v}")

    # --- Prior sensitivity ---
    print("\n--- Prior sensitivity ---")
    rows = prior_sensitivity_analysis(y_t, n_t, y_c, n_c)
    for row in rows:
        print(f"  {row['prior_regime']:12s}  "
              f"P(sup)={row['p_superiority']:.3f}  "
              f"P(MID)={row['p_mid']:.3f}  "
              f"post_mean_T={row['post_mean_T']:.3f}")

    # --- Plots ---
    import matplotlib.pyplot as plt
    fig1 = plot_posterior_evolution([result], save_path=NONE)
    fig2 = plot_prior_sensitivity(rows,       save_path=NONE)
    plt.show()
    print("\nPlots saved.")

    # --- MCMC path (optional) ---
    try:
        import pymc
        print("\n--- MCMC path (PyMC found) ---")
        model, idata = run_mcmc(y_t, n_t, y_c, n_c, prior_regime="neutral",
                                draws=2000, tune=1000, chains=4)
        dq = mcmc_decision_quantities(idata)
        for k, v in dq.items():
            print(f"  {k:<18}: {v}")
        fig3 = plot_mcmc_trace(idata,       save_path=NONE)
        fig4 = plot_posterior_delta(idata,  save_path=NONE)
        plt.show()
    except ImportError:
        print("\nPyMC not found — MCMC path skipped.")
        print("Install with:  pip install pymc arviz")

    print("\nSmoke test passed.")
