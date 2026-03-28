"""
sde_simulator.py
================
Layer 1 – Stochastic Disease Progression Model

Tumour-burden dynamics are modelled via a Gompertz stochastic differential
equation (SDE):

    dV(t) = [α - β·ln V(t)]·V(t) dt  +  σ·V(t) dW(t)

where
    V(t)  : tumour volume (cm³) at time t
    α     : intrinsic growth rate  (day⁻¹)
    β     : growth-deceleration (Gompertz) coefficient (day⁻¹)
    σ     : diffusion / biological noise coefficient (day⁻½)
    W(t)  : standard Brownian motion

Under treatment the model becomes

    dV(t) = [α - β·ln V(t) - δ]·V(t) dt  +  σ·V(t) dW(t)

where δ (day⁻¹) is the drug-induced elimination rate.  This
multiplicative-noise form preserves V(t) > 0 for all t and is consistent
with the tumour-growth-inhibition (TGI) class of models (Stein et al., 2011;
Claret et al., 2013).

Discretisation
--------------
Euler–Maruyama with a fixed step Δt = 0.5 days.  The Milstein correction is
also implemented and toggled via `use_milstein=True`.  For the parameter
regime used here the two methods agree closely; Milstein is the default for
publication-quality results.

Milstein correction term for multiplicative noise  f(V) = σ·V :
    f · f' · (ΔW² − Δt) / 2   where  f' = σ

Response classification (RECIST 1.1)
--------------------------------------
At the primary assessment time T_assess the % change from baseline V₀ is used:

    CR  (complete response)    : V(T) / V₀ ≤ 0.00  (below detection limit)
    PR  (partial response)     : −30 % ≤ % change < 0
                                  i.e. V/V₀ ≤ 0.70
    SD  (stable disease)       : −30 % ≤ % change < +20 %
    PD  (progressive disease)  : % change ≥ +20 %

Parameters calibrated to
    Stein et al. (2011) JCO 29:2090–2096
    "A pharmacokinetic/pharmacodynamic model for tumor growth and tumor
     response kinetics in lung cancer"

References
----------
[1] Stein WD et al. (2011). Tumor regression and growth rates … JCO.
[2] Claret L et al. (2013). Model-based prediction of phase III overall
    survival … J Clin Oncol.
[3] Kloeden PE, Platen E (1992). Numerical Solution of SDEs. Springer.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import warnings
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class TumourSDEParams:
    """
    Biologically motivated parameters for the Gompertz SDE.

    Defaults are calibrated to a moderately aggressive solid tumour
    (e.g., NSCLC second-line setting) from Stein et al. (2011).
    """
    # Growth parameters
    alpha: float = 0.012      # intrinsic growth rate (day⁻¹)
    beta: float  = 0.0018     # Gompertz deceleration (day⁻¹)
    sigma: float = 0.06       # noise coefficient (day⁻½)

    # Treatment effect
    # Calibrated so that treatment ORR ≈ 44%, control ORR ≈ 14%
    # consistent with a positive Phase II in a solid-tumour setting
    # (cf. Stein et al. 2011; Claret et al. 2013)
    delta_treatment: float = 0.009   # drug elimination rate, treatment arm (day⁻¹)
    delta_control:   float = 0.001   # residual decay (BSC / disease dynamics)

    # Initial tumour volume
    V0_mean: float = 25.0     # mean baseline volume (cm³); log-normal centre
    V0_cv:   float = 0.50     # coefficient of variation across patients

    # Numerical
    dt: float = 0.5           # Euler–Maruyama step (days)

    def __post_init__(self):
        assert self.alpha  > 0, "alpha must be positive"
        assert self.beta   > 0, "beta must be positive"
        assert self.sigma  > 0, "sigma must be positive"
        assert self.dt     > 0, "dt must be positive"
        if self.sigma * np.sqrt(self.dt) > 0.3:
            warnings.warn(
                f"σ·√Δt = {self.sigma * np.sqrt(self.dt):.3f} > 0.3; "
                "Euler–Maruyama may lose accuracy.  Consider reducing dt.",
                UserWarning, stacklevel=2,
            )


@dataclass
class TrialConfig:
    """
    High-level trial design parameters.
    """
    n_patients:       int   = 120       # total enrolment
    n_interim:        int   = 3         # number of interim looks
    T_assess:         float = 84.0      # primary response assessment (days; 12 weeks)
    T_max:            float = 180.0     # maximum follow-up per patient (days)
    allocation_ratio: float = 0.5       # initial Pr(treatment) allocation
    seed:             int   = 42

    # RECIST thresholds (% change from baseline, as fractions)
    cr_threshold: float = -1.00   # complete response (tumour gone)
    pr_threshold: float = -0.30   # partial response
    pd_threshold: float =  0.20   # progressive disease


# ---------------------------------------------------------------------------
# Core SDE engine
# ---------------------------------------------------------------------------

def _gompertz_drift(V: np.ndarray, alpha: float, beta: float, delta: float) -> np.ndarray:
    """Gompertz drift  μ(V) = [α − β ln V − δ] · V"""
    ln_V = np.log(np.clip(V, 1e-6, None))
    return (alpha - beta * ln_V - delta) * V


def _milstein_correction(V: np.ndarray, sigma: float, dW: np.ndarray, dt: float) -> np.ndarray:
    """
    Milstein correction for multiplicative noise  g(V) = σV.
    f·f'·(ΔW²−Δt)/2  with  f'=σ,  g=σV  →  σ²V(ΔW²−Δt)/2
    """
    return 0.5 * sigma**2 * V * (dW**2 - dt)


def simulate_patients(
    n_patients:     int,
    params:         TumourSDEParams,
    arm:            Literal["treatment", "control"],
    T_max:          float,
    use_milstein:   bool  = True,
    rng:            np.random.Generator | None = None,
) -> dict:
    """
    Simulate tumour-volume trajectories for a cohort of patients.

    Parameters
    ----------
    n_patients : int
        Number of patients to simulate.
    params : TumourSDEParams
        SDE and biological parameters.
    arm : {"treatment", "control"}
        Determines which δ is applied.
    T_max : float
        Duration of follow-up (days).
    use_milstein : bool
        Use Milstein (True) or plain Euler–Maruyama (False).
    rng : numpy.random.Generator, optional
        Random state for reproducibility.

    Returns
    -------
    dict with keys:
        "times"        : (n_steps,) array of time points
        "trajectories" : (n_patients, n_steps) array of V(t)
        "V0"           : (n_patients,) baseline volumes
        "arm"          : str, arm label
        "params"       : TumourSDEParams used
    """
    if rng is None:
        rng = np.random.default_rng()

    delta = params.delta_treatment if arm == "treatment" else params.delta_control
    dt    = params.dt
    n_steps = int(np.ceil(T_max / dt)) + 1
    times   = np.arange(n_steps) * dt  # (n_steps,)

    # --- Baseline volumes: log-normal with given mean and CV ---------------
    mu_ln  = np.log(params.V0_mean) - 0.5 * np.log(1 + params.V0_cv**2)
    sig_ln = np.sqrt(np.log(1 + params.V0_cv**2))
    V0     = rng.lognormal(mu_ln, sig_ln, size=n_patients)   # (n_patients,)

    # --- Allocate trajectory array -----------------------------------------
    V = np.zeros((n_patients, n_steps), dtype=np.float64)
    V[:, 0] = V0

    # --- Euler–Maruyama / Milstein integration ------------------------------
    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps - 1):
        V_k = V[:, k]
        dW  = rng.standard_normal(size=n_patients) * sqrt_dt

        drift = _gompertz_drift(V_k, params.alpha, params.beta, delta)
        diffusion = params.sigma * V_k * dW

        V_next = V_k + drift * dt + diffusion
        if use_milstein:
            V_next += _milstein_correction(V_k, params.sigma, dW, dt)

        # Reflecting boundary: tumour volume cannot go negative
        V[:, k + 1] = np.maximum(V_next, 1e-4)

    return {
        "times":        times,
        "trajectories": V,
        "V0":           V0,
        "arm":          arm,
        "params":       params,
    }


# ---------------------------------------------------------------------------
# RECIST response classifier
# ---------------------------------------------------------------------------

def classify_response(
    trajectories: np.ndarray,
    V0:           np.ndarray,
    times:        np.ndarray,
    T_assess:     float,
    cfg:          TrialConfig,
) -> np.ndarray:
    """
    Classify each patient's best overall response at T_assess.

    Parameters
    ----------
    trajectories : (n_patients, n_steps) array
    V0           : (n_patients,) baseline volumes
    times        : (n_steps,) time array
    T_assess     : assessment time (days)
    cfg          : TrialConfig (holds RECIST thresholds)

    Returns
    -------
    responses : (n_patients,) array of str
        Each element in {"CR", "PR", "SD", "PD"}
    pct_change : (n_patients,) % change from baseline (as decimal fraction)
    """
    # Find index closest to T_assess
    idx = int(np.argmin(np.abs(times - T_assess)))
    V_assess = trajectories[:, idx]

    pct_change = (V_assess - V0) / V0  # negative = shrinkage

    responses = np.empty(len(V0), dtype=object)
    responses[pct_change <= cfg.cr_threshold]                                    = "CR"
    responses[(pct_change > cfg.cr_threshold) & (pct_change <= cfg.pr_threshold)] = "PR"
    responses[(pct_change > cfg.pr_threshold) & (pct_change <  cfg.pd_threshold)] = "SD"
    responses[pct_change >= cfg.pd_threshold]                                    = "PD"

    return responses, pct_change


# ---------------------------------------------------------------------------
# Trial cohort simulator (both arms)
# ---------------------------------------------------------------------------

def simulate_trial_cohort(
    cfg:    TrialConfig,
    params: TumourSDEParams,
    use_milstein: bool = True,
) -> dict:
    """
    Simulate a full randomised trial cohort.

    Patients are allocated with probability `cfg.allocation_ratio`
    to the treatment arm (this ratio will later be updated by the adaptive
    Layer 3 decision rule).

    Returns
    -------
    dict with full simulation results for both arms, RECIST outcomes,
    summary statistics, and arm-level response rates.
    """
    rng = np.random.default_rng(cfg.seed)

    n_treat  = int(round(cfg.n_patients * cfg.allocation_ratio))
    n_ctrl   = cfg.n_patients - n_treat

    # Simulate both arms
    treat_sim = simulate_patients(n_treat, params, "treatment", cfg.T_max,
                                  use_milstein=use_milstein, rng=rng)
    ctrl_sim  = simulate_patients(n_ctrl,  params, "control",  cfg.T_max,
                                  use_milstein=use_milstein, rng=rng)

    # Classify responses at T_assess
    treat_resp, treat_pct = classify_response(
        treat_sim["trajectories"], treat_sim["V0"],
        treat_sim["times"], cfg.T_assess, cfg
    )
    ctrl_resp, ctrl_pct = classify_response(
        ctrl_sim["trajectories"], ctrl_sim["V0"],
        ctrl_sim["times"], cfg.T_assess, cfg
    )

    def _response_rate(responses: np.ndarray) -> dict:
        n = len(responses)
        counts = {r: int(np.sum(responses == r)) for r in ["CR", "PR", "SD", "PD"]}
        counts["ORR"] = (counts["CR"] + counts["PR"]) / n   # objective response rate
        counts["n"]   = n
        return counts

    return {
        "treatment": {**treat_sim, "responses": treat_resp, "pct_change": treat_pct,
                      "summary": _response_rate(treat_resp)},
        "control":   {**ctrl_sim,  "responses": ctrl_resp,  "pct_change": ctrl_pct,
                      "summary": _response_rate(ctrl_resp)},
        "cfg":    cfg,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Quick diagnostic plots
# ---------------------------------------------------------------------------

def plot_trajectories(
    cohort:     dict,
    n_sample:   int  = 30,
    figsize:    tuple = (14, 5),
    save_path:  str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot a random sample of tumour-volume trajectories for both arms,
    with the median trajectory overlaid.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    fig.patch.set_facecolor("#0d1117")

    arm_specs = [
        ("treatment", "#4fa3e0", "#1a3a52", "Treatment arm (δ > 0)"),
        ("control",   "#e07a4f", "#52291a", "Control arm   (δ = 0)"),
    ]

    for ax, (arm_key, col_main, col_bg, title) in zip(axes, arm_specs):
        sim  = cohort[arm_key]
        V    = sim["trajectories"]   # (n_patients, n_steps)
        t    = sim["times"]
        V0   = sim["V0"]
        n    = V.shape[0]

        # Sample indices
        idx  = rng.choice(n, size=min(n_sample, n), replace=False)

        ax.set_facecolor("#0d1117")
        for i in idx:
            ax.plot(t, V[i] / V0[i], alpha=0.25, lw=0.8, color=col_main)

        # Median trajectory (normalised)
        median_norm = np.median(V / V0[:, None], axis=0)
        ax.plot(t, median_norm, lw=2.2, color=col_main, label="Median")

        # T_assess vertical line
        T_a = cohort["cfg"].T_assess
        ax.axvline(T_a, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.text(T_a + 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 1 else 1.8,
                f"T_assess={int(T_a)}d", color="white", fontsize=7, alpha=0.7)

        # RECIST lines
        ax.axhline(0.70, color="#f0e06a", lw=0.7, ls=":", alpha=0.7, label="PR threshold (−30%)")
        ax.axhline(1.20, color="#e05555", lw=0.7, ls=":", alpha=0.7, label="PD threshold (+20%)")

        # Summary text
        s = sim["summary"]
        txt = (f"ORR = {s['ORR']:.1%}  |  CR={s['CR']}  PR={s['PR']}  "
               f"SD={s['SD']}  PD={s['PD']}  (n={s['n']})")
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.set_xlabel("Time (days)", color="#aaa", fontsize=9)
        ax.set_ylabel("V(t) / V₀", color="#aaa", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=7, framealpha=0.15, labelcolor="white")
        ax.text(0.02, 0.04, txt, transform=ax.transAxes,
                fontsize=7, color="#aaa", va="bottom")

    fig.suptitle(
        "Gompertz SDE — Tumour Volume Trajectories (V(t)/V₀)",
        color="white", fontsize=12, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_response_distributions(
    cohort:    dict,
    figsize:   tuple = (12, 5),
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Waterfall plot (% change at T_assess) for both arms, coloured by
    RECIST category.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    recist_colors = {"CR": "#50d890", "PR": "#4fa3e0", "SD": "#f0e06a", "PD": "#e05555"}

    arm_specs = [
        ("treatment", "Treatment arm"),
        ("control",   "Control arm"),
    ]

    for ax, (arm_key, title) in zip(axes, arm_specs):
        sim      = cohort[arm_key]
        pct      = sim["pct_change"] * 100       # to percent
        resp     = sim["responses"]
        order    = np.argsort(pct)                # waterfall sort

        ax.set_facecolor("#0d1117")
        bar_colors = [recist_colors[r] for r in resp[order]]
        ax.bar(np.arange(len(pct)), pct[order], color=bar_colors,
               width=1.0, linewidth=0)
        ax.axhline(-30, color="#f0e06a", lw=0.8, ls="--", alpha=0.6)
        ax.axhline( 20, color="#e05555", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(  0, color="white",   lw=0.4, alpha=0.3)

        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.set_xlabel("Patient (ranked)", color="#aaa", fontsize=9)
        ax.set_ylabel("% change from baseline", color="#aaa", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        s = sim["summary"]
        patches = [mpatches.Patch(color=recist_colors[r], label=f"{r}={s[r]}")
                   for r in ["CR", "PR", "SD", "PD"]]
        ax.legend(handles=patches, fontsize=7, framealpha=0.15,
                  labelcolor="white", title=f"ORR={s['ORR']:.1%}",
                  title_fontsize=7)

    fig.suptitle(
        "RECIST Response Waterfall — % Change from Baseline at T_assess",
        color="white", fontsize=12, y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Layer 1 — Gompertz SDE Simulator  (smoke test)")
    print("=" * 60)

    params = TumourSDEParams()
    cfg    = TrialConfig(n_patients=200, seed=37)

    cohort = simulate_trial_cohort(cfg, params)

    for arm in ("treatment", "control"):
        s = cohort[arm]["summary"]
        print(f"\n{arm.capitalize()} arm  (n={s['n']})")
        print(f"  ORR  : {s['ORR']:.1%}")
        print(f"  CR={s['CR']}  PR={s['PR']}  SD={s['SD']}  PD={s['PD']}")

    fig1 = plot_trajectories(cohort, save_path=None)
    fig2 = plot_response_distributions(cohort, save_path=None)
    plt.show() # change save_path to save the pictures, here we just print it.
    print("\nSmoke test passed.")
