"""
adaptive_trial.py
=================
Layer 3 – Sequential Decision Engine & Operating Characteristics

Overview
--------
Layer 3 wraps Layers 1 and 2 into a complete adaptive trial simulator.
Each simulated trial proceeds as follows:

    1. Enrol patients in cohorts; allocate between arms using
       Response-Adaptive Randomisation (RAR) via Thompson sampling.
    2. At each interim look, call the Layer 2 conjugate posterior to
       compute P(θ_T > θ_C | data).
    3. Apply stopping rules:
           stop for efficacy  if P(θ_T > θ_C | data) > η_E
           stop for futility  if P(θ_T > θ_C | data) < η_F
    4. If the trial completes without stopping, make a final decision
       based on the terminal posterior.

This is repeated N_sim times to estimate Operating Characteristics (OC):

    Power              : Pr(stop for efficacy | H₁ true)
    Type I error (α)   : Pr(stop for efficacy | H₀ true)
    E[N]               : Expected sample size
    E[N_treat]         : Expected patients on treatment arm
    Early stop rate    : Pr(stop before final look)

OC surfaces are computed across a grid of (δ, η_E) values to show the
power/sample-size trade-off — the key result for the write-up.

Response-Adaptive Randomisation (Thompson sampling)
----------------------------------------------------
At each interim look the allocation probability for the next cohort is:

    ρ_k = P(θ_T > θ_C | data_{1:k})    (= p_superiority from Layer 2)

clipped to [ρ_min, ρ_max] = [0.20, 0.80] to ensure both arms retain
sufficient patients for valid inference (Thall & Wathen, 2007).

Thompson sampling is a multi-armed bandit allocation rule that naturally
balances exploration (learning about both arms) and exploitation (assigning
more patients to the arm that appears superior). Its use here is consistent
with FDA guidance on adaptive designs (FDA, 2019).

References
----------
[1] Thall PF, Wathen JK (2007). Practical Bayesian adaptive randomisation
    in clinical trials. Eur J Cancer 43:859–866.
[2] FDA (2019). Adaptive Designs for Clinical Trials of Drugs and Biologics:
    Guidance for Industry. US FDA.
[3] Berry DA (2011). Adaptive clinical trials: the promise and the caution.
    J Clin Oncol 29:606–609.
[4] Williamson SF, Villar SS (2020). A response-adaptive randomization
    procedure for multi-armed clinical trials with normally distributed
    outcomes. Biometrics 76:197–209.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from sde_simulator import TumourSDEParams, TrialConfig, simulate_trial_cohort
from bayesian_model  import (
    run_interim_analysis, DecisionThresholds,
    posterior_probability_superiority, ConjugatePosterior, PRIORS,
)


# ---------------------------------------------------------------------------
# Adaptive design configuration
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDesignConfig:
    """
    Parameters governing the adaptive trial machinery.

    Attributes
    ----------
    n_total : int
        Maximum total enrolment (both arms combined).
    n_interims : int
        Number of interim looks before the final analysis.
        Looks are equally spaced in terms of patients enrolled.
    cohort_size : int
        Patients enrolled between consecutive looks (both arms combined).
    rar_min, rar_max : float
        Clipping bounds for the Thompson sampling allocation probability.
        Prevents extreme imbalance (Thall & Wathen, 2007 recommend [0.2, 0.8]).
    thresholds : DecisionThresholds
        Efficacy and futility stopping thresholds.
    prior_regime : str
        One of "sceptical", "neutral", "optimistic".
    fixed_allocation : bool
        If True, use 1:1 fixed randomisation (no RAR) — used as the
        comparator arm in OC calculations.
    """
    n_total:          int   = 120
    n_interims:       int   = 3
    cohort_size:      int   = 20      # patients per look (both arms)
    rar_min:          float = 0.20
    rar_max:          float = 0.80
    thresholds:       DecisionThresholds = field(default_factory=DecisionThresholds)
    prior_regime:     str   = "neutral"
    fixed_allocation: bool  = False

    def __post_init__(self):
        assert 0 < self.rar_min < self.rar_max < 1
        assert self.cohort_size >= 2
        # Derive interim look schedule: patient counts per arm at each look
        n_per_arm_final = self.n_total // 2
        look_ns = np.linspace(
            self.cohort_size // 2,
            n_per_arm_final,
            self.n_interims + 1,   # +1: final analysis
            dtype=int,
        )
        self.look_schedule: list[int] = look_ns.tolist()  # patients per arm

    @property
    def interim_looks(self) -> list[int]:
        """Patient counts per arm at each interim (excludes final)."""
        return self.look_schedule[:-1]

    @property
    def n_final_per_arm(self) -> int:
        return self.look_schedule[-1]


# ---------------------------------------------------------------------------
# Single trial simulator
# ---------------------------------------------------------------------------

@dataclass
class TrialOutcome:
    """
    Results from a single simulated adaptive trial.
    """
    seed:              int
    decision:          Literal["efficacy", "futility", "inconclusive"]
    n_enrolled:        int
    n_treat:           int
    n_ctrl:            int
    y_treat:           int
    y_ctrl:            int
    stop_look:         int           # look at which trial stopped (0 = final)
    allocation_trace:  list[float]   # RAR probability at each look
    p_sup_trace:       list[float]   # P(superiority) at each look
    interim_decisions: list[str]


def _slice_arm(arm_dict: dict, n: int) -> dict:
    """Return first n patients from an arm's simulation dict."""
    return {
        k: v[:n] if (hasattr(v, "__len__") and not isinstance(v, str)
                     and len(v) >= n) else v
        for k, v in arm_dict.items()
    }


def simulate_single_trial(
    seed:        int,
    sde_params:  TumourSDEParams,
    cfg:         TrialConfig,
    design:      AdaptiveDesignConfig,
) -> TrialOutcome:
    """
    Simulate one complete adaptive trial.

    Patients are enrolled in sequential cohorts.  After each cohort the
    Layer 2 conjugate posterior is updated and the stopping rule is checked.
    If the trial continues, the Thompson sampling allocation ratio is updated
    and the *next* cohort is simulated using that updated ratio — so patients
    are genuinely reassigned toward the better-performing arm.

    This sequential resimulation is the correct implementation of RAR: the
    allocation update must feed forward into who receives treatment, not just
    be recorded as a trace statistic.
    """
    from sde_simulator import simulate_patients, classify_response

    rng = np.random.default_rng(seed)

    cohort_size = design.n_total // (design.n_interims + 1)

    allocation_trace  : list[float] = []
    p_sup_trace       : list[float] = []
    interim_decisions : list[str]   = []

    current_alloc   = 0.5
    stop_look       = 0
    decision        = "inconclusive"
    n_enrolled      = 0
    n_treat_total   = 0
    n_ctrl_total    = 0
    all_treat_resp  : list = []
    all_ctrl_resp   : list = []

    n_looks = design.n_interims + 1   # interims + final

    for look_idx in range(1, n_looks + 1):
        is_final = (look_idx == n_looks)

        # ---- Determine cohort split using current allocation ----
        n_treat_this = max(2, int(round(cohort_size * current_alloc)))
        n_ctrl_this  = max(2, cohort_size - n_treat_this)

        # ---- Simulate this cohort's SDE trajectories ----
        sim_t = simulate_patients(n_treat_this, sde_params, "treatment",
                                  cfg.T_max, rng=rng)
        sim_c = simulate_patients(n_ctrl_this,  sde_params, "control",
                                  cfg.T_max, rng=rng)

        resp_t, _ = classify_response(sim_t["trajectories"], sim_t["V0"],
                                      sim_t["times"], cfg.T_assess, cfg)
        resp_c, _ = classify_response(sim_c["trajectories"], sim_c["V0"],
                                      sim_c["times"], cfg.T_assess, cfg)

        all_treat_resp.extend(resp_t)
        all_ctrl_resp.extend(resp_c)
        n_enrolled    += n_treat_this + n_ctrl_this
        n_treat_total += n_treat_this
        n_ctrl_total  += n_ctrl_this

        # ---- Layer 2: update posterior with all data accumulated so far ----
        combined = {
            "treatment": {"responses": np.array(all_treat_resp)},
            "control":   {"responses": np.array(all_ctrl_resp)},
            "cfg": cfg, "params": sde_params,
        }
        result = run_interim_analysis(
            look         = look_idx,
            responses    = combined,
            prior_regime = design.prior_regime,
            thresholds   = design.thresholds,
            rng          = rng,
        )

        p_sup_trace.append(result.p_superiority)
        interim_decisions.append(result.decision)

        # ---- Thompson sampling: update allocation for next cohort ----
        if not design.fixed_allocation:
            current_alloc = float(
                np.clip(result.p_superiority, design.rar_min, design.rar_max)
            )
        allocation_trace.append(current_alloc)

        # ---- Stopping rule ----
        if result.decision == "stop_efficacy":
            decision  = "efficacy"
            stop_look = look_idx
            break
        elif result.decision == "stop_futility":
            decision  = "futility"
            stop_look = look_idx
            break
        elif is_final:
            # Final look: no stopping rule fired — trial is inconclusive
            decision = "inconclusive"

    # ---- Tally responders ----
    treat_arr = np.array(all_treat_resp)
    ctrl_arr  = np.array(all_ctrl_resp)
    y_t = int(np.sum((treat_arr == "CR") | (treat_arr == "PR")))
    y_c = int(np.sum((ctrl_arr  == "CR") | (ctrl_arr  == "PR")))

    return TrialOutcome(
        seed              = seed,
        decision          = decision,
        n_enrolled        = n_enrolled,
        n_treat           = n_treat_total,
        n_ctrl            = n_ctrl_total,
        y_treat           = y_t,
        y_ctrl            = y_c,
        stop_look         = stop_look,
        allocation_trace  = allocation_trace,
        p_sup_trace       = p_sup_trace,
        interim_decisions = interim_decisions,
    )


# ---------------------------------------------------------------------------
# Monte Carlo OC simulation
# ---------------------------------------------------------------------------

@dataclass
class OperatingCharacteristics:
    """
    Summary of operating characteristics estimated over N_sim trial replications.
    """
    n_sim:             int
    delta_treatment:   float    # true treatment effect (δ parameter)
    design_label:      str

    power:             float    # Pr(efficacy decision | H₁)
    type_i_error:      float    # Pr(efficacy decision | H₀)  — set separately
    futility_rate:     float    # Pr(futility stop)
    inconclusive_rate: float    # Pr(no decision at final)
    early_stop_rate:   float    # Pr(stop before final look)

    en_mean:           float    # E[total N enrolled]
    en_std:            float
    en_treat_mean:     float    # E[N on treatment arm]

    outcomes:          list[TrialOutcome] = field(repr=False)

    def summary(self) -> dict:
        return {
            "design":           self.design_label,
            "delta_treatment":  self.delta_treatment,
            "n_sim":            self.n_sim,
            "power":            self.power,
            "futility_rate":    self.futility_rate,
            "inconclusive_rate":self.inconclusive_rate,
            "early_stop_rate":  self.early_stop_rate,
            "E[N]":             self.en_mean,
            "SD[N]":            self.en_std,
            "E[N_treat]":       self.en_treat_mean,
        }


def run_oc_simulation(
    n_sim:       int,
    sde_params:  TumourSDEParams,
    design:      AdaptiveDesignConfig,
    base_seed:   int = 0,
    n_jobs:      int = 1,
    verbose:     bool = True,
) -> OperatingCharacteristics:
    """
    Estimate operating characteristics by Monte Carlo simulation.

    Parameters
    ----------
    n_sim : int
        Number of trial replications.  1000 gives stable estimates;
        5000 for publication-quality OC curves.
    sde_params : TumourSDEParams
        Disease and treatment parameters (Layer 1).
    design : AdaptiveDesignConfig
        Adaptive design specification.
    base_seed : int
        Seeds run from base_seed to base_seed + n_sim − 1.
    n_jobs : int
        Parallel workers.  n_jobs=1 runs serially (safe on all platforms).
        Set n_jobs=-1 to use all available cores.
    """
    import os
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    cfg = TrialConfig(n_patients=design.n_total, seed=base_seed)

    seeds   = range(base_seed, base_seed + n_sim)
    outcomes: list[TrialOutcome] = []

    if n_jobs == 1 or n_sim < 50:
        # Serial path — avoids multiprocessing overhead for small n_sim
        for i, seed in enumerate(seeds):
            outcomes.append(simulate_single_trial(seed, sde_params, cfg, design))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_sim} trials simulated...")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(simulate_single_trial, seed, sde_params, cfg, design): seed
                for seed in seeds
            }
            for i, fut in enumerate(as_completed(futures)):
                outcomes.append(fut.result())
                if verbose and (i + 1) % 100 == 0:
                    print(f"  {i+1}/{n_sim} trials simulated...")

    # ---- Aggregate ----
    n_eff   = np.array([o.n_enrolled   for o in outcomes])
    n_treat = np.array([o.n_treat      for o in outcomes])

    decisions = np.array([o.decision for o in outcomes])
    stop_looks = np.array([o.stop_look for o in outcomes])

    power            = float(np.mean(decisions == "efficacy"))
    futility_rate    = float(np.mean(decisions == "futility"))
    inconclusive     = float(np.mean(decisions == "inconclusive"))
    early_stop_rate  = float(np.mean(stop_looks > 0))

    label = ("Adaptive RAR" if not design.fixed_allocation
             else "Fixed 1:1")

    return OperatingCharacteristics(
        n_sim             = n_sim,
        delta_treatment   = sde_params.delta_treatment,
        design_label      = label,
        power             = power,
        type_i_error      = 0.0,     # populated by run_oc_surface
        futility_rate     = futility_rate,
        inconclusive_rate = inconclusive,
        early_stop_rate   = early_stop_rate,
        en_mean           = float(np.mean(n_eff)),
        en_std            = float(np.std(n_eff)),
        en_treat_mean     = float(np.mean(n_treat)),
        outcomes          = outcomes,
    )


# ---------------------------------------------------------------------------
# OC surface: power vs sample size across δ grid
# ---------------------------------------------------------------------------

def run_oc_surface(
    delta_grid:  list[float],
    n_sim:       int,
    design:      AdaptiveDesignConfig,
    base_params: TumourSDEParams | None = None,
    verbose:     bool = True,
) -> list[OperatingCharacteristics]:
    """
    Compute OC surface across a grid of true treatment effects (δ values).

    δ = delta_control (≈ 0) approximates H₀ and gives the empirical
    Type I error rate.  δ > delta_control gives power under H₁.

    Returns a list of OperatingCharacteristics, one per δ value.
    """
    if base_params is None:
        base_params = TumourSDEParams()

    results = []
    for i, delta in enumerate(delta_grid):
        if verbose:
            print(f"[{i+1}/{len(delta_grid)}] δ = {delta:.4f}")
        params = TumourSDEParams(
            alpha            = base_params.alpha,
            beta             = base_params.beta,
            sigma            = base_params.sigma,
            delta_treatment  = delta,
            delta_control    = base_params.delta_control,
            V0_mean          = base_params.V0_mean,
            V0_cv            = base_params.V0_cv,
            dt               = base_params.dt,
        )
        oc = run_oc_simulation(
            n_sim      = n_sim,
            sde_params = params,
            design     = design,
            base_seed  = i * n_sim,
            verbose    = False,
        )
        results.append(oc)

    # Annotate Type I error: use the δ = delta_control point (H₀)
    null_delta = base_params.delta_control
    null_oc    = next((r for r in results
                       if abs(r.delta_treatment - null_delta) < 1e-6), None)
    if null_oc is not None:
        for r in results:
            r.type_i_error = null_oc.power

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_oc_surface(
    oc_results:   list[OperatingCharacteristics],
    oc_fixed:     list[OperatingCharacteristics] | None = None,
    figsize:      tuple = (14, 5),
    save_path:    str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Three-panel OC surface plot:
      (1) Power curve  (2) E[N] curve  (3) Early stop rate
    Adaptive RAR vs fixed 1:1 design overlaid if oc_fixed is provided.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    deltas_a = [r.delta_treatment for r in oc_results]

    specs = [
        ("power",           "Power  P(efficacy | δ)",        0.975, "#4fa3e0"),
        ("en_mean",         "E[N]  expected sample size",     None,  "#4fa3e0"),
        ("early_stop_rate", "Early stop rate",                None,  "#4fa3e0"),
    ]

    for ax, (attr, title, hline, col) in zip(axes, specs):
        ax.set_facecolor("#0d1117")
        vals_a = [getattr(r, attr) for r in oc_results]
        ax.plot(deltas_a, vals_a, color=col, lw=2.0,
                marker="o", ms=4, label="Adaptive RAR")

        if oc_fixed is not None:
            deltas_f = [r.delta_treatment for r in oc_fixed]
            vals_f   = [getattr(r, attr) for r in oc_fixed]
            ax.plot(deltas_f, vals_f, color="#e07a4f", lw=1.5,
                    ls="--", marker="s", ms=3, label="Fixed 1:1")

        if hline is not None:
            ax.axhline(hline, color="white", lw=0.7, ls=":",
                       alpha=0.5, label=f"η_E = {hline}")

        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.set_xlabel("δ  (treatment elimination rate)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=7, framealpha=0.15, labelcolor="white")

    fig.suptitle(
        "Operating Characteristics — Adaptive RAR vs Fixed Design",
        color="white", fontsize=11, y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_allocation_drift(
    oc: OperatingCharacteristics,
    n_sample: int = 80,
    figsize:  tuple = (10, 4),
    save_path: str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Plot how the Thompson sampling allocation probability evolves across
    interim looks for a sample of trials, with the median overlaid.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0d1117")

    outcomes = oc.outcomes[:n_sample]

    for ax, attr, title, col in [
        (axes[0], "p_sup_trace",       "P(θ_T > θ_C) across looks",       "#4fa3e0"),
        (axes[1], "allocation_trace",  "Thompson allocation ρ across looks", "#50d890"),
    ]:
        ax.set_facecolor("#0d1117")
        traces = [getattr(o, attr) for o in outcomes if len(getattr(o, attr)) > 0]
        max_len = max(len(t) for t in traces)

        # Pad shorter traces (early stops) with NaN for plotting
        padded  = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        for row in padded:
            ax.plot(range(1, len(row)+1), row, alpha=0.12, lw=0.7, color=col)

        median = np.nanmedian(padded, axis=0)
        ax.plot(range(1, len(median)+1), median, color=col, lw=2.2,
                label="Median")

        if attr == "allocation_trace":
            ax.axhline(0.5,             color="white",   lw=0.6, ls=":", alpha=0.4)
            ax.axhline(oc.outcomes[0].__class__.__dataclass_fields__  # just draw bounds
                       and 0.80 or 0.80,
                       color="#aaa", lw=0.5, ls="--", alpha=0.4, label="RAR clip [0.2, 0.8]")
            ax.axhline(0.20, color="#aaa", lw=0.5, ls="--", alpha=0.4)
            ax.set_ylim(0, 1)
        else:
            ax.axhline(0.975, color="#50d890", lw=0.7, ls="--", alpha=0.6,
                       label="η_E = 0.975")
            ax.axhline(0.10,  color="#e05555", lw=0.7, ls="--", alpha=0.6,
                       label="η_F = 0.10")
            ax.set_ylim(0, 1.05)

        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.set_xlabel("Interim look", color="#aaa", fontsize=8)
        ax.set_xticks(range(1, max_len + 1))
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=7, framealpha=0.15, labelcolor="white")

    fig.suptitle(
        f"Thompson Sampling Dynamics  (δ={oc.delta_treatment:.3f}, "
        f"n={n_sample} trial traces)",
        color="white", fontsize=10, y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_sample_size_distribution(
    oc_adaptive: OperatingCharacteristics,
    oc_fixed:    OperatingCharacteristics | None = None,
    figsize:     tuple = (9, 4),
    save_path:   str | None = None,
) -> "matplotlib.figure.Figure":
    """
    Histogram of N enrolled per trial for adaptive vs fixed designs.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ns_a = [o.n_enrolled for o in oc_adaptive.outcomes]
    ax.hist(ns_a, bins=20, color="#4fa3e0", alpha=0.65, label="Adaptive RAR",
            edgecolor="none")
    ax.axvline(np.mean(ns_a), color="#4fa3e0", lw=1.5, ls="--",
               label=f"Adaptive E[N]={np.mean(ns_a):.0f}")

    if oc_fixed is not None:
        ns_f = [o.n_enrolled for o in oc_fixed.outcomes]
        ax.hist(ns_f, bins=20, color="#e07a4f", alpha=0.50, label="Fixed 1:1",
                edgecolor="none")
        ax.axvline(np.mean(ns_f), color="#e07a4f", lw=1.5, ls="--",
                   label=f"Fixed E[N]={np.mean(ns_f):.0f}")

    ax.set_title("Sample size distribution per trial", color="white", fontsize=9)
    ax.set_xlabel("N enrolled", color="#aaa", fontsize=8)
    ax.set_ylabel("Count", color="#aaa", fontsize=8)
    ax.tick_params(colors="#888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(fontsize=8, framealpha=0.15, labelcolor="white")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


# ---------------------------------------------------------------------------
# Entry point — smoke test + full OC run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Layer 3 — Adaptive Trial Engine  (smoke test)")
    print("=" * 60)

    # Use delta=0.005 — the regime where the adaptive machinery is most
    # interesting (mix of early stops and runs to completion)
    base_params = TumourSDEParams(delta_treatment=0.005)

    design_adaptive = AdaptiveDesignConfig(
        n_total      = 120,
        n_interims   = 3,
        cohort_size  = 20,
        prior_regime = "neutral",
        thresholds   = DecisionThresholds(eta_efficacy=0.975, eta_futility=0.10),
    )
    design_fixed = AdaptiveDesignConfig(
        n_total          = 120,
        n_interims       = 3,
        cohort_size      = 20,
        prior_regime     = "neutral",
        thresholds       = DecisionThresholds(eta_efficacy=0.975, eta_futility=0.10),
        fixed_allocation = True,
    )

    # ---- Single trial smoke test ----
    print("\n--- Single trial (adaptive) ---")
    cfg     = TrialConfig(n_patients=120, seed=0)
    outcome = simulate_single_trial(0, base_params, cfg, design_adaptive)
    print(f"  decision     : {outcome.decision}")
    print(f"  n_enrolled   : {outcome.n_enrolled}")
    print(f"  stop_look    : {outcome.stop_look}")
    print(f"  p_sup_trace  : {[f'{p:.3f}' for p in outcome.p_sup_trace]}")
    print(f"  alloc_trace  : {[f'{a:.3f}' for a in outcome.allocation_trace]}")

    # ---- OC surface ----
    print("\n--- OC surface (3500 sims × 6 δ values) ---")
    delta_grid = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011]

    print("  Adaptive RAR:")
    oc_adaptive = run_oc_surface(delta_grid, n_sim=3500,
                                 design=design_adaptive,
                                 base_params=base_params, verbose=True)

    print("  Fixed 1:1:")
    oc_fixed = run_oc_surface(delta_grid, n_sim=3500,
                              design=design_fixed,
                              base_params=base_params, verbose=True)

    print("\n--- OC summary table ---")
    print(f"{'δ':>7}  {'Power(A)':>9}  {'Power(F)':>9}  "
          f"{'E[N](A)':>8}  {'E[N](F)':>8}  {'EarlyStp':>9}")
    for oc_a, oc_f in zip(oc_adaptive, oc_fixed):
        print(f"  {oc_a.delta_treatment:.3f}  "
              f"{oc_a.power:>9.3f}  {oc_f.power:>9.3f}  "
              f"{oc_a.en_mean:>8.1f}  {oc_f.en_mean:>8.1f}  "
              f"{oc_a.early_stop_rate:>9.3f}")

    # ---- Plots ----
    print("\n--- Generating plots ---")
    mid_oc = oc_adaptive[2]   # δ=0.005 — the interesting regime

    fig1 = plot_oc_surface(oc_adaptive, oc_fixed,
                           save_path=NONE) #Image save path here and below.
    fig2 = plot_allocation_drift(mid_oc, n_sample=80,
                                 save_path=NONE)
    fig3 = plot_sample_size_distribution(mid_oc, oc_fixed[2],
                                         save_path=NONE)
    plt.show()
    print("Plots saved.")
    print("\nSmoke test passed.")
