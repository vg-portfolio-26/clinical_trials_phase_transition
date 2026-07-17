import logging
import numpy as np
import pandas as pd
from pathlib import Path
from .helpers import log_separator
from lifelines import KaplanMeierFitter
from scipy.stats import beta as beta_dist

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "04_analysis"


def fit_km_curves_by_phase(survival_df: pd.DataFrame) -> dict:
    logging.info("Fitting Kaplan-Meier curves per phase...")

    curves = {}
    for phase, g in survival_df.groupby("phase"):
        kmf = KaplanMeierFitter(label=phase)
        kmf.fit(durations=g["duration_days"], event_observed=g["event"])
        curves[phase] = kmf
 
        median = kmf.median_survival_time_
        median_str = f"{median:.0f}d" if np.isfinite(median) else "not reached"
        logging.info(
            f"  [{phase}] n = {len(g):,}  events = {int(g['event'].sum()):,}  "
            f"median time-to-progress = {median_str}  "
            f"S(5yr) = {kmf.survival_function_at_times(1826).iloc[0]:.3f}  "
            f"S(10yr) = {kmf.survival_function_at_times(3652).iloc[0]:.3f}"
        )
    log_separator()

    return curves
 
 
def fit_km_curves_by_phase_disease(survival_df: pd.DataFrame, min_events: int = 10) -> dict:
    """
    Fit Kaplan-Meier curves per (phase, mesh_tree_group), only for strata
    with at least `min_events` observed progression events. Sparser strata
    are intentionally omitted here; compute_km_progression_probability()
    falls back to the phase-level curve for them.
 
    Returns {(phase, mesh_tree_group): KaplanMeierFitter}.
    """
    
    logging.info(f"Fitting Kaplan-Meier curves per phase - disease tree group (min_events = {min_events})...")

    curves = {}
    n_fit, n_skipped = 0, 0
    for (phase, tree), g in survival_df.groupby(["phase", "mesh_tree_group"]):
        if g["event"].sum() < min_events:
            n_skipped += 1
            continue
        kmf = KaplanMeierFitter(label = f"{phase}/{tree}")
        kmf.fit(durations = g["duration_days"], event_observed = g["event"])
        curves[(phase, tree)] = kmf
        n_fit += 1
 
    logging.info(
        f"Tree-level KM curves fit: {n_fit:,} strata  "
        f"(skipped {n_skipped:,} strata below min_events; phase-level fallback applies)"
    )
    log_separator()

    return curves
 
 
def compute_km_progression_probability(survival_df: pd.DataFrame, phase_curves: dict, tree_curves: dict) -> pd.DataFrame:
    """
    Attach km_progression_prob = 1 - S(duration_days) to every program,
    using the tree-specific curve when available, falling back to the
    phase-level curve otherwise.
    """

    logging.info("Computing per-program KM progression probabilities...")

    df = survival_df.copy()
    probs = np.empty(len(df))
 
    for phase, g in df.groupby("phase"):
        phase_kmf = phase_curves[phase]
        for idx, row in g.iterrows():
            key = (phase, row["mesh_tree_group"])
            kmf = tree_curves.get(key, phase_kmf)
            s_t = kmf.survival_function_at_times(row["duration_days"]).iloc[0]
            probs[df.index.get_loc(idx)] = 1.0 - s_t
 
    df["km_progression_prob"] = probs
    logging.info(
        f"km_progression_prob assigned — mean = {df['km_progression_prob'].mean():.3f}, "
        f"median = {df['km_progression_prob'].median():.3f}"
    )
    df.to_csv(_OUTPUT_DIR / "survival_dataset_scored.csv", index = False)
    log_separator()

    return df
 
 
def survival_dataset_diagnostics(survival_df: pd.DataFrame) -> None:
    logging.info("Survival dataset diagnostics...")

    for phase, g in survival_df.groupby("phase"):
        logging.info(
            f"  [{phase}] n = {len(g):,}  "
            f"events = {int(g['event'].sum()):,} ({g['event'].mean():.1%})  "
            f"median_followup = {g['duration_days'].median():.0f}d  "
            f"mean_km_prob = {g['km_progression_prob'].mean():.3f}"
        )
    log_separator()


def _eb_shrink(
    group_df: pd.DataFrame,
    group_keys: list,
    prior: dict,
    credibility: float = 0.95,
    n_trials_col: str = None,
    success_col: str = "km_progression_prob",
) -> pd.DataFrame:
    """
    Compute EB-shrunk Beta posterior for each group defined by `group_keys`.
 
    success_col : str
        Column holding the per-row outcome signal. Defaults to"km_progression_prob".
    n_trials_col : str or None
        When provided, n_trials = sum of that column per group and
        n_success = sum(success_col * n_trials_col), so a 10-study
        program contributes proportionally more evidence than a 1-study
        program. When None, n_trials = row count and n_success = sum(success_col).
    """
    
    if n_trials_col is not None:
        _df = group_df.copy()
        _df["_weighted_success"] = _df[success_col] * _df[n_trials_col]
        g = (
            _df.groupby(group_keys, as_index = False)
            .agg(
                n_trials = (n_trials_col, "sum"),
                n_success = ("_weighted_success", "sum"),
            )
        )
    else:
        g = (
            group_df.groupby(group_keys, as_index = False)
            .agg(n_trials = (success_col, "size"), n_success = (success_col, "sum"))
        )
 
    g["empirical_rate"] = g["n_success"] / g["n_trials"]
 
    alpha0, beta0 = prior["alpha"], prior["beta"]
    g["posterior_alpha"] = g["n_success"] + alpha0
    g["posterior_beta"] = g["n_trials"] - g["n_success"] + beta0
    g["eb_rate"] = g["posterior_alpha"] / (g["posterior_alpha"] + g["posterior_beta"])
 
    a, b = g["posterior_alpha"], g["posterior_beta"]
    g["std_error"] = np.sqrt((a * b) / (((a + b) ** 2) * (a + b + 1)))
 
    lo = (1 - credibility) / 2
    hi = 1 - lo
    g["ci_lower"] = beta_dist.ppf(lo, a, b)
    g["ci_upper"] = beta_dist.ppf(hi, a, b)
    g["ci_width"] = g["ci_upper"] - g["ci_lower"]
 
    return g


def compute_phase_eb_priors(scored_survival_df: pd.DataFrame, prior_strength_cap: float = 100) -> dict:
    """
    One Beta(alpha, beta) prior per phase, estimated via method-of-moments on group level (mesh_id) mean km_progression_prob. 
    The prior strength is capped at `prior_strength_cap` to avoid overly strong shrinkage when the between-group variance is very small.
    Mesh groups with only one program are excluded from the between-group variance estimate, 
    since a singleton's "rate" reflects only its own point estimate, not genuine between-group heterogeneity.
    """

    logging.info("Computing EB priors on phase level...")
 
    priors = {}
    for phase, g in scored_survival_df.groupby("phase"):
        global_mean = g["km_progression_prob"].mean()
 
        group_sizes = g.groupby("mesh_id").size()
        multi_groups = group_sizes[group_sizes > 1].index
        group_rates_multi = (
            g[g["mesh_id"].isin(multi_groups)]
            .groupby("mesh_id")["km_progression_prob"]
            .mean()
        )
        between_group_var = group_rates_multi.var() if len(group_rates_multi) >= 2 else np.nan
 
        eps = 1e-6
        if pd.isna(between_group_var) or between_group_var <= eps:
            prior_strength = prior_strength_cap
        else:
            prior_strength = (global_mean * (1 - global_mean)) / between_group_var - 1
            prior_strength = float(np.clip(prior_strength, 2, prior_strength_cap))
 
        alpha = global_mean * prior_strength
        beta_param = (1 - global_mean) * prior_strength
 
        priors[phase] = {
            "alpha": alpha, "beta": beta_param,
            "strength": prior_strength, "mean": global_mean,
            "n_programs": len(g),
        }
 
        logging.info(
            f"  [{phase}] mean = {global_mean:.2%}  strength = {prior_strength:.1f}"
            f"{' [AT CAP]' if prior_strength >= 99.9 else ''}  n_programs = {len(g):,}"
        )
    
    log_separator()

    return priors
 
 
def shrink_pairs_by_phase(scored_survival_df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """
    EB-shrunk progression rate per (rxcui, mesh_id, phase), using
    km_progression_prob as the soft outcome and n_studies as the trial weight.
    """

    logging.info("Shrinking on program (pair) level...")
 
    out = []
    for phase, g in scored_survival_df.groupby("phase"):
        shrunk = _eb_shrink(g, ["rxcui", "mesh_id"], priors[phase], n_trials_col = "n_studies")
        shrunk["phase"] = phase
        out.append(shrunk)

    pairs = pd.concat(out, ignore_index = True)
    pairs.to_csv(_OUTPUT_DIR / "shrinked_pairs.csv", index = False)

    return pairs
 
 
def shrink_tree_groups_by_phase(scored_survival_df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """EB-shrunk progression rate per (mesh_tree_group, phase), survival-weighted by n_studies"""

    logging.info("Shrinking on tree level...")
 
    out = []
    for phase, g in scored_survival_df.groupby("phase"):
        shrunk = _eb_shrink(g, ["mesh_tree_group"], priors[phase])
        shrunk["phase"] = phase
        out.append(shrunk)
 
    out = pd.concat(out, ignore_index = True)
    out.to_csv(_OUTPUT_DIR / "shrinked_trees.csv", index = False)
    return out


def compute_strategic_profiles(scored_df: pd.DataFrame, level: str = "pairs") -> pd.DataFrame:
    """
    Assign strategic profiles based on EB-shrunk progression rates and uncertainty
    """
    df = scored_df.copy()
 
    q75_rate = df.groupby("phase")["eb_rate"].transform("quantile", 0.75)
    q25_ci = df.groupby("phase")["ci_width"].transform("quantile", 0.25)
 
    sufficient_evidence = df["n_trials"] > 3
    high_rate = df["eb_rate"] >= q75_rate
    low_uncertainty = df["ci_width"] <= q25_ci
 
    df["strategic_profile"] = "STANDARD"
    df.loc[~sufficient_evidence & high_rate, "strategic_profile"] = "EMERGING_SIGNAL"
    df.loc[sufficient_evidence & high_rate  & low_uncertainty, "strategic_profile"] = "HIGH_CONFIDENCE"
    df.loc[sufficient_evidence & high_rate  & ~low_uncertainty, "strategic_profile"] = "HIGH_RISK_HIGH_REWARD"
    df.loc[sufficient_evidence & ~high_rate &  low_uncertainty, "strategic_profile"] = "STABLE_BUT_MODEST"
 
    df = df.sort_values("eb_rate", ascending = False).reset_index(drop = True)
 
    logging.info(f"Strategic profile: {df['strategic_profile'].value_counts().to_dict()}")
    df.to_csv(_OUTPUT_DIR / f"strategic_profile_{level}.csv", index = False)
    log_separator()
 
    return df