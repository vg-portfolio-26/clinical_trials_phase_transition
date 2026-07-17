import logging
import pandas as pd
from pathlib import Path
from .helpers import log_separator
from .config import DEFAULT_MAX_GAP, TARGET_PHASES, PHASE_RANK, REFERENCE_DATE

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "03_postprocessed_data"


def build_master_datasets(preprocessed_data):
    logging.info("Building study-level master dataset...")

    df = preprocessed_data["studies_preprocessed"].copy()
    logging.info(f"studies table: {df.shape}")

    df = df.merge(preprocessed_data["interventions_preprocessed"], on = "nct_id", how = "left")
    logging.info(f"after merging interventions: {df.shape}")

    df = df.merge(preprocessed_data["browse_preprocessed_tree"], on = "nct_id", how = "left")
    logging.info(f"after merging browse: {df.shape}")

    needed_cols = ["nct_id", "phase", "mesh_id", "drug_name"]
    for col in needed_cols:
        n_rows = len(df)
        df = df.dropna(subset=[col])
        logging.info(f"after dropping NAs ({col}): {df.shape} (-{(n_rows - len(df))/n_rows:.2%})")

    df.to_csv(_OUTPUT_DIR / "master_dataset_studies.csv", index = False)
    logging.info("Study-level master dataset created!")
    log_separator()

    logging.info("Building program-level master dataset...")
    group_keys = ["rxcui", "mesh_id", "phase"]

    df["completed_nct_id"] = df["nct_id"].where(df["completed_with_evidence"] == 1)
    df["terminated_or_withdrawn_nct_id"] = df["nct_id"].where(df["is_terminated_or_withdrawn"])

    programs = (
        df.groupby(group_keys, as_index = False)
        .agg(
            candidate_drug_name = ("candidate_drug_name", "first"),
            mesh_descriptor = ("mesh_descriptor", "first"),
            mesh_tree_group = ("mesh_tree_group", "first"),
            n_studies = ("nct_id", "nunique"),
            n_completed = ("completed_nct_id", "nunique"),
            n_terminated_or_withdrawn = ("terminated_or_withdrawn_nct_id", "nunique"),
            earliest_start = ("start_date", "min"),
            latest_completion = ("effective_completion_date", "max"),
        )
    )

    completed_anchor = (
        df.loc[df["completed_with_evidence"] == 1]
        .groupby(group_keys, as_index = False)
        .agg(earliest_completed_completion = ("effective_completion_date", "min"))
    )

    programs = programs.merge(completed_anchor, on = group_keys, how = "left")
    programs["earliest_completed_completion"] = programs["earliest_completed_completion"].where(programs["n_completed"] > 0)

    logging.info(f"Final program-level master dataset table: {programs.shape}")
    programs.to_csv(_OUTPUT_DIR / "master_dataset_programs.csv", index = False)
    logging.info("Program-level dataset created!")
    log_separator()

    return programs


def compute_empirical_max_gap(master_dataset_programs: pd.DataFrame, percentile: float = 90.0, min_progressors: int = 10,) -> dict:
    logging.info(f"Computing empirical max_gap (p{percentile:.0f}, per-phase) from observed escalation gaps...")
 
    df = master_dataset_programs.copy()
    df["phase_rank"] = df["phase"].map(PHASE_RANK)
 
    phase_gaps: dict[str, list[float]] = {p: [] for p in TARGET_PHASES}
 
    for (_, _), g in df.groupby(["rxcui", "mesh_id"], sort = False):
        g = g.sort_values("phase_rank")
        future = g[["phase_rank", "earliest_start"]].dropna(subset=["earliest_start"])
 
        for _, row in g.iterrows():
            phase = row["phase"]
            curr_start = row["earliest_start"]
            if phase not in TARGET_PHASES:
                continue
            if row["n_completed"] == 0 or pd.isna(curr_start):
                continue
 
            fut = future[
                (future["phase_rank"] > row["phase_rank"])
                & (future["earliest_start"] >= curr_start)
            ]
            if fut.empty:
                continue
 
            gap = (fut["earliest_start"].min() - curr_start).days
            phase_gaps[phase].append(gap)
 
    thresholds: dict[str, int] = {}
    all_gaps: list[float] = []
 
    for phase in TARGET_PHASES:
        gaps = phase_gaps[phase]
        all_gaps.extend(gaps)
        lit = DEFAULT_MAX_GAP
 
        if len(gaps) >= min_progressors:
            s = pd.Series(gaps, dtype = float)
            empirical = int(round(s.quantile(percentile / 100.0)))
            thresholds[phase] = empirical
            logging.info(
                f"  [{phase}] p{percentile:.0f} = {empirical}d "
                f"({empirical / 365.25:.2f} yr) | "
                f"median = {s.median():.0f}d | "
                f"max = {s.max():.0f}d | "
                f"n = {len(gaps):,}"
            )
        else:
            thresholds[phase] = lit
            logging.warning(f"  [{phase}] only {len(gaps)} gaps — below min ({min_progressors}); using literature default ({lit}d)")
 
    if all_gaps:
        s = pd.Series(all_gaps, dtype = float)
        logging.info(
            f"Overall gap distribution (n = {len(all_gaps):,}): "
            f"p25 = {s.quantile(0.25):.0f}d  "
            f"median = {s.median():.0f}d  "
            f"p{percentile:.0f} = {s.quantile(percentile/100):.0f}d  "
            f"p99 = {s.quantile(0.99):.0f}d  "
            f"max = {s.max():.0f}d"
        )
 
    log_separator()
    return thresholds


def build_survival_dataset(
        master_dataset_programs: pd.DataFrame,
        max_gap: int | dict[str, int] = DEFAULT_MAX_GAP,
        reference_date: pd.Timestamp = pd.Timestamp(REFERENCE_DATE)
) -> pd.DataFrame:
    """
    Construct the (duration, event) survival table: one row per (rxcui, mesh_id, phase) program in a target phase.
 
    event = 1  : confirmed progression (a higher-phase program for the
                 same drug-mesh pair starts within [curr_start, curr_start + phase_max_gap]).
                 duration = days from curr_start to that next-phase start.
 
    event = 0  : everything else (right-censored). Two sub-cases, both
                 simply contribute "had not progressed as of duration":
                   (a) status-confirmed failure (all studies terminated or
                       withdrawn) — censored at latest_completion (the last
                       date we have any information for this program), or
                       reference_date if latest_completion is missing.
                   (b) genuinely ongoing / unknown — censored at reference_date.
    """

    logging.info("Building survival dataset...")
 
    df = master_dataset_programs.copy()
    df["phase_rank"] = df["phase"].map(PHASE_RANK)
 
    rows = []
    for (drug, mesh), g in df.groupby(["rxcui", "mesh_id"], sort = False):
        g = g.sort_values("phase_rank")
        future = g[["phase_rank", "earliest_start"]].dropna()
 
        for _, row in g.iterrows():
            phase = row["phase"]
            if phase not in TARGET_PHASES:
                continue
 
            curr_rank  = row["phase_rank"]
            curr_start = row["earliest_start"]
            if pd.isna(curr_start):
                continue

            if isinstance(max_gap, dict):
                phase_max = int(max_gap.get(phase, DEFAULT_MAX_GAP))
            else:
                phase_max = int(max_gap)

            event, duration = 0, None
 
            # Event: confirmed progression
            if row["n_completed"] > 0:
                fut = future[future["phase_rank"] > curr_rank]
                if not fut.empty:
                    fut_valid = fut[
                        (fut["earliest_start"] >= curr_start)
                        & (fut["earliest_start"] - curr_start <= pd.Timedelta(days=phase_max))
                    ]
                    if not fut_valid.empty:
                        next_start = fut_valid["earliest_start"].min()
                        event = 1
                        duration = (next_start - curr_start).days
 
            # Censored: status-confirmed failure
            if duration is None:
                n_studies = row.get("n_studies", 0) or 0
                n_term = row.get("n_terminated_or_withdrawn", 0) or 0
                if n_studies > 0 and n_term == n_studies:
                    censor_date = row.get("latest_completion")
                    censor_date = censor_date if pd.notna(censor_date) else reference_date
                    duration = max((censor_date - curr_start).days, 0)
 
            # Censored: ongoing / unknown
            if duration is None:
                duration = max((reference_date - curr_start).days, 0)
 
            rows.append({
                "rxcui": drug, "candidate_drug_name": row.get("candidate_drug_name"),
                "mesh_id": mesh, "mesh_descriptor": row.get("mesh_descriptor"),
                "phase": phase,
                "mesh_tree_group": row.get("mesh_tree_group"),
                "n_studies": row.get("n_studies", 1),
                "duration_days": duration,
                "event": event,
            })
 
    survival_df = pd.DataFrame(rows)
    survival_df = survival_df[survival_df["duration_days"] > 0]
 
    n_events = int(survival_df["event"].sum())
    n_censored = len(survival_df) - n_events
    logging.info(
        f"Survival dataset built: n = {len(survival_df):,}  "
        f"events = {n_events:,} ({n_events / len(survival_df):.1%})  "
        f"censored = {n_censored:,} ({n_censored / len(survival_df):.1%})"
    )
    survival_df.to_csv(_OUTPUT_DIR / "survival_dataset.csv", index = False)
    log_separator()
    return survival_df