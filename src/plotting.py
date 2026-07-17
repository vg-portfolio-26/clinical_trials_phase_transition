import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PHASE_ORDER  = ["phase1", "phase1/phase2", "phase2", "phase2/phase3"]
_PHASE_LABELS = {"phase1": "Phase 1", "phase1/phase2": "Phase 1/2", "phase2": "Phase 2", "phase2/phase3": "Phase 2/3"}
_PHASE_SHORT = {"phase1": "Ph1", "phase1/phase2": "Ph1/2", "phase2": "Ph2",  "phase2/phase3": "Ph2/3"}

_PROFILE_COLORS = {
    "HIGH_CONFIDENCE": "#2E7D32",
    "HIGH_RISK_HIGH_REWARD": "#E65100",
    "STABLE_BUT_MODEST": "#1565C0",
    "EMERGING_SIGNAL": "#6A1B9A",
    "STANDARD": "#9E9E9E"
}

_PHASE_COLORS = {
    "phase1": "#1976D2",
    "phase1/phase2": "#43A047",
    "phase2": "#FB8C00",
    "phase2/phase3": "#8E24AA"
}

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "05_visualize_results"


def plot_km_curves(phase_curves: dict, horizon_days: int = 4000) -> None:
    logging.info("Plotting phase cumulative progression probability...")

    records = []
    fig, ax = plt.subplots(figsize = (7, 4.2))

    for phase in _PHASE_ORDER:
        if phase not in phase_curves:
            continue
        kmf = phase_curves[phase]
        t = np.linspace(0, horizon_days, 200)
        surv = kmf.survival_function_at_times(t).values
        prog = 1 - surv

        ci = kmf.confidence_interval_survival_function_
        ci_t = ci.index.values
        ci_lo = 1 - np.interp(t, ci_t, ci.iloc[:, 1].values)
        ci_hi = 1 - np.interp(t, ci_t, ci.iloc[:, 0].values)

        color = _PHASE_COLORS[phase]
        ax.plot(t / 365.25, prog, label = _PHASE_LABELS[phase], color = color, lw = 2)
        ax.fill_between(t / 365.25, ci_lo, ci_hi, color = color, alpha = 0.15)

        for time_years, progression_probability, lower_ci, upper_ci in zip(t / 365.25, prog, ci_lo, ci_hi):
            records.append(
                {
                    "phase": phase,
                    "time_years": time_years,
                    "progression_probability": progression_probability,
                    "ci_lower": lower_ci,
                    "ci_upper": upper_ci,
                }
            )

    pd.DataFrame(records).to_csv(_OUTPUT_DIR / "km_curves_data.csv", index = False)

    ax.set_xlabel("Years since phase start")
    ax.set_ylabel("Cumulative progression probability")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, horizon_days / 365.25)
    ax.legend(fontsize = 8.5, loc = "upper right", framealpha = 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle = "--", linewidth = 0.4, alpha = 0.4)
 
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIR / "km_curves.png", dpi = 180, bbox_inches = "tight")
    plt.close()
    logging.info("Km curves plot saved!")


def plot_disease_opportunity_map(tree_scores: pd.DataFrame) -> None:
    logging.info("Plotting disease opportunity map...")

    nlm = pd.read_csv("output/01_raw_data/nlm_mesh_map.csv")
    lookup = (
        nlm[nlm["tree_number"] == nlm["mesh_tree_group"]]
        [["mesh_tree_group", "mesh_descriptor"]]
        .drop_duplicates()
    )
    df = tree_scores.merge(lookup, on = "mesh_tree_group", how = "left")
    df["disease"] = (
        df["mesh_descriptor"]
        .fillna(df["mesh_tree_group"])
        .str.title()
        .str[:44]
    )
 
    disease_order = (
        df.groupby("disease")["eb_rate"]
        .mean()
        .sort_values(ascending = True)
        .index.tolist()
    )
    disease_pos = {d: i for i, d in enumerate(disease_order)}
    phase_pos = {p: i for i, p in enumerate(_PHASE_ORDER)}
 
    df["y"] = df["disease"].map(disease_pos)
    df["x"] = df["phase"].map(phase_pos)
 
    max_s = 700
    size_scale = max_s / np.sqrt(df["n_trials"].max())
    df["s"] = np.sqrt(df["n_trials"]) * size_scale

    export_df = df[["disease", "phase", "eb_rate", "n_trials", "s", "x", "y"]].copy()
    export_df = export_df.rename(columns = {"s": "marker_size", "x": "x_position", "y": "y_position"})
    export_df.to_csv(_OUTPUT_DIR / "disease_map_data.csv", index = False)
 
    fig, ax = plt.subplots(figsize = (7, max(8, len(disease_order) * 0.31)))
 
    sc = ax.scatter(
        df["x"], df["y"],
        s = df["s"], c = df["eb_rate"],
        cmap = "YlGn", alpha = 0.88,
        edgecolors = "#555", linewidths = 0.4,
        vmin = 0, vmax = df["eb_rate"].max(),
    )
 
    ax.set_xticks(range(len(_PHASE_ORDER)))
    ax.set_xticklabels([_PHASE_LABELS[p] for p in _PHASE_ORDER], fontsize = 8.5)
    ax.set_yticks(range(len(disease_order)))
    ax.set_yticklabels(disease_order, fontsize = 7)
    ax.set_xlim(-0.6, len(_PHASE_ORDER) - 0.4)
    ax.set_ylim(-0.8, len(disease_order) - 0.2)
    ax.grid(linestyle = "--", linewidth = 0.4, alpha = 0.45)
    ax.spines[["top", "right"]].set_visible(False)
 
    cbar = plt.colorbar(sc, ax = ax, shrink = 0.30, pad = 0.02)
    cbar.set_label("EB Progression Rate", fontsize = 8)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    refs = [r for r in [50, 500, 3000] if r <= df["n_trials"].max()]
    for r in refs:
        ax.scatter([], [], s = np.sqrt(r) * size_scale, c = "#bdbdbd", edgecolors = "#555", linewidths = 0.4, label = f"{r:,} programs")
    ax.legend(title = "Evidence depth", fontsize = 7, title_fontsize = 7.5, loc = "upper right", framealpha = 0.88, borderpad = 0.8)
 
    plt.tight_layout()
    plt.savefig(_OUTPUT_DIR / "disease_map.png", dpi = 180, bbox_inches = "tight")
    plt.close()
    logging.info("Disease opportunity map saved!")


def plot_top_programs(pair_scores: pd.DataFrame, scored_survival_df: pd.DataFrame, top_n: int = 5) -> None:
    logging.info("Plotting top programs...")

    names = (
        scored_survival_df[["rxcui", "mesh_id", "candidate_drug_name", "mesh_descriptor"]]
        .drop_duplicates(subset = ["rxcui", "mesh_id"])
    )
    programs = (
        pair_scores
        .merge(names, on = ["rxcui", "mesh_id"], how = "left", suffixes = ("", "_from_names"))
        .reset_index(drop = True)
    )
    if "candidate_drug_name_from_names" in programs.columns:
        for column, source in [("candidate_drug_name", "candidate_drug_name_from_names"), ("mesh_descriptor", "mesh_descriptor_from_names")]:
            programs[column] = programs[column].combine_first(programs[source])
        programs = programs.drop(columns = ["candidate_drug_name_from_names", "mesh_descriptor_from_names"])

    prepared = programs[["rxcui", "mesh_id", "candidate_drug_name", "mesh_descriptor", "phase", "eb_rate", "ci_lower", "ci_upper", "strategic_profile", "n_trials"]].copy()
    prepared["label"] = (
        prepared["candidate_drug_name"].fillna(prepared["rxcui"]).str.title()
        + "  ·  "
        + prepared["mesh_descriptor"].fillna(prepared["mesh_id"]).str.title()
        + "  ["
        + prepared["phase"].map(_PHASE_SHORT)
        + "]"
    )
    prepared["sort_primary"] = np.where(
        prepared["strategic_profile"].eq("EMERGING_SIGNAL"),
        prepared["n_trials"],
        prepared["eb_rate"],
    )
    prepared["sort_secondary"] = np.where(
        prepared["strategic_profile"].eq("EMERGING_SIGNAL"),
        prepared["eb_rate"],
        prepared["n_trials"],
    )

    export_df = (
        prepared.sort_values(["strategic_profile", "sort_primary", "sort_secondary"], ascending = [True, False, False], kind = "mergesort")
        .reset_index(drop = True)
        .drop(columns = ["sort_primary", "sort_secondary"])
    )
    export_df.to_csv(_OUTPUT_DIR / "top_programs_data.csv", index = False)

    top10_df = (
        export_df.sort_values(["strategic_profile", "eb_rate", "n_trials"], ascending = [True, False, False], kind = "mergesort")
        .groupby("strategic_profile", group_keys = False)
        .head(10)
        .reset_index(drop = True)
    )
    top10_df.to_csv(_OUTPUT_DIR / "top10_programs_data.csv", index = False)

    ranked = (
        prepared[prepared["strategic_profile"] != "STANDARD"]
        .sort_values(["strategic_profile", "sort_primary", "sort_secondary"], ascending = [True, False, False], kind = "mergesort")
        .groupby("strategic_profile", group_keys = False)
        .head(top_n)
        .reset_index(drop = True)
        .drop(columns = ["sort_primary", "sort_secondary"])
    )
    colors = ranked["strategic_profile"].map(_PROFILE_COLORS).fillna(_PROFILE_COLORS["STANDARD"])
    xerr = [ranked["eb_rate"] - ranked["ci_lower"], ranked["ci_upper"] - ranked["eb_rate"]]

    fig, ax = plt.subplots(figsize = (9, max(5, len(ranked) * 0.38)))
    y = np.arange(len(ranked))
    ax.barh(y, ranked["eb_rate"], xerr = xerr, color = colors, height = 0.65, capsize = 3, error_kw = {"lw": 1.2, "color": "#757575"})
    ax.set_yticks(y)
    ax.set_yticklabels(ranked["label"], fontsize = 7.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("EB Progression Rate")
    ax.set_title(f"Top {top_n} Programs per Strategic Profile", fontweight = "bold", pad = 8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    present = ranked["strategic_profile"].unique()
    patches = [
        mpatches.Patch(color = _PROFILE_COLORS.get(p, "#9E9E9E"), label = p.replace("_", " ").title())
        for p in _PROFILE_COLORS if p in present
    ]
    ax.legend(handles = patches, fontsize = 7.5, loc = "upper right", framealpha = 0.85)

    plt.tight_layout()
    plt.savefig(_OUTPUT_DIR / "top_programs.png", dpi = 180, bbox_inches = "tight")
    plt.close()
    logging.info("Top programs plot saved!")