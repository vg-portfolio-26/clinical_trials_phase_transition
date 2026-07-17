import os
import re
import sys
import json
import time
import shutil
import logging
import warnings
import unicodedata
import pandas as pd
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from .config import SPELLING_MAP, DEFAULT_MAX_GAP, TARGET_PHASES, INCLUDED_PHASES

def create_pipeline_folders():
    folders = [
        "output",
        "output/00_logs",
        "output/01_raw_data",
        "output/02_preprocessed_data",
        "output/03_postprocessed_data",
        "output/04_analysis",
        "output/05_visualize_results",
        "output/99_src"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok = True)

    shutil.copytree("src", "output/99_src", dirs_exist_ok = True, 
        ignore = shutil.ignore_patterns("__pycache__","*.pyc")
    )


def normalize_string(term: str, remove_all_spaces: bool = False) -> str:

    if pd.isna(term) or term == "":
        return None

    output = str(term).strip().lower()
    output = unicodedata.normalize("NFKD", output).encode("ascii", "ignore").decode("ascii")

    output = re.sub(r"[–—]", "-", output)

    for pattern, target in SPELLING_MAP.items():
         output = re.sub(pattern, target, output)

    if remove_all_spaces:
        output = re.sub(r"\s+", " ", output).strip()
    else:
        output = output.strip()

    return output


def setup_logging():
    log_filename = (f"output/00_logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    try:
        sys.stdout.reconfigure(encoding = "utf-8", errors = "backslashreplace")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding = "utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force = True
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("weasyprint").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message = ".*You are sending unauthenticated requests to the HF Hub.*")


def log_separator(title=None):
    logging.info("=" * 80)

    if title:
        logging.info(title)
        logging.info("=" * 80)


def log_step(initial_n, step_name, before, after):
    step_drop = (before - after) / before if before > 0 else 0
    total_drop = (initial_n - after) / initial_n if initial_n > 0 else 0
    logging.info(f"{step_name}: {after} rows (-{step_drop:.2%} step, -{total_drop:.2%} total)")


def _row_count(df):
    return len(df) if df is not None else None


def _summary_for_table(raw_df, preprocessed_df, label):
    raw_rows = _row_count(raw_df)
    preprocessed_rows = _row_count(preprocessed_df)

    if raw_rows is None or raw_rows == 0:
        retention_pct = None
    else:
        retention_pct = round(preprocessed_rows / raw_rows, 4) if preprocessed_rows is not None else None

    return {
        "raw_rows": raw_rows,
        "preprocessed_rows": preprocessed_rows,
        "retention_pct": retention_pct,
        "label": label,
    }


def _artifact_row_count(output_dir, relative_path):
    path = Path(output_dir) / relative_path
    if not path.exists():
        return None

    try:
        return len(pd.read_csv(path))
    except Exception:
        return None


def write_metadata(
    ctgov_data_raw,
    external_data,
    preprocessed_data,
    survival_df=None,
    pair_scores=None,
    tree_scores=None,
    start_time=None,
    status = "completed",
    error = None,
    output_dir = "output",
):
    output_dir = Path(output_dir)
    output_path = output_dir / "run_metadata.json"

    raw_studies = ctgov_data_raw.get("studies") if ctgov_data_raw else None
    raw_interventions = ctgov_data_raw.get("interventions") if ctgov_data_raw else None
    raw_browse = ctgov_data_raw.get("browse") if ctgov_data_raw else None
    nlm_mesh_map = external_data.get("nlm_mesh_map") if external_data else None
    rxnorm_vocabulary = external_data.get("rxnorm_vocabulary") if external_data else None

    preprocessed_studies = preprocessed_data.get("studies_preprocessed") if preprocessed_data else None
    preprocessed_interventions = preprocessed_data.get("interventions_preprocessed") if preprocessed_data else None
    preprocessed_browse = preprocessed_data.get("browse_preprocessed_tree") if preprocessed_data else None

    runtime_seconds = None
    if start_time is not None:
        runtime_seconds = round(time.perf_counter() - start_time, 2)

    interventions_match_rate = None
    if preprocessed_interventions is not None and len(preprocessed_interventions) > 0:
        interventions_match_rate = round(preprocessed_interventions["rxcui"].notna().mean(), 4)

    mesh_study_coverage = None
    mesh_term_coverage = None
    if raw_browse is not None and preprocessed_browse is not None and len(raw_browse) > 0:
        raw_study_count = raw_browse["nct_id"].nunique() if "nct_id" in raw_browse.columns else None
        preprocessed_study_count = preprocessed_browse["nct_id"].nunique() if "nct_id" in preprocessed_browse.columns else None
        raw_term_count = raw_browse["mesh_term"].nunique() if "mesh_term" in raw_browse.columns else None
        preprocessed_term_count = preprocessed_browse["mesh_term"].nunique() if "mesh_term" in preprocessed_browse.columns else None

        if raw_study_count not in (None, 0):
            mesh_study_coverage = round(preprocessed_study_count / raw_study_count, 4) if preprocessed_study_count is not None else None
        if raw_term_count not in (None, 0):
            mesh_term_coverage = round(preprocessed_term_count / raw_term_count, 4) if preprocessed_term_count is not None else None

    survival_programs = _row_count(survival_df) if survival_df is not None else None
    survival_event_count = int(survival_df["event"].sum()) if survival_df is not None and "event" in survival_df.columns else None
    survival_event_rate = None
    if survival_programs and survival_event_count is not None:
        survival_event_rate = round(survival_event_count / survival_programs, 4)

    metadata = {
        "pipeline": {
            "name": "clinical_trial_simulator",
            "version": "dev",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "runtime_seconds": runtime_seconds,
        },
        "inputs": {
            "raw_studies": _row_count(raw_studies),
            "raw_interventions": _row_count(raw_interventions),
            "raw_browse": _row_count(raw_browse),
            "nlm_mesh_map": _row_count(nlm_mesh_map),
            "rxnorm_vocabulary": _row_count(rxnorm_vocabulary),
        },
        "config": {
            "included_phases": INCLUDED_PHASES,
            "target_phases": TARGET_PHASES,
            "default_max_gap_days": DEFAULT_MAX_GAP,
        },
        "preprocessing": {
            "studies": _summary_for_table(raw_studies, preprocessed_studies, "studies"),
            "interventions": _summary_for_table(raw_interventions, preprocessed_interventions, "interventions"),
            "browse": _summary_for_table(raw_browse, preprocessed_browse, "browse"),
        },
        "quality_metrics": {
            "rxnorm_match_rate": interventions_match_rate,
            "mesh_study_coverage": mesh_study_coverage,
            "mesh_term_coverage": mesh_term_coverage,
        },
        "analysis": {
            "program_count": survival_programs,
            "event_count": survival_event_count,
            "event_rate": survival_event_rate,
            "phase_counts": None,
            "strata_count": None,
            "pair_score_rows": _row_count(pair_scores),
            "tree_score_rows": _row_count(tree_scores),
        },
        "outputs": {
            "artifacts": [
                {
                    "path": "02_preprocessed_data/studies_preprocessed.csv",
                    "rows": _artifact_row_count(output_dir, "02_preprocessed_data/studies_preprocessed.csv"),
                },
                {
                    "path": "02_preprocessed_data/interventions_preprocessed.csv",
                    "rows": _artifact_row_count(output_dir, "02_preprocessed_data/interventions_preprocessed.csv"),
                },
                {
                    "path": "02_preprocessed_data/browse_preprocessed_tree.csv",
                    "rows": _artifact_row_count(output_dir, "02_preprocessed_data/browse_preprocessed_tree.csv"),
                },
                {
                    "path": "03_postprocessed_data/master_dataset_programs.csv",
                    "rows": _artifact_row_count(output_dir, "03_postprocessed_data/master_dataset_programs.csv"),
                },
                {
                    "path": "03_postprocessed_data/survival_dataset.csv",
                    "rows": _artifact_row_count(output_dir, "03_postprocessed_data/survival_dataset.csv"),
                },
                {
                    "path": "04_analysis/shrinked_pairs.csv",
                    "rows": _artifact_row_count(output_dir, "04_analysis/shrinked_pairs.csv"),
                },
                {
                    "path": "04_analysis/shrinked_trees.csv",
                    "rows": _artifact_row_count(output_dir, "04_analysis/shrinked_trees.csv"),
                },
            ]
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "pandas": importlib_metadata.version("pandas"),
            "numpy": importlib_metadata.version("numpy"),
            "rapidfuzz": importlib_metadata.version("rapidfuzz"),
            "lifelines": importlib_metadata.version("lifelines"),
        },
    }

    if survival_df is not None and len(survival_df) > 0:
        metadata["analysis"]["phase_counts"] = survival_df["phase"].value_counts().to_dict()
        metadata["analysis"]["strata_count"] = len(survival_df.groupby(["phase", "mesh_tree_group"]))

    if error is not None:
        metadata["pipeline"]["error"] = str(error)

    output_dir.mkdir(parents = True, exist_ok = True)
    with open(output_path, "w", encoding = "utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    logging.info(f"Metadata successfully written!")