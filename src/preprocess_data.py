import re
import logging
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
from .config import PREPROCESS_FILTERS
from .helpers import normalize_string, log_step, log_separator

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "02_preprocessed_data"


def preprocess_studies(studies: pd.DataFrame):
    logging.info("Preprocessing studies table...")
 
    initial_n = len(studies)
    logging.info(f"Initial rows: {initial_n}")
 
    studies["study_type"] = studies["study_type"].apply(normalize_string, remove_all_spaces = False)
    studies["overall_status"] = studies["overall_status"].apply(normalize_string, remove_all_spaces = False)
    studies["is_terminated_or_withdrawn"] = studies["overall_status"].isin(["terminated", "withdrawn"])
    studies["phase"] = studies["phase"].apply(normalize_string, remove_all_spaces = True)
    studies = studies.drop_duplicates()
    log_step(initial_n, f"Rows after dropping duplicates", initial_n, len(studies))
 
    for filter_spec in PREPROCESS_FILTERS:
        before = len(studies)
        col = filter_spec["column"]
        studies = studies[studies[col].isin(filter_spec["allowed_values"])]
        log_step(initial_n, filter_spec["step_name"], before, len(studies))
 
    before = len(studies)
    studies["start_date"] = pd.to_datetime(studies["start_date"], errors = "coerce")
    studies["start_year"] = studies["start_date"].dt.year
    studies = studies[studies["start_year"].between(2000, 2025)]
    studies.drop("start_year", axis = 1, inplace = True)
    log_step(initial_n, "Temporal cohort filter (2000-2025)", before, len(studies))
 
    studies["completion_date"] = pd.to_datetime(studies["completion_date"], errors = "coerce")
    studies["primary_completion_date"] = pd.to_datetime(studies["primary_completion_date"], errors = "coerce")
    studies["results_first_posted_date"] = pd.to_datetime(studies["results_first_posted_date"], errors = "coerce")
    studies["effective_completion_date"] = studies["primary_completion_date"].fillna(studies["completion_date"])

    studies["has_results_after_completion"] = (
        studies["results_first_posted_date"].notna()
        & studies["effective_completion_date"].notna()
        & (
            studies["results_first_posted_date"]
            >= studies["effective_completion_date"]
        )
    )

    studies["completed_with_evidence"] = (
        studies["overall_status"].eq("completed")
        & (
            studies["has_results_after_completion"]
            | studies["effective_completion_date"].notna()
        )
    ).astype(int)
 
    n_status_completed = studies["overall_status"].eq("completed").sum()
    n_completed_with_evidence = int(studies["completed_with_evidence"].sum())
    n_excluded_no_results = n_status_completed - n_completed_with_evidence

    logging.info("Completion audit:")
    logging.info(f"overall_status = completed: {n_status_completed:,}")
    logging.info(f"completed_with_evidence = True: {n_completed_with_evidence:,}")
    logging.info(f"excluded (no completion evidence): {n_excluded_no_results:,} ({n_excluded_no_results / n_status_completed:.1%} of completed studies)")
    logging.info(f"Final dataset: {len(studies)} rows (-{(initial_n - len(studies)) / initial_n:.2%} total loss)")

    studies.to_csv(_OUTPUT_DIR / "studies_preprocessed.csv", index = False)
    logging.info("Studies preprocessed successfully!")
    log_separator()

    return studies


def preprocess_rxnorm_vocabulary(rxnorm_vocabulary: pd.DataFrame):
    logging.info("Preprocessing rxnorm vocabulary...")

    initial_n = len(rxnorm_vocabulary)
    logging.info(f"Initial rows: {initial_n}")

    rxnorm_vocabulary["rxcui"] = rxnorm_vocabulary["rxcui"].astype(int).astype(str).str.replace(r"^", "RX", regex = True)
    rxnorm_vocabulary["drug_name"] = rxnorm_vocabulary["drug_name"].apply(normalize_string, remove_all_spaces = False)
    
    before = len(rxnorm_vocabulary)
    rxnorm_vocabulary = rxnorm_vocabulary.drop_duplicates()
    log_step(initial_n, f"Rows after dropping duplicates", before, len(rxnorm_vocabulary))

    rxnorm_vocabulary.to_csv(_OUTPUT_DIR / "rxnorm_vocabulary_preprocessed.csv", index = False)
    logging.info("rxnorm vocabulary preprocessed successfully!")
    log_separator()

    return rxnorm_vocabulary


def preprocess_interventions(interventions, rxnorm_vocabulary_preprocessed):
    logging.info("Preprocessing interventions table...")

    initial_n = len(interventions)
    logging.info(f"Initial rows: {initial_n}")

    before = len(interventions)
    interventions["intervention_type"] = interventions["intervention_type"].apply(normalize_string, remove_all_spaces = True)
    interventions["name"] = interventions["name"].apply(normalize_string,remove_all_spaces = False)
    interventions.rename(columns = {"name": "drug_name"}, inplace = True)
    interventions = interventions.drop_duplicates()
    log_step(initial_n, f"Rows after dropping duplicates", before, len(interventions))

    before = len(interventions)
    interventions = interventions[interventions["intervention_type"] == "drug"]
    log_step(initial_n, f"Rows after filtering for intervention_type = drug", before, len(interventions))

    before = len(interventions)
    interventions = interventions[~interventions["drug_name"].astype(str).str.fullmatch(r"\d+")]
    log_step(initial_n, f"Rows after filtering for numeric drug names", before, len(interventions))

    before = len(interventions)
    pattern = re.compile(r"placebo|control|sham|observation|cohort", re.I)
    interventions = interventions[~interventions["drug_name"].str.contains(pattern, na = False)]
    log_step(initial_n, f"Rows after filtering for non drug names", before, len(interventions))

    before = len(interventions)
    pattern = re.compile(r",|\+|\band\b|\bwith\b|\bcombination\b|\bmixture\b|\bor\b", re.I)
    interventions = interventions[~interventions["drug_name"].str.contains(pattern, na = False)]
    log_step(initial_n, f"Rows after filtering for non-deterministic drug names", before, len(interventions))
    logging.info(f"Intervention table shape: {interventions.shape}")

    unique_drugs = (
        interventions["drug_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .drop_duplicates()
            .tolist()
        )
    pd.DataFrame({"drug_name": unique_drugs}).to_csv(_OUTPUT_DIR / "unique_drugs.csv", index = False)

    before = len(interventions)
    logging.info("Adding rxnorm info via exact match...")
    interventions = interventions.merge(rxnorm_vocabulary_preprocessed, on = "drug_name", how = "left")
    n_drug_rxcui = interventions['rxcui'].notna().sum()
    logging.info(f"Number of drug names with exact matched rxcui value: {n_drug_rxcui} ({n_drug_rxcui / before:.2%})")

    interventions["matched_exact"] = interventions["rxcui"].notna()
    interventions["fuzzy_match_score"] = pd.NA
    interventions = fuzzy_match_interventions(interventions, rxnorm_vocabulary_preprocessed)
    interventions = interventions.dropna(subset=["rxcui"])
    logging.info(f"Final number of drug names with rxcui value: {len(interventions)} ({len(interventions) / before:.2%})")

    mask = interventions["matched_exact"] & interventions["candidate_drug_name"].isna()
    interventions.loc[mask, "candidate_drug_name"] = interventions.loc[mask, "drug_name"]
    
    logging.info(f"Final intervention table after removing rxcui NA values: {interventions.shape}")
    interventions.to_csv(_OUTPUT_DIR / "interventions_preprocessed.csv", index = False)
    logging.info(f"interventions preprocessed successfully!")
    log_separator()

    return interventions


def preprocess_browse(browse: pd.DataFrame):
    logging.info("Preprocessing browse table...")

    initial_n = len(browse)
    logging.info(f"Initial rows: {initial_n}")

    browse["mesh_term"] = browse["mesh_term"].apply(normalize_string,remove_all_spaces = False)
    browse = browse.drop_duplicates()
    log_step(initial_n, f"Rows after dropping duplicates", initial_n, len(browse))

    browse.to_csv(_OUTPUT_DIR / "browse_preprocessed.csv", index = False)
    logging.info(f"Browse preprocessed successfully!")
    log_separator()

    return browse


def build_canonical_mesh_mapping(mesh_mapping: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building canonical MeSH mapping table...")

    df = mesh_mapping.copy()
    initial_n = len(df)
    initial_terms = df["term"].nunique()

    df["term"] = df["term"].astype(str).str.strip().str.lower()

    priority_order = ["descriptor", "synonym"]
    df["match_priority"] = df["match_type"].apply(
        lambda x: priority_order.index(x)
        if x in priority_order else len(priority_order)
    )

    df = (
        df.sort_values(["term", "match_priority"])
        .drop_duplicates(subset = ["term"], keep = "first")
        .reset_index(drop = True)
    )

    final_terms = df["term"].nunique()

    logging.info(f"Initial rows: {initial_n:,}")
    log_step(initial_n, f"Rows after dropping duplicates", initial_n, len(df))
    logging.info(f"Initial unique terms: {initial_terms:,}")
    logging.info(f"Final unique terms: {final_terms:,}")

    if final_terms != initial_terms:
        logging.warning("Ontology ambiguity resolved via priority rule")

    canonical_cols = ["term", "mesh_id", "mesh_descriptor", "tree_number", "mesh_tree_group"]
    canonical = df[canonical_cols].drop_duplicates()

    canonical.to_csv(_OUTPUT_DIR / "mesh_mapping_canonical.csv", index = False)
    logging.info("Canonical MeSH mapping table created!")
    log_separator()

    return canonical


def preprocess_canonical_mesh_mapping(df: pd.DataFrame):
    logging.info("Preprocessing MeSH map table...")

    initial_n = len(df)
    logging.info(f"Initial rows: {initial_n}")

    df["term"] = df["term"].apply(normalize_string,remove_all_spaces = False)

    df = df.drop_duplicates()
    log_step(initial_n, f"Rows after dropping duplicates", initial_n, len(df))

    df.to_csv(_OUTPUT_DIR / "mesh_mapping_canonical_preprocessed.csv", index = False)
    logging.info(f"MeSH map table preprocessed successfully!")
    log_separator()

    return df


def add_tree_info_to_browse(browse: pd.DataFrame, mesh_mapping: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Adding MeSH tree data to browse terms...")

    df = browse.copy()
    logging.info(f"Initial browse table shape: {df.shape}")

    required_cols = {"nct_id", "mesh_term"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required browse columns: {missing}")

    mapped_df = df.merge(
        mesh_mapping,
        left_on = "mesh_term",
        right_on = "term",
        how = "left"
    )

    mapped_df = mapped_df[mapped_df["mesh_id"].notna()]
    log_step(len(df), f"Rows after dropping missing mesh_id", len(df), len(mapped_df))

    before = len(mapped_df)
    mapped_df["tree_depth"] = (mapped_df["tree_number"].fillna("").str.count(r"\."))
    has_mesh_list = mapped_df.groupby("nct_id")["mesh_type"].transform(lambda x: (x == "mesh-list").any())

    mapped_df_filtered = pd.concat([
        mapped_df[has_mesh_list & (mapped_df["mesh_type"] == "mesh-list")],
        mapped_df[~has_mesh_list & (mapped_df["mesh_type"] == "mesh-ancestor")]
        .sort_values("tree_depth", ascending = False)
        .drop_duplicates("nct_id")
    ])

    mapped_df_filtered = (
        mapped_df_filtered
        .drop(columns = "tree_depth")
        .sort_values("nct_id")
        .reset_index(drop = True)
    )

    log_step(len(df), f"Rows after dropping mesh-ancestors duplicates", before, len(mapped_df_filtered))

    total_unique_terms = mapped_df_filtered["mesh_term"].nunique()
    unique_term_coverage = total_unique_terms / df['mesh_term'].nunique()
    mapped_unique_studies = mapped_df_filtered["nct_id"].nunique()
    study_coverage = mapped_unique_studies / df['nct_id'].nunique()

    logging.info(f"Final browse table shape: {mapped_df_filtered.shape}")
    logging.info(f"Unique browse terms with MeSH tree data: {total_unique_terms:,} ({unique_term_coverage:.2%} coverage)")
    logging.info(f"Unique studies with MeSH tree data: {mapped_unique_studies:,} ({study_coverage:.2%} coverage)")

    mapped_df_filtered.to_csv(_OUTPUT_DIR / "browse_preprocessed_tree.csv", index = False)
    logging.info("MeSH tree data added successfully to browse terms!")
    log_separator()

    return mapped_df_filtered


def fuzzy_match_interventions(interventions: pd.DataFrame, rxnorm_vocabulary_preprocessed: pd.DataFrame, threshold: float = 90):
    logging.info("Adding rxnorm info via RapidFuzz...")

    unmatched = interventions[interventions["rxcui"].isna()].copy()

    if unmatched.empty:
        logging.info("All interventions already have rxcui!")
        return interventions

    vocab_lookup = {
        name: row
        for name, row in zip(
            rxnorm_vocabulary_preprocessed["drug_name"],
            rxnorm_vocabulary_preprocessed.to_dict("records")
        )
    }

    vocab_names = list(vocab_lookup.keys())

    best_matches = []
    for drug_name in unmatched["drug_name"].dropna().unique():
        drug_name_norm = str(drug_name).strip().lower()

        if drug_name_norm in vocab_lookup:
            row = vocab_lookup[drug_name_norm]
            best_matches.append({
                "drug_name": drug_name,
                "candidate_drug_name": row["drug_name"],
                "rxcui": row["rxcui"],
                "tty": row.get("tty"),
                "fuzzy_match_score": 1.0,
            })
            continue

        match = process.extractOne(
            drug_name_norm,
            vocab_names,
            scorer = fuzz.token_set_ratio
        )

        if match is None:
            continue

        candidate_name, score, _ = match

        if score < threshold:
            continue

        vocab_row = vocab_lookup[candidate_name]

        best_matches.append({
            "drug_name": drug_name,
            "candidate_drug_name": vocab_row["drug_name"],
            "rxcui": vocab_row["rxcui"],
            "tty": vocab_row.get("tty"),
            "fuzzy_match_score": score / 100.0,
        })

    if not best_matches:
        logging.info("No fuzzy matches found!")
        return interventions

    best_df = pd.DataFrame(best_matches)

    interventions = interventions.merge(
        best_df,
        on = "drug_name",
        how = "left",
        suffixes = ("", "_fuzzy"),
    )

    mask = interventions["rxcui"].isna() & interventions["rxcui_fuzzy"].notna()

    interventions.loc[mask, "rxcui"] = interventions.loc[mask, "rxcui_fuzzy"]
    interventions.loc[mask, "tty"] = interventions.loc[mask, "tty_fuzzy"]
    interventions.loc[mask, "fuzzy_match_score"] = interventions.loc[mask, "fuzzy_match_score_fuzzy"]

    interventions = interventions.drop(columns=["rxcui_fuzzy", "tty_fuzzy", "fuzzy_match_score_fuzzy"], errors = "ignore")

    return interventions