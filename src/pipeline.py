import time
from .helpers import create_pipeline_folders, setup_logging, log_separator, write_metadata

from .load_data import (
    load_ctgov_tables,
    validate_ctgov_tables,
    get_nlm_mesh_descriptors,
    build_mesh_mapping_table,
    validate_nlm_mesh_map,
    load_rxnorm_vocabulary
)

from .preprocess_data import (
    preprocess_studies,
    preprocess_interventions,
    preprocess_browse,
    build_canonical_mesh_mapping,
    preprocess_canonical_mesh_mapping,
    add_tree_info_to_browse,
    preprocess_rxnorm_vocabulary
)

from .postprocess_data import (
    build_master_datasets,
    compute_empirical_max_gap,
    build_survival_dataset
)

from .analysis import (
    compute_phase_eb_priors,
    shrink_pairs_by_phase,
    shrink_tree_groups_by_phase,
    compute_strategic_profiles,
    fit_km_curves_by_phase,
    fit_km_curves_by_phase_disease,
    compute_km_progression_probability,
    survival_dataset_diagnostics
)

from .plotting import (
    plot_km_curves,
    plot_disease_opportunity_map,
    plot_top_programs
)


def main():
    create_pipeline_folders()
    setup_logging()
    start_time = time.perf_counter()
    log_separator("Starting pipeline")

    try:
        ctgov_data_raw, external_data = get_raw_data()
        preprocessed_data = preprocess_data(ctgov_data_raw, external_data)
        survival_df = post_process_data(preprocessed_data)
        pair_scores, tree_scores = analysis(survival_df)
        write_metadata(ctgov_data_raw, external_data, preprocessed_data, survival_df, pair_scores, tree_scores, start_time = start_time, status = "completed")
        log_separator("Pipeline completed successfully!")

    except Exception as exc:
        write_metadata(None, None, start_time = start_time, status = "failed", error = str(exc))
        raise


def get_raw_data():
    ctgov_data_raw = load_ctgov_tables()
    validate_ctgov_tables(ctgov_data_raw)
    nlm_mesh_map = build_mesh_mapping_table(get_nlm_mesh_descriptors("Zipters/nlm-mesh-raw-desc2026"))
    validate_nlm_mesh_map(nlm_mesh_map)
    rxnorm_vocabulary = load_rxnorm_vocabulary()
    external_data = {"nlm_mesh_map": nlm_mesh_map, "rxnorm_vocabulary": rxnorm_vocabulary}
    log_separator()

    return ctgov_data_raw, external_data


def preprocess_data(ctgov_data_raw, external_data) -> dict:
    studies_preprocessed = preprocess_studies(ctgov_data_raw["studies"])
    rxnorm_vocabulary_preprocessed = preprocess_rxnorm_vocabulary(external_data["rxnorm_vocabulary"])
    interventions_preprocessed = preprocess_interventions(ctgov_data_raw["interventions"], rxnorm_vocabulary_preprocessed)
    browse_preprocessed = preprocess_browse(ctgov_data_raw["browse"])
    nlm_mesh_map_canonical = build_canonical_mesh_mapping(external_data["nlm_mesh_map"])
    nlm_mesh_map_canonical_preprocessed = preprocess_canonical_mesh_mapping(nlm_mesh_map_canonical)
    browse_preprocessed_tree = add_tree_info_to_browse(browse_preprocessed, nlm_mesh_map_canonical_preprocessed)
    log_separator()

    return {
        "studies_preprocessed": studies_preprocessed,
        "interventions_preprocessed": interventions_preprocessed,
        "browse_preprocessed_tree": browse_preprocessed_tree
    }


def post_process_data(preprocessed_data):
    master_dataset_programs = build_master_datasets(preprocessed_data)
    max_gap = compute_empirical_max_gap(master_dataset_programs)
    survival_df = build_survival_dataset(master_dataset_programs, max_gap = max_gap)
    log_separator()
    
    return survival_df
 
 
def analysis(survival_df):
    # Survival modelling
    phase_curves = fit_km_curves_by_phase(survival_df)
    tree_curves  = fit_km_curves_by_phase_disease(survival_df, min_events = 10)
    scored_survival_df = compute_km_progression_probability(survival_df, phase_curves, tree_curves)
    survival_dataset_diagnostics(scored_survival_df)
 
    # EB shrinkage & rankings
    priors = compute_phase_eb_priors(scored_survival_df)

    pair_scores = shrink_pairs_by_phase(scored_survival_df, priors)
    pair_scores = compute_strategic_profiles(pair_scores, level = "pairs")
 
    tree_scores = shrink_tree_groups_by_phase(scored_survival_df, priors)
    tree_scores = compute_strategic_profiles(tree_scores, level = "tree")
    log_separator()
 
    # Plots
    plot_km_curves(phase_curves)
    plot_disease_opportunity_map(tree_scores)
    plot_top_programs(pair_scores, scored_survival_df)
    log_separator()
 
    return pair_scores, tree_scores


if __name__ == "__main__":
    main()