<p style="font-size: 2em; font-weight: 700; margin-bottom: 0;">
  Clinical Trials Phase Transition
</p>
<p style="font-size: 1.5em; font-weight: 600; margin-top: 0;">
  A Survival-Analytic Approach on Public Registry Data
</p>

#

A self-contained data engineering and statistical analytics pipeline that transforms raw NLM ClinicalTrials.gov registrations into calibrated drug-development progression probabilities and actionable investment signals using a survival analysis framework.

The complete methodology, including the statistical reasoning behind each design choice, a full discussion of the results, and an explicit treatment of the project's limitations and assumptions, is presented in Clinical_trials_phase_transition.pdf, located in the report/ folder. 
A one page report can also be found in the same folder.

#### Keywords: 
Clinical trial progression modeling; Survival analysis; Empirical Bayes shrinkage; Public data

---

## Pipeline Architecture

```
NLM ClinicalTrials.gov       NLM MeSH XML              NLM RxNorm
(HuggingFace)               (HuggingFace)          (local cached CSV)
      │                           │                        │
      ▼                           ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · load_data.py          Data Ingestion             │
│  583k studies · 986k interventions · 4.2M browse terms      │
│  298k MeSH descriptors · 9.6k RxNorm entries                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 · preprocess_data.py    Preprocessing              │
│  Interventional drug studies · Phase 1–3 · 2000–2025        │
│  Drug → RxNorm (exact + RapidFuzz fuzzy, 64.4% coverage)    │
│  Disease → MeSH tree (99.7% study coverage)                 │
│  Output: 120,674 studies · 204,289 interventions            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 · postprocess_data.py   Program Assembly           │
│  Aggregate studies → 67,346 (drug, disease, phase) programs │
│  Empirical max_gap calibration (p90, per-phase)             │
│  Build survival dataset: duration + event, no TTP cliff     │
│  Output: 51,209 programs · 8,580 events (16.8%)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 · analysis.py           Survival + EB Scoring      │
│  Kaplan-Meier per phase + per (phase × disease area)        │
│  km_progression_prob = 1 − S(t)                             │
│  Empirical Bayes shrinkage → eb_rate + 95% credible CI      │
│  Strategic profiles: ranking on eb_rate, n_trials and CI    |
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5 · plotting.py           Business Report Plots      │
│  1: KM Curves — cumulative progression by phase             │
│  2: Disease Opportunity Map — phase × disease bubble        │
│  3: Top Programs — top 10 per strategic profile by eb_rate  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Source | Access | Description |
|---|---|---|
| NLM ClinicalTrials.gov | HuggingFace (`Zipters/aact_ctgov_*`) | Studies, interventions, conditions |
| NLM MeSH 2026 | HuggingFace (`Zipters/nlm-mesh-raw-desc2026`) | Disease ontology XML |
| NLM RxNorm | Cached local CSV | Drug name → RXCUI mapping |

---

## Output

| File | Description |
|---|---|
| `04_analysis/shrinked_pairs.csv` | EB rates per (drug, disease, phase) |
| `04_analysis/shrinked_trees.csv` | EB rates per (disease category, phase) |
| `04_analysis/strategic_profile_pairs.csv` | Pair rates + strategic profiles |
| `04_analysis/strategic_profile_tree.csv` | Tree rates + strategic profiles |
| `05_visualize_results/plot1_km_curves.png` | Phase progression curves |
| `05_visualize_results/plot2_disease_map.png` | Disease opportunity map |
| `05_visualize_results/top_programs_data.csv` | Top programs per strategic profile |
| `run_metadata.json` | Pipeline version, timestamps, key counts |

---

## Setup

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repository
git clone https://github.com/vg-portfolio-26/clinical_trials_phase_transition.git
cd clinical_trials_phase_transition

# 2. Build the image
docker compose build

# 3. Run the pipeline
docker compose up
```

A full run takes approximately 15 minutes, dominated by the RapidFuzz fuzzy matching step (~7 min for 170k unmatched drug names).

Python dependencies (managed inside the container):
Python 3.11 · pandas 2.2.3 · lifelines 0.30.3 · rapidfuzz · scipy · matplotlib · seaborn · datasets · huggingface-hub

---

## Project Structure

```
cached_data/
└── rxnorm_vocabulary.csv        # Cached rxnorm vocabulary

scripts/
├── create_rxnorm_vocabulary.py  # Script to create and cache the rxnorm vocabulary
└── run_pipeline.py              # Pipeline entrypoint

src/
├── pipeline.py                  # Orchestration
├── config.py                    # Filters, column requirements, phase definitions
├── load_data.py                 # Stage 1: data ingestion and validation
├── preprocess_data.py           # Stage 2: filtering, matching, MeSH resolution
├── postprocess_data.py          # Stage 3: program assembly, survival dataset
├── analysis.py                  # Stage 4: KM curves, EB shrinkage, scoring
├── plotting.py                  # Stage 5: business report plots
└── helpers.py                   # Logging, normalization, folder utilities
```

---

## License

Repo and content not licensed for use or redistribution.