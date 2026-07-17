# Clinical Trials Phase Transition: A Survival-Analytic Approach Using Public Registry Data

A self-contained data engineering and statistical analytics pipeline that transforms raw NLM ClinicalTrials.gov registrations into calibrated drug-development progression probabilities and actionable investment signals using a survival analysis framework.

The complete methodology, including the statistical reasoning behind each design choice, a full discussion of the results, and an explicit treatment of the project's limitations and assumptions, is presented in Clinical_Trials_Phase_Transition.pdf, located in the report/ folder.
A 1 page report can also be found in the same folder.

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
│  km_progression_prob = 1 − S(t) — continuous soft label     │
│  Empirical Bayes shrinkage → eb_rate + 95% credible CI      │
│  Programs ranked by eb_rate; profiles use n_trials + CI     │
│  Strategic profiles: HIGH_CONFIDENCE · HIGH_RISK · etc.     │
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
| NLM ClinicalTrials.gov (AACT) | HuggingFace (`Zipters/aact_ctgov_*`) | Studies, interventions, conditions |
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
| `05_visualize_results/plot3_top_programs.png` | Top 10 programs per strategic profile, ranked by eb_rate |
| `run_metadata.json` | Pipeline version, timestamps, key counts |

### Strategic Profiles

| Profile | Condition | Interpretation |
|---|---|---|
| HIGH_CONFIDENCE | High rate, narrow CI, n > 3 | Act — strong signal, well evidenced |
| HIGH_RISK_HIGH_REWARD | High rate, wide CI, n > 3 | Investigate — upside present, verify |
| STABLE_BUT_MODEST | Low rate, narrow CI, n > 3 | Monitor — reliably below phase median |
| EMERGING_SIGNAL | High rate, n ≤ 3 | Watch — promising but prior-dominated |
| STANDARD | Everything else | Background — no actionable signal yet |

---

## Setup

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repository
git clone <repo>
cd <repo>

# 2. Build the image
docker compose build

# 3. Run the pipeline
docker compose up
```

A full run takes approximately 15 minutes, dominated by the RapidFuzz fuzzy matching step (~7 min for 170k unmatched drug names).

Python dependencies (managed inside the container): Python 3.11 · pandas 2.2.3 · lifelines 0.30.3 · rapidfuzz · scipy · matplotlib · seaborn · datasets · huggingface-hub

---

## Project Structure

```
src/
├── pipeline.py          # Orchestration
├── config.py            # Filters, column requirements, phase definitions
├── load_data.py         # Stage 1: data ingestion and validation
├── preprocess_data.py   # Stage 2: filtering, matching, MeSH resolution
├── postprocess_data.py  # Stage 3: program assembly, survival dataset
├── analysis.py          # Stage 4: KM curves, EB shrinkage, scoring
├── plotting.py          # Stage 5: business report plots
└── helpers.py           # Logging, normalization, folder utilities
```

---

## License

Repo and content not licensed for use or redistribution.