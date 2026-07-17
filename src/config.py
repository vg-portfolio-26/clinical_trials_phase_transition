VALID_STATUSES = [
    "completed",
    "terminated",
    "withdrawn",
    "suspended"
]

INCLUDED_PHASES = [
    "phase1",
    "phase1/phase2",
    "phase2",
    "phase2/phase3",
    "phase3"
]

TARGET_PHASES = [
    "phase1",
    "phase1/phase2",
    "phase2",
    "phase2/phase3"
]

PHASE_RANK = {
    "phase1": 0, "phase1/phase2": 1,
    "phase2": 2, "phase2/phase3": 3, "phase3": 4,
}

DEFAULT_MAX_GAP = 3650

REFERENCE_DATE = "2026-06-01"

PREPROCESS_FILTERS = [
    {
        "column": "study_type",
        "step_name": "Study type filter",
        "allowed_values": ["interventional"],
    },
    {
        "column": "overall_status",
        "step_name": "Status filter",
        "allowed_values": VALID_STATUSES,
    },
    {
        "column": "phase",
        "step_name": "Phase filter",
        "allowed_values": INCLUDED_PHASES,
    }
]

SPELLING_MAP = {
    r"\bhaemato": "hemato",
    r"\bhaemo": "hemo",
    r"\bhaem\b": "heme",
    r"\banaemia\b": "anemia",
    r"\banaemic\b": "anemic",
    r"\bleukaemia\b": "leukemia",
    r"\bleukaemic\b": "leukemic",
    r"\boedema\b": "edema",
    r"\boesopha": "esopha",
    r"\bischaemi": "ischemi",
    r"\bfoetal\b": "fetal",
    r"\bfoetus\b": "fetus",
    r"\bremodelling\b": "remodeling",
    r"\bmodelling\b": "modeling",
    r"\btumour\b": "tumor",
    r"\btumours\b": "tumors",
    r"\bbehaviour\b": "behavior",
    r"\bbehavioural\b": "behavioral",
    r"\bgynaeco": "gyneco",
    r"\bpaediat": "pediat",
    r"\borthopaed": "orthoped",
    r"\blabour\b": "labor",
    r"\bcolour\b": "color",
    r"\bcolours\b": "colors",
    r"\bfibre\b": "fiber",
    r"\bcentre\b": "center",
}

REQUIRED_CTGOV_COLUMNS = {
    "studies": [
        "nct_id",
        "phase",
        "enrollment",
        "study_type",
        "overall_status",
        "start_date",
        "completion_date",
        "primary_completion_date",
        "number_of_arms",
        "has_dmc",
        "is_fda_regulated_drug",
        "is_fda_regulated_device",
        "enrollment_type",
        "results_first_posted_date",
    ],
    "interventions": [
        "nct_id",
        "intervention_type",
        "name"
    ],
    "browse": [
        "nct_id",
        "mesh_term",
        "mesh_type",
    ]
}

REQUIRED_NLM_MAP_COLUMNS = [
    "term",
    "mesh_id",
    "mesh_descriptor",
    "match_type",
    "tree_number",
    "mesh_tree_group"
]

REQUIRED_RXNORM_COLUMNS = [
    "drug_name",
    "rxcui",
    "tty"
]