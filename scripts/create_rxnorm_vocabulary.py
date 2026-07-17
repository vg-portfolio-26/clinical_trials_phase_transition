# %%
import sys
import math
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
from src.load_data import load_table
from src.helpers import normalize_string

INGREDIENT_TTYS = {"IN", "PIN", "MIN"}

def fetch_unique_drugs_names(interventions):

    interventions["intervention_type"] = interventions["intervention_type"].apply(normalize_string, remove_all_spaces = True)
    interventions["name"] = interventions["name"].apply(normalize_string,remove_all_spaces = False)
    interventions.rename(columns = {"name": "drug_name"}, inplace = True)
    interventions = interventions.drop_duplicates()
    interventions = interventions[interventions["intervention_type"] == "drug"]
    interventions = interventions[~interventions["drug_name"].astype(str).str.fullmatch(r"\d+")]
    pattern = re.compile(r"placebo|control|sham|observation|cohort", re.I)
    interventions = interventions[~interventions["drug_name"].str.contains(pattern, na = False)]
    pattern = re.compile(r",|\+|\band\b|\bwith\b|\bcombination\b|\bmixture\b|\bor\b", re.I)
    interventions = interventions[~interventions["drug_name"].str.contains(pattern, na = False)]

    unique_drugs = (
        interventions["drug_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .drop_duplicates()
            .tolist()
        )
    
    output_path = SCRIPT_ROOT / "cached_data" / "unique_drugs.csv"
    output_path.parent.mkdir(parents = True, exist_ok = True)
    pd.DataFrame({"drug_name": unique_drugs}).to_csv(output_path, index = False)

    return True

def normalize_rxcui(rxcui):
    try:
        return str(int(float(rxcui)))
    except (TypeError, ValueError):
        return None


def get_rxcui(drug_name):
    try:
        r = requests.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params = {"name": drug_name},
            timeout = 10,
        )
        r.raise_for_status()

        ids = r.json().get("idGroup", {}).get("rxnormId")
        return normalize_rxcui(ids[0]) if ids else None

    except Exception:
        return None


def get_tty_for_rxcui(rxcui):

    rxcui = normalize_rxcui(rxcui)
    if rxcui is None:
        return None

    try:
        r = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout = 10)
        r.raise_for_status()
        return r.json().get("properties", {}).get("tty")

    except Exception:
        return None
    

def normalize_vocab_columns(df):
    if df.empty:
        return df.copy()

    df = df.copy()

    if "drug_name" in df:
        df["drug_name"] = (
            df["drug_name"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "rxcui" in df:
        df["rxcui"] = pd.to_numeric(
            df["rxcui"],
            errors = "coerce",
        )
        df = df.dropna(subset=["rxcui"])
        df["rxcui"] = df["rxcui"].astype(int)

    if "tty" in df:
        df["tty"] = (
            df["tty"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def choose_preferred_rxcui(exact_df, synonym_df):

    exact = exact_df.dropna(subset = ["drug_name", "rxcui", "tty"]).copy()
    synonym = synonym_df.dropna(subset = ["drug_name", "rxcui", "tty"]).copy()
    exact["source_rank"] = 0
    synonym["source_rank"] = 1
    tty_rank = {"IN": 0, "PIN": 1, "MIN": 2}
    exact["tty_rank"] = exact["tty"].map(tty_rank).fillna(99)
    synonym["tty_rank"] = synonym["tty"].map(tty_rank).fillna(99)

    return (
        pd.concat([exact, synonym], ignore_index = True)
        .sort_values(
            ["drug_name", "source_rank", "tty_rank", "rxcui"],
            kind = "mergesort",
        )
        .drop_duplicates("drug_name")
        [["drug_name", "rxcui", "tty"]]
        .reset_index(drop = True)
    )


def get_all_names(rxcui):

    rxcui = normalize_rxcui(rxcui)
    if rxcui is None:
        return []

    rows = []
    seen = set()

    def add_name(name, tty):
        if not name:
            return

        name = name.strip().lower()
        key = (name, tty)

        if key not in seen:
            seen.add(key)
            rows.append({
                "drug_name": name,
                "rxcui": rxcui,
                "tty": tty,
            })

    # Related concepts
    try:
        r = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json", timeout = 10)
        r.raise_for_status()

        groups = (
            r.json()
            .get("allRelatedGroup", {})
            .get("conceptGroup", [])
        )

        for group in groups:
            for concept in group.get("conceptProperties", []):
                add_name(concept.get("name"), concept.get("tty"))

    except Exception:
        pass

    # Primary concept
    try:
        r = requests.get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json", timeout = 10)
        r.raise_for_status()
        props = r.json().get("properties", {})
        add_name(props.get("name"), props.get("tty"))

    except Exception:
        pass

    return rows

# %%
# Data
df = load_table("Zipters/aact_ctgov_interventions_raw")
fetch_unique_drugs_names(df)
df = pd.read_csv(SCRIPT_ROOT / "cached_data" / "unique_drugs.csv")
unique_drugs = df["drug_name"]
print(f"Unique drugs: {len(unique_drugs):,}")

# %%
# RxNorm lookup
CHECKPOINT_DIR = SCRIPT_ROOT / "rxnorm_checkpoints"
CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)

processed_drugs = set()

for file in CHECKPOINT_DIR.glob("rxnorm_*.csv"):
    try:
        df_tmp = pd.read_csv(file)
        
        if "drug_name" in df_tmp:
            processed_drugs.update(
                df_tmp["drug_name"]
                .dropna()
                .astype(str)
            )

    except Exception as e:
        print(f"Skipping {file}: {e}")

remaining_drugs = [
    drug for drug in unique_drugs
    if drug not in processed_drugs
]

print(f"Total unique drugs: {len(unique_drugs):,}")
print(f"Already processed: {len(processed_drugs):,}")
print(f"Remaining: {len(remaining_drugs):,}")

if not remaining_drugs:
    print("All drugs already processed.")
    raise SystemExit

chunk_size = math.ceil(len(unique_drugs) * 0.01)
print(f"Chunk size: {chunk_size:,}")

existing = list(CHECKPOINT_DIR.glob("rxnorm_*.csv"))
chunk_id = (
    max(int(f.stem.split("_")[1]) for f in existing) + 1
    if existing else 1
)

print(f"Starting from chunk {chunk_id:03d}")
chunk = []
for drug in tqdm(remaining_drugs):
    chunk.append({
        "drug_name": drug,
        "rxcui": get_rxcui(drug),
        "lookup_time": pd.Timestamp.now(),
    })
    time.sleep(0.1)

    if len(chunk) >= chunk_size:
        outfile = CHECKPOINT_DIR / f"rxnorm_{chunk_id:03d}.csv"
        pd.DataFrame(chunk).to_csv(outfile, index = False)
        print(f"Saved {outfile}")
        chunk.clear()
        chunk_id += 1

if chunk:
    outfile = CHECKPOINT_DIR / f"rxnorm_{chunk_id:03d}.csv"
    pd.DataFrame(chunk).to_csv(outfile, index = False)
    print(f"Saved {outfile}")

print("RxNorm lookup complete.")

# %%
# Exact vocabulary
files = sorted(CHECKPOINT_DIR.glob("rxnorm_*.csv"))
print(f"Found {len(files)} files")

vocab_exact = (
    pd.concat((pd.read_csv(f) for f in files), ignore_index = True)
    .dropna(subset = ["rxcui"])
    [["drug_name", "rxcui"]]
    .drop_duplicates()
)

vocab_exact["rxcui"] = pd.to_numeric(vocab_exact["rxcui"], errors = "coerce")

vocab_exact = (
    vocab_exact
    .dropna(subset = ["rxcui"])
    .astype({"rxcui": int})
)

print(f"Vocabulary size: {len(vocab_exact):,}")
vocab_exact.to_csv(SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact.csv", index = False)
print("Saved: rxnorm_vocabulary_exact.csv")

# %%
# TTY
VOCAB_PATH = SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact.csv"

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {VOCAB_PATH}. Run exact vocabulary pipeline first.")

vocab_exact = pd.read_csv(VOCAB_PATH)

vocab_exact["tty"] = [
    get_tty_for_rxcui(rxcui)
    for rxcui in tqdm(vocab_exact["rxcui"], desc = "Fetching tty")
]

print(vocab_exact.head())
print(f"Rows with tty info: {vocab_exact['tty'].notna().sum()}")
vocab_exact.to_csv(SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact_with_tty.csv", index = False)
print("Saved: rxnorm_vocabulary_exact_with_tty.csv")

vocab_exact_real = vocab_exact[vocab_exact["tty"].isin(INGREDIENT_TTYS)].copy()
print(f"Kept: {len(vocab_exact_real):,}")
vocab_exact_real.to_csv(SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact_real_drugs.csv", index = False)
print("Saved: rxnorm_vocabulary_exact_real_drugs.csv")

# %%
# Synonmn
VOCAB_PATH = SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact_real_drugs.csv"

if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {VOCAB_PATH}. Run exact vocabulary pipeline first.")

vocab_exact_real = pd.read_csv(VOCAB_PATH)

vocab_exact_real["rxcui"] = pd.to_numeric(
    vocab_exact_real["rxcui"], errors = "coerce"
).dropna().astype(int)

vocab_exact_real = vocab_exact_real[vocab_exact_real["tty"].isin(INGREDIENT_TTYS)].copy()
print(f"Loaded exact ingredient vocabulary: {len(vocab_exact_real):,}")

unique_rxcuis = vocab_exact_real["rxcui"].astype(str).unique()
print(f"Unique RXCUIs: {len(unique_rxcuis):,}")

CHECKPOINT_DIR = SCRIPT_ROOT / "rxnorm_synonym_checkpoints"
CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)

processed_rxcuis = set()
for file in CHECKPOINT_DIR.glob("rxnorm_synonyms_*.csv"):
    try:
        df = pd.read_csv(file)
        if "rxcui" in df.columns:
            processed_rxcuis.update(df["rxcui"].astype(str).unique())

    except Exception as e:
        print(f"Skipping {file}: {e}")

print(f"Already processed RXCUIs: {len(processed_rxcuis):,}")

remaining_rxcuis = [
    rxcui for rxcui in unique_rxcuis
    if rxcui not in processed_rxcuis
]

print(f"Remaining RXCUIs: {len(remaining_rxcuis):,}")

if not remaining_rxcuis:
    print("All RXCUIs already processed.")
    raise SystemExit

chunk_size = max(1, math.ceil(len(remaining_rxcuis) * 0.10))
print(f"Chunk size: {chunk_size:,} RXCUIs")

existing_files = sorted(CHECKPOINT_DIR.glob("rxnorm_synonyms_*.csv"))

chunk_id = (
    max(int(f.stem.split("_")[-1]) for f in existing_files) + 1
    if existing_files
    else 1
)

print(f"Starting chunk: {chunk_id:03d}")

chunk_rows = []
processed_in_run = 0

for rxcui in tqdm(remaining_rxcuis):
    processed_in_run += 1
    chunk_rows.extend(get_all_names(rxcui))
    time.sleep(0.05)
    if processed_in_run % chunk_size == 0:
        outfile = CHECKPOINT_DIR / f"rxnorm_synonyms_{chunk_id:03d}.csv"

        (
            pd.DataFrame(chunk_rows)
            .drop_duplicates()
            .to_csv(outfile, index = False)
        )

        print(f"Saved {outfile}")
        chunk_rows = []
        chunk_id += 1

if chunk_rows:
    outfile = CHECKPOINT_DIR / f"rxnorm_synonyms_{chunk_id:03d}.csv"

    (
        pd.DataFrame(chunk_rows)
        .drop_duplicates()
        .to_csv(outfile, index = False)
    )

    print(f"Saved {outfile}")

print("Synonym expansion complete.")

# %%
# Final vocabulary
if "vocab_exact_real" not in globals():
    vocab_exact_real = pd.read_csv(SCRIPT_ROOT / "cached_data" / "rxnorm_vocabulary_exact_real_drugs.csv")

CHECKPOINT_DIR = SCRIPT_ROOT / "rxnorm_synonym_checkpoints"
CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)
synonym_files = sorted(CHECKPOINT_DIR.glob("rxnorm_synonyms_*.csv"))

if synonym_files:
    synonym_vocab = pd.concat((pd.read_csv(f) for f in synonym_files), ignore_index = True)
else:
    synonym_vocab = pd.DataFrame(columns = ["drug_name", "rxcui", "tty"])

synonym_vocab = normalize_vocab_columns(synonym_vocab)
synonym_vocab = synonym_vocab[synonym_vocab["tty"].isin(INGREDIENT_TTYS)].copy()
final_vocab = choose_preferred_rxcui(vocab_exact_real,synonym_vocab)

print(f"Final vocabulary rows: {len(final_vocab):,}")
print(final_vocab.head())

duplicates = final_vocab["drug_name"].duplicated().sum()
if duplicates:
    print(f"Warning: {duplicates} duplicate drug names remain.")

final_vocab.to_csv(SCRIPT_ROOT / "cached_data" / "rxnorm_final_vocabulary.csv", index = False)

OUTPUT_DIR = ROOT / "cached_data"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
final_vocab.to_csv(OUTPUT_DIR / "rxnorm_final_vocabulary.csv", index = False)
print("Saved: cached_data/rxnorm_final_vocabulary.csv")