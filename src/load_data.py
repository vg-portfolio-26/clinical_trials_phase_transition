import logging
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from .helpers import log_separator, normalize_string
from .config import REQUIRED_CTGOV_COLUMNS, REQUIRED_NLM_MAP_COLUMNS, REQUIRED_RXNORM_COLUMNS

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "01_raw_data"


def load_table(repo_id: str) -> "pd.DataFrame":
    return load_dataset(repo_id)["train"].to_pandas()


def load_ctgov_tables() -> dict:
    logging.info("Loading datasets from Hugging Face...")

    config = {
        "studies": "Zipters/aact_ctgov_studies_raw",
        "interventions": "Zipters/aact_ctgov_interventions_raw",
        "browse": "Zipters/aact_ctgov_browse_conditions_raw"
    }

    tables = {}
    for name, repo_id in config.items():
        logging.info(f"Loading table: {name}")
        table = load_table(repo_id)
        tables[name] = table
        csv_path = _OUTPUT_DIR / f"{name}.csv"
        table.to_csv(csv_path, index = False)
        logging.info(f"Table saved: {csv_path}")

    logging.info("Data loaded and CSV files saved successfully!")
    log_separator()

    return tables


def validate_ctgov_tables(data: dict):
    logging.info("Validating ctgov tables...")

    required_tables = ["studies", "interventions", "browse"]

    for t in required_tables:
        if t not in data:
            raise ValueError(f"Missing table: {t}!")

    for name, df in data.items():
        if df is None or len(df) == 0:
            raise ValueError(f"Empty table: {name}!")

    for table_name, required_cols in REQUIRED_CTGOV_COLUMNS.items():
        missing_cols = [
            col for col in required_cols
            if col not in data[table_name].columns
        ]

        if missing_cols:
            raise ValueError(
                f"Table '{table_name}' missing columns: {missing_cols}"
            )
        
    for name, df in data.items():
        logging.info(f"{name}: {df.shape}")

    logging.info("ctgov tables validation successful!")
    log_separator()


def get_nlm_mesh_descriptors(repo_id: str) -> str:
    logging.info("Downloading NLM MeSH descriptors...")

    xml_path = hf_hub_download(
        repo_id=repo_id,
        filename = "desc2026.xml",
        repo_type = "dataset"
    )

    logging.info(f"NLM MeSH descriptors successfully downloaded")

    return xml_path


def build_mesh_mapping_table(mesh_xml_path: str) -> pd.DataFrame:
    logging.info("Building MeSH mapping table...")

    rows = []
    context = ET.iterparse(mesh_xml_path, events = ("end",))

    for _, elem in context:

        if elem.tag.endswith("DescriptorRecord"):

            mesh_id = None
            descriptor_name = None
            synonyms = set()
            tree_numbers = set()

            for child in elem:
                if child.tag.endswith("DescriptorUI"):
                    mesh_id = child.text
                elif child.tag.endswith("DescriptorName"):
                    for sub in child:
                        if sub.tag.endswith("String"):
                            descriptor_name = sub.text
                elif child.tag.endswith("TreeNumberList"):
                    for tree_child in child:
                        if tree_child.tag.endswith("TreeNumber"):
                            if tree_child.text:
                                tree_numbers.add(tree_child.text)
                elif child.tag.endswith("ConceptList"):
                    for concept in child:
                        for concept_child in concept:
                            if concept_child.tag.endswith("TermList"):
                                for term in concept_child:
                                    for term_child in term:
                                        if term_child.tag.endswith("String"):
                                            synonyms.add(
                                                normalize_string(term_child.text)
                                            )

            tree_number = None
            mesh_tree_group = None
            if tree_numbers:
                tree_number = sorted(tree_numbers)[0]
                mesh_tree_group = tree_number.split('.')[0]

            if mesh_id and descriptor_name:
                descriptor_norm = normalize_string(descriptor_name)
                row = {
                    "term": descriptor_norm,
                    "mesh_id": mesh_id,
                    "mesh_descriptor": descriptor_norm,
                    "match_type": "descriptor",
                    "tree_number": tree_number,
                    "mesh_tree_group": mesh_tree_group
                }
                rows.append(row)

                for synonym in synonyms:
                    if synonym:
                        rows.append({
                            "term": synonym,
                            "mesh_id": mesh_id,
                            "mesh_descriptor": descriptor_norm,
                            "match_type": "synonym",
                            "tree_number": tree_number,
                            "mesh_tree_group": mesh_tree_group
                        })

            elem.clear()

    mapping_df = (
        pd.DataFrame(rows)
        .drop_duplicates()
        .reset_index(drop = True)
    )

    mapping_df.to_csv(_OUTPUT_DIR / "nlm_mesh_map.csv", index = False)
    logging.info("MeSH mapping table created!")
    log_separator()

    return mapping_df


def validate_nlm_mesh_map(nlm_mesh_map: pd.DataFrame):
    logging.info("Validating NLM MeSH mapping table...")

    if nlm_mesh_map is None or len(nlm_mesh_map) == 0:
        raise ValueError("NLM MeSH mapping table is empty!")

    for col in REQUIRED_NLM_MAP_COLUMNS:
        if col not in nlm_mesh_map.columns:
            raise ValueError(f"Missing column in NLM MeSH mapping table: {col}")
        
    logging.info(f"NLM MeSH map: {nlm_mesh_map.shape}")
    logging.info("NLM MeSH mapping table validation successful!")
    log_separator()


def load_rxnorm_vocabulary():
    logging.info("Loading rxnorm vocabulary...")

    root = Path(__file__).resolve().parents[1]
    rxnorm_path = root / "cached_data" / "rxnorm_vocabulary.csv"
    rxnorm = pd.read_csv(rxnorm_path)

    if rxnorm is None or len(rxnorm) == 0:
        raise ValueError("rxnorm vocabulary is empty!")
    
    for col in REQUIRED_RXNORM_COLUMNS:
        if col not in rxnorm.columns:
            raise ValueError(f"Missing column in rxnorm vocabulary table: {col}")

    logging.info(f"rxnorm vocabulary: {rxnorm.shape}")
    logging.info("rxnorm vocabulary loaded and validated successfully!")
    log_separator()
    
    return rxnorm