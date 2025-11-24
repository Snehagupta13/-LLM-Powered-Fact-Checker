# utils.py
# Helper functions
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

# Load environment variables, e.g. OPENAI_API_KEY
load_dotenv()

# Basic logging config
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_fact_checker")


def get_project_root() -> Path:
    """Return the project root (directory containing this file's parent)."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return get_project_root() / "data"


def embeddings_dir() -> Path:
    return get_project_root() / "embeddings"


def load_verified_facts(csv_name: str = "verified_facts.csv") -> pd.DataFrame:
    """
    Load the verified facts CSV from the data directory.
    Expected columns: id, statement, source
    """
    path = data_dir() / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Verified facts CSV not found at {path}")
    df = pd.read_csv(path)
    if "statement" not in df.columns:
        raise ValueError("CSV must contain a 'statement' column.")
    return df


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your environment or .env file."
        )
    return key

