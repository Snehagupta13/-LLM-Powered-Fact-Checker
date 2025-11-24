# build_fact_base.py
# Chunking + embedding the fact DB
import argparse
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import (
    data_dir,
    embeddings_dir,
    load_verified_facts,
    logger,
    save_json,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_embeddings(
    statements: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute sentence embeddings for all statements using SentenceTransformers.
    """
    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    all_embeddings = []
    for i in tqdm(range(0, len(statements), batch_size), desc="Embedding facts"):
        batch = statements[i : i + batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a simple FAISS index with inner product (cosine similarity).
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors of dimension %d", embeddings.shape[0], d)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build fact base embeddings and FAISS index.")
    parser.add_argument(
        "--csv",
        default="verified_facts.csv",
        help="Name of CSV file in data/ containing verified facts.",
    )
    args = parser.parse_args()

    df = load_verified_facts(args.csv)
    statements = df["statement"].fillna("").tolist()

    # Build embeddings
    embeddings = build_embeddings(statements)
    emb_dir = embeddings_dir()
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Save raw embeddings (optional)
    np.save(emb_dir / "embeddings.npy", embeddings)
    logger.info("Saved embeddings to %s", emb_dir / "embeddings.npy")

    # Build FAISS index
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(emb_dir / "vector_store.faiss"))
    logger.info("Saved FAISS index to %s", emb_dir / "vector_store.faiss")

    # Save metadata for lookup
    metadata = []
    for _, row in df.iterrows():
        metadata.append(
            {
                "id": int(row["id"]) if "id" in df.columns else None,
                "statement": str(row["statement"]),
                "source": str(row.get("source", "")),
            }
        )

    save_json(metadata, emb_dir / "metadata.json")
    logger.info("Saved metadata to %s", emb_dir / "metadata.json")

    # For convenience, also create processed_chunks.json in data/
    processed_chunks_path = data_dir() / "processed_chunks.json"
    with processed_chunks_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Saved processed chunks to %s", processed_chunks_path)


if __name__ == "__main__":
    main()
