# vector_retriever.py
# Retrieve top-k facts
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .utils import embeddings_dir, load_json, logger


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedFact:
    index: int
    score: float
    statement: str
    source: str


class FactRetriever:
    """
    Wraps FAISS + metadata + embedding model for semantic retrieval.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        emb_dir = embeddings_dir()

        self.index_path = index_path or (emb_dir / "vector_store.faiss")
        self.metadata_path = metadata_path or (emb_dir / "metadata.json")

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. Run build_fact_base first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata JSON not found at {self.metadata_path}. Run build_fact_base first."
            )

        self.index = faiss.read_index(str(self.index_path))
        self.metadata: List[Dict] = load_json(self.metadata_path)

        if len(self.metadata) != self.index.ntotal:
            logger.warning(
                "Metadata length (%d) != index.ntotal (%d)",
                len(self.metadata),
                self.index.ntotal,
            )

        logger.info(
            "FactRetriever initialized with model %s, %d facts.",
            self.model_name,
            len(self.metadata),
        )

    def _encode(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = emb.astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedFact]:
        if not query.strip():
            return []

        query_vec = self._encode(query)
        scores, indices = self.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[RetrievedFact] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievedFact(
                    index=int(idx),
                    score=float(score),
                    statement=str(meta.get("statement", "")),
                    source=str(meta.get("source", "")),
                )
            )

        return results
