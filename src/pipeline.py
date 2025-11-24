# pipeline.py
# End-to-end pipeline (main logic)
from typing import Any, Dict, List

from .extract_claims import extract_claims
from .utils import logger
from .vector_retriever import FactRetriever
from .compare_llm import compare_claim_with_evidence


def run_fact_check(
    text: str,
    top_k: int = 5,
    min_similarity: float = 0.2,
) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    - Extract claims
    - Retrieve top-k evidence for each claim
    - Use LLM to compare and produce verdicts

    Returns a dictionary suitable for JSON dumping.
    """
    logger.info("Running fact-check pipeline on input: %s", text)

    claims = extract_claims(text)
    retriever = FactRetriever()

    results: List[Dict[str, Any]] = []

    for claim_obj in claims:
        claim_text = claim_obj["claim"]
        entities = claim_obj.get("entities", [])

        retrieved = retriever.retrieve(claim_text, top_k=top_k)
        # Filter by similarity threshold if desired
        filtered = [
            {
                "index": r.index,
                "score": r.score,
                "statement": r.statement,
                "source": r.source,
            }
            for r in retrieved
            if r.score >= min_similarity
        ]

        logger.info(
            "Claim: %s | Retrieved %d/%d facts above threshold %.2f",
            claim_text,
            len(filtered),
            len(retrieved),
            min_similarity,
        )

        analysis = compare_claim_with_evidence(
            claim=claim_text, retrieved_facts=filtered
        )

        results.append(
            {
                "claim": claim_text,
                "entities": entities,
                "retrieved_facts": filtered,
                "analysis": analysis,
            }
        )

    output = {
        "input_text": text,
        "claims": results,
    }

    return output

