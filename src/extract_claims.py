# src/extract_claims.py

import json
import os
from typing import Dict, List
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .utils import logger


SYSTEM_PROMPT = """
You extract factual claims and named entities from text.
Return ONLY JSON:

{
  "claims": [
    {
      "claim": "string",
      "entities": [
        {"text": "entity text", "label": "TYPE"}
      ]
    }
  ]
}
"""


@lru_cache(maxsize=1)
def _get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
    )


def extract_claims(text: str) -> List[Dict]:
    text = text.strip()
    if not text:
        return []

    llm = _get_llm()

    prompt = f"""
Text:
\"\"\"{text}\"\"\"

Extract factual claims and entities.
Return ONLY JSON with 'claims'.
"""

    logger.info("Extracting claims using Groq LLM...")

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    raw = response.content.strip()

    try:
        data = json.loads(raw)
        claims = data.get("claims", [])
        cleaned = []
        for c in claims:
            claim_text = c.get("claim", "").strip()
            if not claim_text:
                continue
            entities = c.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            cleaned.append({
                "claim": claim_text,
                "entities": [
                    {"text": e.get("text", ""), "label": e.get("label", "")}
                    for e in entities if e.get("text")
                ]
            })
        if cleaned:
            return cleaned
    except Exception as e:
        logger.error("Failed to parse claim extraction JSON: %s", e)

    return [{"claim": text, "entities": []}]
