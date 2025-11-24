import json
from typing import Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


from .utils import logger, os


SYSTEM_PROMPT = """You are a strict fact-checking assistant.
You will be given:
1. A claim.
2. A list of retrieved factual statements.

Your task:
- Compare the claim with the evidence.
- Return ONLY valid JSON with fields:
  - verdict: "True", "False", "Unverifiable", or "Likely True", "Likely False".
  - evidence: list of strings.
  - reasoning: short explanation.
  - confidence: float 0 to 1.

Follow evidence strictly. If unclear → choose "Unverifiable".
"""


def build_user_prompt(claim: str, evidence: List[str]) -> str:
    evidence_block = "\n".join(f"- {e}" for e in evidence) if evidence else "No evidence found."
    return f"""Claim:
\"\"\"{claim}\"\"\"

Evidence:
{evidence_block}

Respond ONLY in JSON.
"""


def compare_claim_with_evidence(
    claim: str,
    retrieved_facts: List[Dict],
    model: str = "llama-3.1-8b-instant"
) -> Dict:

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing in environment variables.")

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,
        max_tokens=512,
    )

    evidence_texts = [f["statement"] for f in retrieved_facts]

    user_prompt = build_user_prompt(claim, evidence_texts)

    logger.info("Calling Groq model %s for fact comparison...", model)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()

    # Attempt JSON parsing
    try:
        return json.loads(content)
    except:
        logger.error("JSON parsing failed. Raw output: %s", content)
        return {
            "verdict": "Unverifiable",
            "evidence": evidence_texts,
            "reasoning": "LLM JSON parsing failed.",
            "confidence": 0.3
        }
