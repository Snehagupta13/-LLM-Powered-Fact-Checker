# LLM Fact Checker

A modular pipeline for fact-checking using LLMs and vector search.
# LLM-Powered Fact Checker (RAG)

This project implements a lightweight, LLM-powered fact-checking system that:

1. Takes a short news or social media statement.
2. Extracts key claims and entities.
3. Retrieves relevant facts from a vector store built over trusted government statements.
4. Uses an LLM to compare the claim vs evidence.
5. Outputs a verdict: ✅ True / ❌ False / 🤷 Unverifiable, with reasoning.

---

## Project Structure

```bash
.
├── app/
├── data/
├── embeddings/
├── src/
├── samples/
├── notebooks/
└── video/

