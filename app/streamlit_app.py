# streamlit_app.py
# Optional Streamlit UI

import sys
import os
import json
import streamlit as st

# Ensure project root is in sys.path for 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import run_fact_check

st.set_page_config(page_title="LLM Fact Checker", layout="wide")

st.title("🔍 LLM-Powered Fact Checker")

st.markdown(
    "Paste a short news or social media claim below and check it against the trusted fact base."
)

default_text = "The Indian government has announced free electricity to all farmers starting July 2025."

input_text = st.text_area("Input text", value=default_text, height=150)
top_k = st.slider("Top-k evidence to retrieve", min_value=3, max_value=10, value=5)

if st.button("Run Fact Check"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running fact check..."):
            result = run_fact_check(input_text, top_k=top_k)

        st.subheader("Raw JSON Output")
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

        st.subheader("Claims & Verdicts")
        for i, c in enumerate(result["claims"], start=1):
            st.markdown(f"### Claim {i}")
            st.write(c["claim"])

            analysis = c["analysis"]
            verdict = analysis.get("verdict", "Unverifiable")
            confidence = analysis.get("confidence", 0.0)
            reasoning = analysis.get("reasoning", "")

            st.markdown(f"- **Verdict:** `{verdict}` (confidence: {confidence:.2f})")
            st.markdown(f"- **Reasoning:** {reasoning}")

            with st.expander("Evidence"):
                for f in c["retrieved_facts"]:
                    st.markdown(
                        f"- **Score:** {f['score']:.3f} | **Source:** {f['source']}  \n"
                        f"  {f['statement']}"
                    )

        st.success("Done ✅")
