import streamlit as st
import pandas as pd
from factcheck_utils import process_claim, fetch_article, max_sent_cosine, decide_label
from sentence_transformers import SentenceTransformer, util
import nltk

# Setup
nltk.download('punkt', quiet=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Fact Checker", layout="wide")

st.title("🔍 AI-Powered Fact Checker")
st.markdown("""
Enter a **claim** below, and we'll fact-check it using:
- An LLM (Gemma)
- Keyword & semantic similarity
- News article validation
""")

claim = st.text_area("✍️ Enter a claim:", height=100)

if st.button("Fact-Check Now"):
    if not claim.strip():
        st.error("Please enter a valid claim.")
    else:
        with st.spinner("🧠 Processing..."):
            result = process_claim(claim)
            st.success("✅ Fact-check complete!")

            st.subheader("📄 Fact-check Summary")
            st.write(f"**Claim:** {result['claim']}")
            st.write(f"**Evidence:** {result['evidence']}")
            st.write(f"**Score:** `{result['score']}`")
            st.write(f"**Initial Label:** `{result['label']}`")

            if result['url'] and result['url'] != "NOT_FOUND":
                st.markdown(f"🔗 **Source:** [{result['url']}]({result['url']})")
            else:
                st.warning("⚠️ No valid source found.")

            # Scrutinizer Part
            st.subheader("📰 Scrutinizer")
            article_text = fetch_article(result['url']) if result['url'] else ""
            if article_text and len(article_text.split()) >= 20:
                sim = max_sent_cosine(model, result['evidence'], article_text)
                match = sim >= 0.55  # Adjust threshold if needed
                p_value = 0.0  # Assume match, you can enhance this later

                row = {
                    "label": result['label'],
                    "match": match
                }
                final_label = decide_label(row)

                st.write(f"**Max Sentence Similarity:** `{round(sim, 4)}`")
                st.write(f"**Final Label:** 🏁 `{final_label}`")
            else:
                st.warning("🚫 Could not retrieve or validate article content.")
