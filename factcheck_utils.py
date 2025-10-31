import pandas as pd
import numpy as np
import nltk
import re
import time
import json
import string
import torch
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from newspaper import Article
import ollama
from ddgs import DDGS

# === NLTK setup ===
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# === Global Setup ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# === Preprocessing & NLP ===
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def extract_keywords(text):
    tokens = word_tokenize(clean_text(text))
    tagged = pos_tag(tokens)
    keywords = set()
    for word, tag in tagged:
        if word not in stop_words and word.isalpha():
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos)
            keywords.add(lemma)
    return keywords

def keyword_score(kw1, kw2):
    return round(len(kw1 & kw2) / len(kw1) * 100, 2) if kw1 else 0.0

def semantic_score(claim, evidence):
    try:
        emb1 = model.encode(claim, convert_to_tensor=True)
        emb2 = model.encode(evidence, convert_to_tensor=True)
        return round(float(util.cos_sim(emb1, emb2)[0][0]) * 100, 2)
    except Exception:
        return 0.0

def compute_score(claim, evidence):
    kw_claim = extract_keywords(claim)
    kw_evidence = extract_keywords(evidence)
    basic = keyword_score(kw_claim, kw_evidence)
    sem = semantic_score(claim, evidence)
    return round((basic + sem) / 2, 2)


# === LLM Integration ===
def call_llm(prompt):
    try:
        response = ollama.chat(model='gemma3', messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    except Exception:
        return "ERROR"

def classify_stance(claim, evidence):
    ask = f"Claim: {claim}\nEvidence: {evidence}\nAnswer in one word: Does the evidence support, refute, or is uncertain about the claim?"
    result = call_llm(ask).lower()
    if "refute" in result:
        return "Refuted"
    elif "support" in result:
        return "Supported"
    return "Uncertain"


# === URL Handling ===
def extract_url(text):
    urls = re.findall(r'(https?://\S+)', text)
    return urls[0] if urls else None

def is_url_valid(url):
    try:
        resp = requests.head(url, timeout=5, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False

def search_duckduckgo(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        for r in results:
            if "href" in r:
                return r["href"]
    return None

def has_converged(score_trace, epsilon=1.0, patience=2):
    if len(score_trace) < patience + 1:
        return False
    diffs = [abs(score_trace[-i]['score'] - score_trace[-i-1]['score']) for i in range(1, patience + 1)]
    return all(diff < epsilon for diff in diffs)


# === Core Fact-checking ===
def process_claim(claim, max_attempts=6):
    prompt = f"""Fact-check the following claim. Respond in 2–3 sentences. 
You must cite a reliable news URL at the end of your response (include only one link).
Claim: \"{claim}\""""

    best_score = 0
    score_trace = []
    final_explanation, final_url = "", None

    for attempt in range(1, max_attempts + 1):
        response = call_llm(prompt)
        explanation = response.strip()
        url = extract_url(explanation)

        # URL validation and fallback
        if url and not is_url_valid(url):
            alt_url = search_duckduckgo(claim)
            if alt_url:
                explanation = re.sub(r'https?://\S+', alt_url, explanation)
                url = alt_url
            else:
                url = None

        score = compute_score(claim, explanation)
        score_trace.append({'attempt': attempt, 'score': score, 'explanation': explanation, 'url': url})

        if score > best_score:
            best_score = score
            final_explanation = explanation
            final_url = url

        if attempt >= 3 and has_converged(score_trace):
            break

        prompt = f"Improve your fact-checking explanation for this claim and include a real news URL:\n\"{claim}\""
        time.sleep(0.3)

    final_label = classify_stance(claim, final_explanation)

    return {
        "claim": claim,
        "evidence": final_explanation,
        "url": final_url if final_url else "NOT_FOUND",
        "score": best_score,
        "label": final_label,
        "attempts": len(score_trace),
        "score_trace": json.dumps(score_trace)
    }


# === Scrutinizer ===
def fetch_article(url: str) -> str:
    try:
        art = Article(url, language="en")
        art.download()
        art.parse()
        return art.text.strip()
    except Exception:
        return ""

def max_sent_cosine(model, source: str, article: str) -> float:
    try:
        sents = nltk.sent_tokenize(article)
        if not sents:
            return 0.0
        art_embs = model.encode(sents, convert_to_tensor=True)
        src_emb = model.encode(source, convert_to_tensor=True)
        return float(util.cos_sim(src_emb, art_embs)[0].max().item())
    except Exception:
        return 0.0


# === Final Label Resolver ===
def decide_label(row):
    label = row['label']
    match = row['match']

    if label == "Supported" and match:
        return "Strongly Supported"
    elif label == "Supported" and not match:
        return "Weakly Supported"
    elif label == "Refuted" and match:
        return "Strongly Refuted"
    elif label == "Refuted" and not match:
        return "Weakly Refuted"
    elif label == "Uncertain" and match:
        return "Possibly True"
    else:
        return "Unknown"