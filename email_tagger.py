"""
email_tagger.py

Drop this into a server (or a lambda + API gateway) and point your mail system
to POST incoming mail (subject, body, message_id) to /classify.

Supports:
 - Local embedding-based zero-shot tagging (default, cheap)
 - Optional trainable classifier (tiny sklearn LR)
 - Optional HuggingFace Inference API zero-shot path (if you prefer hosted)
"""

import os
import json
from typing import List, Dict, Tuple
from flask import Flask, request, jsonify
import numpy as np
from dotenv import load_dotenv

# For local embedding mode
from sentence_transformers import SentenceTransformer, util

# Optional training utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import requests
import threading

load_dotenv()

# Config
HF_API_KEY = os.getenv("HF_API_KEY")
USE_LOCAL = os.getenv("USE_LOCAL", "true").lower() in ("true", "1", "yes")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # small+fast
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
TAG_SIM_THRESHOLD = float(os.getenv("TAG_SIM_THRESHOLD", "0.55"))  # tune this

# Filepaths for persistence
CLASSIFIER_PATH = "email_lr_classifier.joblib"
MLB_PATH = "email_mlb.joblib"
EMBED_CACHE_PATH = "tag_template_embeddings.npz"

TAG_TEMPLATES = {
    "billing": [
        "invoice",
        "billing question",
        "refund",
        "payment failed",
        "charge on my account",
    ],
    "support": [
        "I need help",
        "support request",
        "how to",
        "can't login",
        "error when I try to",
    ],
    "feature_request": [
        "feature request",
        "would be great if",
        "please add",
        "can you add",
    ],
    "complaint": [
        "complaint",
        "not happy",
        "very disappointed",
        "this is unacceptable",
    ],
    "sales": [
        "pricing",
        "purchase",
        "interested in buying",
        "demo request",
    ],
    "login_issue": [
        "reset password",
        "forgot password",
        "can't sign in",
        "two factor",
    ],
}

# Load sentence-transformers model lazily (thread-safe-ish)
_model_lock = threading.Lock()
_model = None
_tag_template_embeddings = None


def load_model():
    global _model, _tag_template_embeddings
    with _model_lock:
        if _model is None:
            print("Loading embedder:", EMBED_MODEL_NAME)
            _model = SentenceTransformer(EMBED_MODEL_NAME)
            # precompute tag template embeddings
            _tag_template_embeddings = compute_tag_template_embeddings(_model)
    return _model


def compute_tag_template_embeddings(model) -> Dict[str, np.ndarray]:
    tag_emb = {}
    for tag, examples in TAG_TEMPLATES.items():
        embs = model.encode(examples, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        tag_emb[tag] = np.mean(embs, axis=0)
    return tag_emb


def embed_text(text: str, model) -> np.ndarray:
    if not text:
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=float)
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return emb


def classify_with_embeddings(subject: str, body: str, top_k: int = 3) -> List[Tuple[str, float]]:
    model = load_model()
    combined = (subject or "") + "\n\n" + (body or "")
    emb = embed_text(combined, model)
    scores = []
    for tag, t_emb in _tag_template_embeddings.items():
        sim = float(np.dot(emb, t_emb))  # cosine because embeddings are normalized
        scores.append((tag, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    # filter by threshold
    filtered = [(t, float(s)) for (t, s) in scores if s >= TAG_SIM_THRESHOLD]
    # fall back: take top_k even if under threshold (useful for routing)
    if not filtered:
        filtered = [(t, float(s)) for (t, s) in scores[:top_k]]
    return filtered[:top_k]


def train_classifier(labeled_examples: List[Dict[str, object]]):
    """
    labeled_examples = [
      {"subject": "...", "body":"...", "tags": ["support","login_issue"]},
      ...
    ]
    """
    model = load_model()
    texts = []
    labels = []
    for ex in labeled_examples:
        texts.append((ex.get("subject") or "") + "\n\n" + (ex.get("body") or ""))
        labels.append(ex.get("tags", []))
    X = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, Y)
    # persist
    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(mlb, MLB_PATH)
    return {"status": "ok", "classes": mlb.classes_.tolist()}


def classify_with_trained(subject: str, body: str, top_k: int = 3):
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(MLB_PATH):
        return []
    clf = joblib.load(CLASSIFIER_PATH)
    mlb = joblib.load(MLB_PATH)
    model = load_model()
    text = (subject or "") + "\n\n" + (body or "")
    x = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].reshape(1, -1)
    probs = clf.predict_proba(x)  # list of per-class arrays or single array depending on sklearn
    # sklearn returns list if multi-class; for multioutput it may be circumvented.
    # To adapt: handle both shapes
    if isinstance(probs, list):
        probs = np.array([p[0] for p in probs]).reshape(1, -1)
    probs = probs[0]
    tag_probs = list(zip(mlb.classes_.tolist(), probs.tolist()))
    tag_probs.sort(key=lambda x: x[1], reverse=True)
    return [(t, float(p)) for t, p in tag_probs[:top_k]]


# Optional: Hugging Face zero-shot via inference API
def classify_with_hf_zero_shot(subject: str, body: str, candidate_labels: List[str], hf_api_key: str = HF_API_KEY):
    """
    Uses HF's zero-shot API (a hosted infer endpoint) if you want to avoid local models.
    """
    if not hf_api_key:
        raise ValueError("HF_API_KEY not set")
    text = (subject or "") + "\n\n" + (body or "")
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"  # robust zero-shot
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels, "hypothesis_template": "This email is about {}."}}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    r = resp.json()
    # r should contain labels and scores
    labels = r.get("labels", [])
    scores = r.get("scores", [])
    return list(zip(labels, scores))


# Flask app for integration
app = Flask("email_tagger")


@app.route("/classify", methods=["POST"])
def web_classify():
    """
    Expect JSON: {"message_id": "...", "subject":"...", "body":"...", "use":"local|trained|hf"}
    Returns: {"message_id":"...", "tags":[{"tag":"support","score":0.92}, ...]}
    """
    payload = request.get_json(force=True)
    subject = payload.get("subject", "")
    body = payload.get("body", "")
    message_id = payload.get("message_id", None)
    mode = payload.get("use", "local")
    try:
        if mode == "trained":
            results = classify_with_trained(subject, body)
        elif mode == "hf":
            candidate_labels = list(TAG_TEMPLATES.keys())
            hf_res = classify_with_hf_zero_shot(subject, body, candidate_labels)
            results = hf_res
        else:
            results = classify_with_embeddings(subject, body)

        tags = [{"tag": t, "score": float(s)} for t, s in results]
        return jsonify({"message_id": message_id, "tags": tags})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def web_train():
    """
    Accepts labeled examples to train local classifier:
    JSON: {"examples": [{"subject":"", "body":"", "tags":["support"]}, ...]}
    """
    payload = request.get_json(force=True)
    examples = payload.get("examples", [])
    if not examples:
        return jsonify({"error": "no examples provided"}), 400
    res = train_classifier(examples)
    return jsonify(res)


if __name__ == "__main__":
    # Preload model in main thread (optional)
    if USE_LOCAL:
        load_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
