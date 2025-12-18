# evaluate_visuals_improved_fixed.py
"""
Improved evaluation & visualization for RAG system (fixed CSV encoding handling).

Features:
 - Retrieval Recall@K (1..10)
 - Optional CrossEncoder reranking (top_N)
 - Semantic similarity (cosine) evaluation for generated answers
 - Bootstrapped 95% CI for Recall@K
 - Nice plots saved in eval_outputs/
"""

import os
import traceback
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer, util
import random

# Config
EVAL_CSV = "eval_questions1.csv"
OUTPUT_DIR = "eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
K_VALUES = list(range(1, 11))

# ---------- safe CSV loader ----------
def safe_read_csv(path, encodings=None, **kwargs):
    """
    Try multiple encodings to read a CSV. Returns DataFrame or raises last exception.
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            print(f"Loaded CSV using encoding: {enc}")
            return df
        except Exception as e:
            last_exc = e
    # Optional: try chardet to detect encoding
    try:
        import chardet  # optional dependency
        with open(path, "rb") as f:
            raw = f.read(100000)  # first 100k
        guess = chardet.detect(raw)
        enc = guess.get("encoding")
        if enc:
            try:
                df = pd.read_csv(path, encoding=enc, **kwargs)
                print(f"Loaded CSV using chardet-detected encoding: {enc}")
                return df
            except Exception as e:
                last_exc = e
    except Exception:
        pass
    raise last_exc

# ---------- try import pipeline functions ----------
try:
    from cb import retrieve_relevant_chunks, generate_answer
except Exception as e:
    print("Warning: cb.retrieve_relevant_chunks / generate_answer not importable:", e)
    retrieve_relevant_chunks = None
    generate_answer = None

# Optional cross-encoder reranker (slower, but stronger)
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N_FOR_RERANK = 10

# ---------- Load eval data (robust to encoding) ----------
if not os.path.exists(EVAL_CSV):
    raise FileNotFoundError(f"Missing {EVAL_CSV} in working dir")

df = safe_read_csv(EVAL_CSV, keep_default_na=False)

if 'question' not in df.columns or 'expected_answer' not in df.columns:
    raise ValueError("CSV must include 'question' and 'expected_answer'")

# ---------- Models ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = None
if USE_RERANKER:
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANKER_MODEL)
        print("Reranker loaded:", RERANKER_MODEL)
    except Exception as e:
        print("Could not load CrossEncoder; continuing without reranker. Error:", e)
        reranker = None

# ---------- Helper: semantic hit (cosine threshold) ----------
def semantic_hit(expected_text: str, retrieved_texts: list, threshold=0.55):
    """
    Returns True if any retrieved_text has cosine similarity >= threshold with expected_text.
    """
    if not expected_text or not retrieved_texts:
        return False
    try:
        emb_expected = embed_model.encode(expected_text, convert_to_tensor=True)
        emb_docs = embed_model.encode(retrieved_texts, convert_to_tensor=True)
        sims = util.cos_sim(emb_expected, emb_docs).cpu().numpy().flatten()
        return float(sims.max()) >= threshold
    except Exception:
        exp = expected_text.lower().strip()[:200]
        for t in retrieved_texts:
            if exp in (t or "").lower():
                return True
        return False

# ---------- Main evaluation ----------
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
    q = str(row['question'])
    expected = str(row['expected_answer'])
    retrieved = []
    try:
        if retrieve_relevant_chunks is None:
            raise RuntimeError("retrieve_relevant_chunks not available")
        retrieved = retrieve_relevant_chunks(q, top_k=max(K_VALUES + [TOP_N_FOR_RERANK]))
    except Exception as e:
        if idx == 0:
            traceback.print_exc()
        retrieved = []

    raw_texts = [t for t, m, s in retrieved]
    raw_meta = [m for t, m, s in retrieved]

    reranked_texts = raw_texts
    reranked_meta = raw_meta
    if reranker and raw_texts:
        cand_texts = raw_texts[:TOP_N_FOR_RERANK]
        cand_meta = raw_meta[:TOP_N_FOR_RERANK]
        pairs = [(q, c) for c in cand_texts]
        try:
            scores = reranker.predict(pairs)
            order = np.argsort(scores)[::-1]
            reranked_texts = [cand_texts[i] for i in order]
            reranked_meta = [cand_meta[i] for i in order]
        except Exception:
            # if reranker fails for any sample, keep original order
            pass

    hit_raw = {k:0 for k in K_VALUES}
    hit_rerank = {k:0 for k in K_VALUES}
    for k in K_VALUES:
        top_raw = raw_texts[:k]
        top_rer = reranked_texts[:k]
        hit_raw[k] = int(semantic_hit(expected, top_raw))
        hit_rerank[k] = int(semantic_hit(expected, top_rer))

    predicted = ""
    try:
        if generate_answer and reranked_texts:
            predicted = generate_answer(q, list(zip(reranked_texts, reranked_meta, [0]*len(reranked_texts))))
            if isinstance(predicted, bytes):
                predicted = predicted.decode("utf-8", errors="ignore")
            predicted = str(predicted).strip()
    except Exception:
        predicted = ""

    sim = None
    if predicted:
        try:
            emb1 = embed_model.encode(predicted, convert_to_tensor=True)
            emb2 = embed_model.encode(expected, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2).item())
        except Exception:
            sim = None

    results.append({
        "index": idx, "question": q, "expected_answer": expected,
        "predicted_answer": predicted, "hit_raw": hit_raw, "hit_rerank": hit_rerank,
        "similarity": sim
    })

# ---------- Compute Recall@K ----------
recall_raw = {k: np.mean([r['hit_raw'][k] for r in results]) for k in K_VALUES}
recall_rerank = {k: np.mean([r['hit_rerank'][k] for r in results]) for k in K_VALUES}

# ---------- Bootstrapped CI ----------
def bootstrap_recall(values, n_boot=500, alpha=0.05):
    arr = np.array(values)
    stats = []
    n = len(arr)
    for _ in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        stats.append(sample.mean())
    lo = np.percentile(stats, 100*alpha/2)
    hi = np.percentile(stats, 100*(1-alpha/2))
    return (np.mean(arr), lo, hi)

boot_raw = {}
boot_rerank = {}
for k in K_VALUES:
    vals_raw = [r['hit_raw'][k] for r in results]
    vals_rerank = [r['hit_rerank'][k] for r in results]
    boot_raw[k] = bootstrap_recall(vals_raw, n_boot=500)
    boot_rerank[k] = bootstrap_recall(vals_rerank, n_boot=500)

# ---------- Save recall table ----------
recall_table = pd.DataFrame({
    "K": K_VALUES,
    "recall_raw": [recall_raw[k] for k in K_VALUES],
    "recall_rerank": [recall_rerank[k] for k in K_VALUES],
    "raw_lo": [boot_raw[k][1] for k in K_VALUES],
    "raw_hi": [boot_raw[k][2] for k in K_VALUES],
    "rerank_lo": [boot_rerank[k][1] for k in K_VALUES],
    "rerank_hi": [boot_rerank[k][2] for k in K_VALUES],
})
recall_table.to_csv(os.path.join(OUTPUT_DIR, "recall_comparison.csv"), index=False)

# ---------- Plot recall with CI ----------
plt.figure(figsize=(8,5))
plt.plot(recall_table['K'], recall_table['recall_raw'], marker='o', label='Retriever')
plt.fill_between(recall_table['K'], recall_table['raw_lo'], recall_table['raw_hi'], alpha=0.15)
plt.plot(recall_table['K'], recall_table['recall_rerank'], marker='o', label='Retriever + Reranker')
plt.fill_between(recall_table['K'], recall_table['rerank_lo'], recall_table['rerank_hi'], alpha=0.15)
plt.xticks(K_VALUES)
plt.xlabel("Top-K")
plt.ylabel("Recall@K")
plt.title("Recall@K: Retriever vs Retriever+Reranker (95% CI shaded)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "recall_with_reranker.png"), dpi=200)
plt.close()
print("Saved:", os.path.join(OUTPUT_DIR, "recall_with_reranker.png"))

# ---------- Similarity histogram ----------
sims = [r['similarity'] for r in results if r['similarity'] is not None]
if sims:
    plt.figure(figsize=(8,5))
    sns.histplot(sims, bins=20, kde=True)
    plt.xlabel("Cosine similarity (predicted vs expected)")
    plt.title(f"Generated answer similarity (n={len(sims)})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "generated_similarity_hist.png"), dpi=200)
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "generated_similarity_hist.png"))
else:
    print("No generated answers to compute similarity histogram.")

# ---------- Confusion matrix for metadata labels if present ----------
label_col = None
for cand in ['true_disaster', 'true_label', 'expected_label', 'label']:
    if cand in df.columns:
        label_col = cand
        break

if label_col and retrieve_relevant_chunks:
    y_true = df[label_col].astype(str).tolist()
    y_pred = []
    for r in results:
        try:
            top1 = retrieve_relevant_chunks(r['question'], top_k=1)
            if top1:
                _, meta, _ = top1[0]
                y_pred.append(meta.get('disaster','Unknown') if isinstance(meta, dict) else str(meta))
            else:
                y_pred.append("Unknown")
        except Exception:
            y_pred.append("Unknown")
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (metadata from top-1 retrieved chunk)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()
    print("Saved:", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print(classification_report(y_true, y_pred, labels=labels))
else:
    print("No metadata labels or retrieval function; confusion matrix skipped.")

# ---------- Save detailed results ----------
pd.DataFrame(results).to_json(os.path.join(OUTPUT_DIR, "detailed_results.json"), orient='records', force_ascii=False)
print("Detailed results written to:", os.path.join(OUTPUT_DIR, "detailed_results.json"))
print("All outputs saved in", OUTPUT_DIR)
