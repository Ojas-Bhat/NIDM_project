# datapop3.py
import os
import re
import json
import hashlib
import argparse
import warnings
from datetime import datetime

import pdfplumber
import nltk
import pytesseract
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, ASCENDING, errors

# ---------- CONFIG (adjust if needed) ----------
# If Tesseract is installed elsewhere, set the path here:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "disaster_hub"
COLL_NAME = "documents"

# Model: all-MiniLM-L6-v2 (384-d)
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

# Mute noisy rendering warnings (optional, safe)
warnings.filterwarnings("ignore", message="Cannot set .* color")

# ---------- DB SETUP ----------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLL_NAME]

# Make re-runs idempotent: one row per (doc_id, text_hash)
try:
    collection.create_index(
        [("doc_id", ASCENDING), ("text_hash", ASCENDING)],
        unique=True,
        name="uniq_doc_chunk"
    )
except errors.PyMongoError:
    pass

# Ensure NLTK punkt is available (safe if run multiple times)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ---------- OCR CACHE HELPERS ----------
def _cache_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_cache")

def _cache_dir_for(pdf_path: str) -> str:
    h = hashlib.sha1(pdf_path.encode("utf-8")).hexdigest()
    d = os.path.join(_cache_root(), h)
    os.makedirs(d, exist_ok=True)
    return d

def _cache_key(pdf_path: str, page_index: int) -> str:
    # include file mtime so cache invalidates when the PDF changes
    try:
        mtime = os.path.getmtime(pdf_path)
    except Exception:
        mtime = 0
    return f"p{page_index}_m{int(mtime)}.json"

def _load_ocr_cache(pdf_path: str, page_index: int) -> str:
    d = _cache_dir_for(pdf_path)
    k = _cache_key(pdf_path, page_index)
    p = os.path.join(d, k)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("text", "")
        except Exception:
            return ""
    return ""

def _save_ocr_cache(pdf_path: str, page_index: int, text: str):
    d = _cache_dir_for(pdf_path)
    k = _cache_key(pdf_path, page_index)
    p = os.path.join(d, k)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"text": text, "ts": datetime.utcnow().isoformat()}, f)
    except Exception:
        pass


# ---------- TEXT EXTRACTION (Selective OCR) ----------
def extract_text_from_pdf(
    pdf_path: str,
    *,
    enable_ocr: bool = True,
    ocr_dpi: int = 300,
    min_text_chars: int = 40,   # if page text < this, treat as "no text"
    ocr_lang: str = "eng",      # e.g., "eng+hin" if needed
) -> str:
    """
    Extract text efficiently:
      - Use embedded text layer when present
      - OCR only pages with insufficient text
      - Cache OCR results per page to avoid re-running Tesseract
    """
    full_text = []
    ocr_count = 0
    pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            pages += 1

            # 1) Try text layer
            page_text = page.extract_text() or ""
            if len(page_text.strip()) >= min_text_chars:
                full_text.append(page_text.strip())
                continue  # no OCR needed

            # 2) If OCR disabled, keep whatever tiny text we have
            if not enable_ocr:
                if page_text:
                    full_text.append(page_text.strip())
                continue

            # 3) Cache first
            cached = _load_ocr_cache(pdf_path, i)
            if cached:
                full_text.append(cached.strip())
                ocr_count += 1  # still count as OCR page (served from cache)
                continue

            # 4) OCR this page
            try:
                pil_image = page.to_image(resolution=ocr_dpi).original
                config = r"--oem 1 --psm 3"  # LSTM engine, automatic page segmentation
                ocr_text = pytesseract.image_to_string(pil_image, lang=ocr_lang, config=config) or ""
                ocr_text = ocr_text.strip()
                _save_ocr_cache(pdf_path, i, ocr_text)
                full_text.append(ocr_text)
                ocr_count += 1
            except Exception:
                # If OCR fails, at least store the tiny text layer (if any)
                full_text.append(page_text.strip())

    print(f"→ Selective OCR summary: {ocr_count}/{pages} page(s) OCR’d for {os.path.basename(pdf_path)}")
    return "\n".join(t for t in full_text if t)


# ---------- CHUNKING ----------
def clean_and_chunk_text(text: str, max_chunk_size: int = 500, min_chunk_len: int = 80):
    """
    Chunk by sentences; keep chunks within max size and drop tiny fragments.
    """
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + 1 + len(sentence) <= max_chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if len(current) >= min_chunk_len:
                chunks.append(current)
            current = sentence
    if current and len(current) >= min_chunk_len:
        chunks.append(current)
    return chunks


# ---------- METADATA (Scoring-based) ----------
def _disaster_patterns():
    # Each hazard has a list of regex patterns
    return {
        "Cyclone": [
            r"\bcyclone(s)?\b", r"\bcyclonic\b",
            r"\bvery severe cyclonic storm\b", r"\bvs?cs\b",
            r"\bhurricane(s)?\b", r"\btyphoon(s)?\b",
            r"\btropical cyclone(s)?\b", r"\btropical storm(s)?\b"
        ],
        "Flood": [
            r"\bflood(s|ing)?\b", r"\binundation(s)?\b",
            r"\briverine flood(s)?\b", r"\bflash flood(s)?\b"
        ],
        "Earthquake": [
            r"\bearthquake(s)?\b", r"\bseismic\b",
            r"\bmagnitude\s*\d+(\.\d+)?\b", r"\bepicenter\b"
        ],
        "Landslide": [
            r"\blandslide(s)?\b", r"\bdebris flow(s)?\b", r"\bslope failure(s)?\b"
        ],
        "Drought": [
            r"\bdrought(s)?\b", r"\bwater scarcity\b", r"\bmeteorological drought\b"
        ],
        "Tsunami": [
            r"\btsunami(s)?\b"
        ],
        "Fire": [
            r"\b(wild)?fire(s)?\b", r"\bforest fire(s)?\b", r"\bwildland fire(s)?\b"
        ],
        "Heat Wave": [
            r"\bheat ?wave(s)?\b", r"\bheatwave(s)?\b",
            r"\bmaximum temperature(s)?\b", r"\bwet bulb\b"
        ],
        # Keep "Storm" generic, never override strong specific hazards:
        "Storm": [
            r"\bstorm(s)?\b", r"\bgale(s)?\b", r"\bwindstorm(s)?\b"
        ],
        "Pandemic": [
            r"\bpandemic(s)?\b", r"\bcovid-?19\b", r"\bsars-cov-2\b"
        ],
    }

def _count_matches(text: str, patterns: list[str]) -> int:
    t = text.lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, t, flags=re.IGNORECASE))
    return total

def _score_disaster(text: str, file_name: str) -> dict:
    pats = _disaster_patterns()
    scores = {k: 0 for k in pats.keys()}

    # Count in body text
    for k, plist in pats.items():
        scores[k] += _count_matches(text, plist)

    # Filename is a strong hint → add a larger boost
    for k, plist in pats.items():
        boost = _count_matches(file_name, plist)
        if boost:
            scores[k] += boost * 5  # filename weight

    return scores

def _pick_disaster(scores: dict) -> str:
    # choose highest score with some tie/override rules
    if all(v == 0 for v in scores.values()):
        return "Unknown"

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0

    # If top is generic "Storm" but a specific hazard is close, prefer specific one
    if top == "Storm":
        for specific in ["Cyclone", "Flood", "Heat Wave", "Fire", "Landslide", "Earthquake", "Tsunami", "Pandemic"]:
            if scores.get(specific, 0) >= (0.8 * top_score) and scores.get(specific, 0) > 0:
                return specific

    # If top and second are very close and second is more specific & non-zero, prefer second
    # (light tie-break to reduce OCR noise flips)
    specific_order = ["Cyclone", "Flood", "Heat Wave", "Fire", "Landslide", "Earthquake", "Tsunami", "Pandemic", "Storm"]
    if second_score > 0 and top_score < 1.2 * second_score:
        # prefer the higher-priority specific hazard between the two
        t1 = specific_order.index(top) if top in specific_order else len(specific_order)
        # find which hazard has second_score (not robust if multiple tie; acceptable heuristic)
        second = next((k for k, v in scores.items() if v == second_score and k != top), None)
        if second and second in specific_order:
            t2 = specific_order.index(second)
            if t2 < t1:
                return second

    return top

def extract_metadata(text: str, file_name: str) -> dict:
    metadata = {"source": file_name}

    # Year (1900-2099)
    year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", text)
    metadata["year"] = int(year_match.group(0)) if year_match else None

    # Score hazards using both body & filename
    scores = _score_disaster(text, file_name)
    disaster = _pick_disaster(scores)

    metadata["disaster"] = disaster
    return metadata


# ---------- STORAGE ----------
def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

from pymongo import MongoClient
from datetime import datetime

from pymongo import MongoClient
from datetime import datetime

from sentence_transformers import SentenceTransformer
import numpy as np

# load model globally to avoid reloading
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def store_document(filename, chunks, metadata):
    """
    Store document chunks into MongoDB with embeddings for semantic search.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["disaster_hub"]
    docs_col = db["documents"]

    metadata["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doc_id = os.path.splitext(filename)[0]

    # Generate embeddings for all chunks
    print(f"→ Generating embeddings for {len(chunks)} chunks ...")
    embeddings = EMBEDDER.encode(chunks, show_progress_bar=False).tolist()

    batch = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        text_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
        batch.append({
            "doc_id": doc_id,
            "text_hash": text_hash,
            "filename": filename,
            "chunk_index": i,
            "text": chunk,
            "embedding": emb,              # <— ADD THIS LINE
            "metadata": metadata
        })

    if batch:
        try:
            docs_col.insert_many(batch, ordered=False)
            print(f"✅ Stored {len(batch)} chunks for {filename}")
        except Exception as e:
            print(f"⚠️ MongoDB insert warning: {e}")
    else:
        print(f"⚠️ No chunks to store for {filename}")

    client.close()




# ---------- PIPELINE ----------
def process_pdf(
    pdf_path: str,
    *,
    enable_ocr: bool = True,
    ocr_dpi: int = 300,
    min_text_chars: int = 40,
    ocr_lang: str = "eng",
):
    """
    Full pipeline for a single PDF: extract -> chunk -> metadata -> store.
    """
    if not os.path.exists(pdf_path):
        print(f"✗ Skipped (file not found): {pdf_path}")
        return

    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]

    print(f"→ Extracting: {pdf_path}")
    raw_text = extract_text_from_pdf(
        pdf_path,
        enable_ocr=enable_ocr,
        ocr_dpi=ocr_dpi,
        min_text_chars=min_text_chars,
        ocr_lang=ocr_lang,
    )
    if not raw_text or not raw_text.strip():
        print("  ✗ No extractable text.")
        return

    print("→ Chunking text ...")
    chunks = clean_and_chunk_text(raw_text, max_chunk_size=500, min_chunk_len=80)
    if not chunks:
        print("  ✗ No valid chunks.")
        return

    print("→ Building metadata ...")
    metadata = extract_metadata(raw_text, file_name)
    print(f"  • metadata: {metadata}")

    print("→ Storing in MongoDB ...")
    store_document(filename=file_name, chunks=chunks, metadata=metadata)

    print(f"  ✓ Stored {len(chunks)} chunks for {file_name}\n")


def process_folder(
    folder_path: str,
    *,
    enable_ocr: bool = True,
    ocr_dpi: int = 300,
    min_text_chars: int = 40,
    ocr_lang: str = "eng",
):
    """
    Process every PDF in a folder (non-recursive).
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder_path}")

    print(f"Found {len(pdfs)} PDF(s) in {folder_path}")
    for fname in pdfs:
        process_pdf(
            os.path.join(folder_path, fname),
            enable_ocr=enable_ocr,
            ocr_dpi=ocr_dpi,
            min_text_chars=min_text_chars,
            ocr_lang=ocr_lang,
        )


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Ingest disaster PDFs into MongoDB with embeddings (selective OCR + smarter metadata)."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_folder = os.path.join(script_dir, "files")

    parser.add_argument("--file", "-f", help="Path to a single PDF to ingest.")
    parser.add_argument("--folder", "-d", default=default_folder,
                        help=f"Folder containing PDFs to ingest (default: {default_folder})")

    # OCR controls
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (use text layer only).")
    parser.add_argument("--ocr-lang", default="eng", help='OCR language(s), e.g. "eng+hin".')
    parser.add_argument("--ocr-dpi", type=int, default=300, help="Rendering DPI for OCR pages.")
    parser.add_argument("--min-text-chars", type=int, default=40,
                        help="If a page's extracted text has fewer chars than this, treat as no text and OCR it.")

    args = parser.parse_args()

    ocr_enabled = not args.no_ocr

    if args.file:
        process_pdf(
            args.file,
            enable_ocr=ocr_enabled,
            ocr_dpi=args.ocr_dpi,
            min_text_chars=args.min_text_chars,
            ocr_lang=args.ocr_lang,
        )
    else:
        process_folder(
            args.folder,
            enable_ocr=ocr_enabled,
            ocr_dpi=args.ocr_dpi,
            min_text_chars=args.min_text_chars,
            ocr_lang=args.ocr_lang,
        )


if __name__ == "__main__":
    main()