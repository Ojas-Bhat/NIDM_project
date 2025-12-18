import subprocess
import json
import re
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

# --- MongoDB Setup ---
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_hub"]
collection = db["documents"]

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Parse Query for Filters ---
def extract_filters_from_query(query):
    filters = {}

    # Year detection (e.g., "2018 floods in Kerala")
    year_match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", query)
    if year_match:
        filters["metadata.year"] = int(year_match.group(0))

    # Disaster type detection (simple keyword matching)
    disaster_types = [
        "Flood", "Earthquake", "Cyclone", "Landslide",
        "Drought", "Tsunami", "Pandemic", "Storm", "Fire"
    ]
    for dtype in disaster_types:
        if dtype.lower() in query.lower():
            filters["metadata.disaster"] = dtype
            break

    return filters


# --- Retrieve Relevant Chunks ---
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])[0]

    filters = extract_filters_from_query(query)
    mongo_query = filters if filters else {}

    docs = list(collection.find(mongo_query, {"text": 1, "embedding": 1, "metadata": 1}))

    if not docs:
        return []

    embeddings = np.array([doc["embedding"] for doc in docs])
    doc_texts = [doc["text"] for doc in docs]

    # Compute cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = np.dot(doc_norms, query_norm)

    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(doc_texts[i], docs[i]["metadata"], float(sims[i])) for i in top_indices]


# --- Query Ollama ---
def query_ollama(model, prompt):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")


# --- Generate Answer ---
def generate_answer(user_query, retrieved, model="llama3"):
    context = "\n\n".join([f"[{m['disaster']} {m.get('year','?')}]\n{text}" for text, m, _ in retrieved])
    prompt = f"""You are a disaster knowledge assistant.
Answer the question based only on the context below.

Context:
{context}

Question: {user_query}

Answer:"""

    return query_ollama(model, prompt)


# --- Example Usage ---
if __name__ == "__main__":
    user_query = "Tell me about the Chennai Flood "

    retrieved = retrieve_relevant_chunks(user_query, top_k=5)
    if not retrieved:
        print("No relevant documents found.")
    else:
        answer = generate_answer(user_query, retrieved, model="llama3")
        print(answer)