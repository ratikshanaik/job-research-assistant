import os
import json
from pathlib import Path

import numpy as np
import faiss
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-3.8-flash")
EMBED_MODEL = "gemini-embedding-001"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

index = faiss.read_index(str(DATA_DIR / "job_index.faiss"))
with open(DATA_DIR / "chunk_metadata.json", "r") as f:
    all_chunks = json.load(f)

def get_embedding(text):
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values

def retrieve_relevant_chunks(query, top_k=10):
    query_vector = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    return [all_chunks[i] for i in indices[0]]

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([
        f"Job: {c['title']} at {c['company']} ({c['location']})\n{c['text']}"
        for c in retrieved_chunks
    ])
    prompt = f"""You are a helpful job market research assistant. Answer using ONLY the job postings below. If the answer isn't in the postings, say so honestly.

Job Postings:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text

def ask(query, top_k=10):
    chunks = retrieve_relevant_chunks(query, top_k)
    answer = generate_answer(query, chunks)

    seen, sources = set(), []
    for c in chunks:
        if c["url"] not in seen:
            seen.add(c["url"])
            sources.append({"title": c["title"], "company": c["company"],
                            "location": c["location"], "url": c["url"]})
    return {"answer": answer, "sources": sources}