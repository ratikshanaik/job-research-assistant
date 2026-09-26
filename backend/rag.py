import os
import json
import numpy as np
import faiss
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

index = faiss.read_index("../data/job_index.faiss")
with open("../data/chunk_metadata.json", "r") as f:
    all_chunks = json.load(f)

def get_embedding(text):
    result = client.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def retrieve_relevant_chunks(query, top_k=5):
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
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text

def ask(query, top_k=5):
    chunks = retrieve_relevant_chunks(query, top_k)
    answer = generate_answer(query, chunks)
    sources = list(set([f"{c['title']} at {c['company']}" for c in chunks]))
    return {"answer": answer, "sources": sources}