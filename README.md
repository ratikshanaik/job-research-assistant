# AI/ML Job Research Assistant

A RAG-based assistant that answers natural-language questions about live AI/ML job postings, instead of manually reading through dozens of listings.

**Live app:** https://job-research-assistant-wruhnhjflapnbk2jqnwpt4.streamlit.app/
**Backend API:** https://job-research-assistant.onrender.com/docs

Response from backend - https://drive.google.com/file/d/19BxAkDjR-WHgBb_2VlIk_h3dujfZBQI4/view?usp=sharing

## Why I built this
Job-hunting means reading through dozens of nearly-identical postings to spot patterns — which skills actually recur, which companies want what. This lets you ask that directly instead of reading each one manually.

## Architecture


## Tech stack
Python, FastAPI, Streamlit, FAISS, Google Gemini API (generation + embeddings), Adzuna API, deployed on Render (backend) and Streamlit Community Cloud (frontend).

## What I'd improve with more time
- Hybrid search (keyword + semantic) instead of pure vector similarity
- Re-ranking retrieved results before generation
- Formal RAG evaluation (RAGAS metrics: faithfulness, relevance)
- Persistent index that refreshes on a schedule instead of a static snapshot

## Run it locally
See `backend/` and `frontend/` folders — each has its own `requirements.txt`.
