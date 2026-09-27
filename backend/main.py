from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask

app = FastAPI(title="AI/ML Job Research Assistant")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Job Research Assistant API is running"}

@app.post("/ask")
def ask_question(query: Query):
    return ask(query.question)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)