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