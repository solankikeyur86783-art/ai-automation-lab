import os
import re
import uuid
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client.models import PointStruct

from shared import qdrant, embedder, COLLECTION, ensure_collection
from agent import lead_agent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

app = FastAPI(title="AI Company Knowledge Brain")


@app.on_event("startup")
def startup():
    ensure_collection()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


class Document(BaseModel):
    source: str
    text: str


class IngestRequest(BaseModel):
    documents: List[Document]


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class LeadInput(BaseModel):
    name: str
    email: str
    company: Optional[str] = None
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    ensure_collection()
    points = []
    for doc in req.documents:
        chunks = chunk_text(doc.text)
        vectors = list(embedder.embed(chunks))
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={"source": doc.source, "chunk_index": i, "text": chunk},
            ))
    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"ingested_chunks": len(points)}


@app.post("/ask")
def ask(req: AskRequest):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set")
    q_vec = list(embedder.embed([req.question]))[0].tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=q_vec, limit=req.top_k)
    if not results:
        return {"answer": "I don't have information about that in the company knowledge base.", "sources": []}
    context = "\n\n".join(f"[Source: {r.payload['source']}]\n{r.payload['text']}" for r in results)
    system_prompt = (
        "You are a company knowledge assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say you don't have that information. "
        "Keep answers concise and always mention which source(s) you used."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}"
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    sources = sorted(set(r.payload["source"] for r in results))
    return {"answer": answer, "sources": sources}


@app.post("/agent/process-lead")
def process_lead(req: LeadInput):
    initial_state = {
        "name": req.name,
        "email": req.email,
        "company": req.company or "N/A",
        "message": req.message,
        "used_knowledge_base": False,
        "sources": [],
        "suggested_reply": "",
    }
    result = lead_agent.invoke(initial_state)
    return result
