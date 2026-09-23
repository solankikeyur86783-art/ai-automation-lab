import os
import json
import re
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from shared import qdrant, embedder, COLLECTION
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

APPROVAL_MIN = 40
APPROVAL_MAX = 70


class LeadState(TypedDict, total=False):
    name: str
    email: str
    company: str
    message: str
    score: int
    category: str
    reason: str
    status: str
    used_knowledge_base: bool
    suggested_reply: str
    sources: List[str]


def _call_groq(system_prompt: str, user_prompt: str) -> str:
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
    return resp.json()["choices"][0]["message"]["content"]


def qualify_node(state: LeadState) -> LeadState:
    system_prompt = (
        "You are a B2B lead qualification assistant. Given a lead's details, respond ONLY "
        "with a JSON object with exactly these keys: score (integer 1-100), "
        "category (one of: hot, warm, cold), reason (one short sentence, under 20 words). "
        "No extra text."
    )
    user_prompt = (
        f"Name: {state.get('name', '')}\n"
        f"Company: {state.get('company', 'N/A')}\n"
        f"Message: {state.get('message', '')}"
    )
    raw = _call_groq(system_prompt, user_prompt)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {
            "score": 50, "category": "warm", "reason": "Could not parse AI response."
        }
    state["score"] = int(parsed.get("score", 50))
    state["category"] = parsed.get("category", "warm")
    state["reason"] = parsed.get("reason", "")
    return state


def route_decision(state: LeadState) -> str:
    score = state.get("score", 50)
    message = state.get("message", "").lower()
    if APPROVAL_MIN <= score <= APPROVAL_MAX:
        return "approval"
    question_words = ["price", "pricing", "cost", "how does", "what is", "sop", "policy", "?"]
    if any(w in message for w in question_words):
        return "knowledge"
    return "direct"


def knowledge_node(state: LeadState) -> LeadState:
    q_vec = list(embedder.embed([state.get("message", "")]))[0].tolist()
    results = qdrant.search(collection_name=COLLECTION, query_vector=q_vec, limit=3)
    state["used_knowledge_base"] = True
    if not results:
        state["suggested_reply"] = "No relevant company knowledge found for this query."
        state["sources"] = []
        return state
    context = "\n\n".join(f"[Source: {r.payload['source']}]\n{r.payload['text']}" for r in results)
    system_prompt = (
        "You are a sales assistant drafting a helpful reply to a lead using only the provided "
        "company knowledge. Keep it to 2-3 sentences, friendly and professional."
    )
    user_prompt = f"Context:\n{context}\n\nLead's message: {state.get('message', '')}"
    state["suggested_reply"] = _call_groq(system_prompt, user_prompt)
    state["sources"] = sorted(set(r.payload["source"] for r in results))
    return state


def approval_node(state: LeadState) -> LeadState:
    state["status"] = "pending_approval"
    return state


def finalize_node(state: LeadState) -> LeadState:
    state["status"] = "processed"
    return state


def build_graph():
    graph = StateGraph(LeadState)
    graph.add_node("qualify", qualify_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("approval", approval_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("qualify")
    graph.add_conditional_edges(
        "qualify",
        route_decision,
        {
            "approval": "approval",
            "knowledge": "knowledge",
            "direct": "finalize",
        },
    )
    graph.add_edge("knowledge", "finalize")
    graph.add_edge("approval", END)
    graph.add_edge("finalize", END)
    return graph.compile()


lead_agent = build_graph()
