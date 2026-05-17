"""
Query Intelligence API
========================
Endpoints for query analysis and an agent endpoint.

Endpoints:
  POST /queries          — submit a query, get structured intelligence back
  GET  /queries/{id}     — retrieve a stored query record
  GET  /queries          — list all stored queries
  GET  /health           — health check with provider status
  POST /agent            — talk to the agent in natural language
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import (
    QueryRequest,
    QueryRecord,
    QueryListItem,
    AgentRequest,
    AgentResponse,
)
from store import store
from intelligence import extract_intelligence, PROVIDERS
from agent import run_agent

load_dotenv()


# ── App setup ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  Query Intelligence API")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)
    available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("Anthropic ✓")
    if os.environ.get("GROQ_API_KEY"):
        available.append("Groq ✓")
    if os.environ.get("GEMINI_API_KEY"):
        available.append("Gemini ✓")
    if not available:
        print("  ⚠  No API keys found. Set at least one in .env")
    else:
        print(f"  Providers: {', '.join(available)}")
    print("=" * 50)
    yield


app = FastAPI(
    title="Query Intelligence API",
    description=(
        "Extracts structured intelligence from natural language research queries. "
        "Also exposes a /agent endpoint backed by an agentic loop with tools."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──  endpoints ─────────────────────────────────────────────────────────────

@app.post(
    "/queries",
    response_model=QueryRecord,
    status_code=201,
    summary="Submit a query",
    description="Accepts a natural language query, extracts structured intelligence via LLM, persists and returns the result.",
)
async def create_query(body: QueryRequest):
    try:
        intelligence, provider = extract_intelligence(body.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record = store.save(body.query, intelligence, provider)
    return record


@app.get(
    "/queries/{query_id}",
    response_model=QueryRecord,
    summary="Retrieve a query",
    description="Returns the stored query and its extracted intelligence by ID.",
)
async def get_query(query_id: str):
    record = store.get(query_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Query '{query_id}' not found.")
    return record


@app.get(
    "/queries",
    response_model=list[QueryListItem],
    summary="List all queries",
    description="Returns a summary list of all stored queries, newest first.",
)
async def list_queries():
    return [
        QueryListItem(
            id=r.id,
            query=r.query,
            intent=r.intelligence.intent,
            complexity=r.intelligence.complexity,
            created_at=r.created_at,
        )
        for r in store.list_all()
    ]


@app.get(
    "/health",
    summary="Health check",
    description="Returns API status and which LLM providers are configured.",
)
async def health():
    return {
        "status": "ok",
        "queries_stored": store.count(),
        "providers": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "groq":      bool(os.environ.get("GROQ_API_KEY")),
            "gemini":    bool(os.environ.get("GEMINI_API_KEY")),
        },
        "fallback_chain": [p[0] for p in PROVIDERS],
    }


# ── Agent endpoint ─────────────────────────────────────────────────────────────

@app.post(
    "/agent",
    response_model=AgentResponse,
    summary="Talk to the agent",
    description=(
        "Send a natural language message to the agent. "
        "The agent decides which tools to call, runs them, and replies. "
        "Example messages: "
        "'Analyze this: find fintech startups in India', "
        "'What queries have we stored?', "
        "'Retrieve query id <id>'"
    ),
)
async def agent_endpoint(body: AgentRequest):
    try:
        reply, tools_used, provider = run_agent(body.message)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return AgentResponse(reply=reply, tools_used=tools_used, provider=provider)