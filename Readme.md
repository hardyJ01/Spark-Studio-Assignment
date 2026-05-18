# Query Intelligence API

> Submit a messy natural language query. Get back clean, structured intelligence — instantly.

---

## What This Does

Imagine a researcher types:

> *"find battery technology startups in Southeast Asia that raised Series A"*

Most systems store that as plain text. **This system understands it.**

```json
{
  "intent": "market research",
  "entities": ["battery technology", "Southeast Asia", "startups"],
  "filters": { "region": "SEA", "stage": "Series A" },
  "complexity": "moderate",
  "suggested_next_steps": [
    "Search Crunchbase for SEA clean energy startups",
    "Filter by Series A funding rounds",
    "Cross-reference with patent filings in SG, ID, VN"
  ],
  "confidence": 0.92
}
```

It also has an **agent** — so instead of hitting an endpoint manually, you can just say:
> *"Analyze this query, then show me everything we've stored so far"*

And the agent figures out what to do, calls the right tools, and replies in plain English.

---

## Two Ways to Use It

### 1. Direct — just want the intelligence
```bash
POST /queries
{ "query": "find AI startups in Europe focused on healthcare" }
```

### 2. Agent — let it think for you
```bash
POST /agent
{ "message": "Analyze this: fintech companies in India, then summarize all stored queries" }
```

The agent can chain multiple actions in one go. You don't tell it *how* — it figures that out.

---

## Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `POST` | `/queries` | Analyze a query, save and return intelligence |
| `GET` | `/queries/{id}` | Retrieve a specific past query |
| `GET` | `/queries` | List all stored queries |
| `GET` | `/health` | Check which AI providers are live |
| `POST` | `/agent` | Talk to the agent in plain English |

📖 **Auto-generated interactive docs:** `http://localhost:8000/docs`

---

## How It's Built

```
├── main.py          →  API routes (FastAPI)
├── intelligence.py  →  extracts structured data from queries (LLM layer)
├── agent.py         →  decides what to do, calls tools, replies (reasoning layer)
├── store.py         →  saves and retrieves query records (data layer)
└── models.py        →  data shapes and validation (Pydantic)
```

Each file has exactly one job. Clean separation — easy to extend.

---

## The Two Interesting Technical Choices

### Tool Use over plain prompting
For the extraction, instead of asking the LLM *"please return JSON"* — which it might or might not follow — this uses the Anthropic SDK's **tool use** feature with a forced schema. The API enforces the output shape at the protocol level. No prompt hacking, no JSON parsing failures.

### Agentic loop
The `/agent` endpoint isn't just a smarter prompt. It runs a real loop — the model sees a user message, decides which tool to call, gets the result, and decides if it needs to call another tool before replying. It can chain `analyze_query → summarize_all` in a single request without being told to.

---

## Works Without a Paid API Key

Built with a three-provider fallback chain:

```
Anthropic  →  Groq (free)  →  Gemini (free)
```

If one provider is unavailable or unconfigured, it silently moves to the next. The `/health` endpoint shows exactly which providers are live. Every response includes a `provider` field so you always know which model answered.

---

## Setup (under 2 minutes)

```bash
# Clone and install
git clone https://github.com/yourname/query-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add at least one API key
cp .env.example .env

# Run
uvicorn main:app --reload
```

Free keys (no credit card needed):
- **Groq** → https://console.groq.com
- **Gemini** → https://aistudio.google.com

---

## Example Agent Conversation

```
You:   "Analyze this: renewable energy policy changes in Germany"

Agent: Called analyze_query →
       Intent: policy research
       Entities: renewable energy, Germany
       Filters: { sector: energy, region: Germany }
       Suggested next steps: Check Bundestag records, EU energy directives...
       
       Confidence: 0.89 · Powered by: groq/llama-3.3-70b-versatile
```

---

## 🔮 If I Had More Time (Roadmap to Production)

If I were scaling this service for production use, I would focus on observability, cost-reduction, and system resilience over simply adding more features:

1. **Semantic Caching Layer:** LLM calls are expensive and slow. I would implement semantic caching (e.g., Redis + a lightweight embedding model). If a user asks *"Looking for European AI startups"* and we already processed *"Find AI startups in Europe"*, the system should return the cached intelligence instantly without hitting the Anthropic API.

2. **LLM Observability & Tracing:** I would integrate LangSmith or OpenTelemetry to track token usage, generation latency, and most importantly, tool-call failure rates. You cannot improve an agent if you cannot measure where its reasoning fails.

3. **Async Job Queues (Celery/Redis):** Currently, the `/agent` endpoint blocks the HTTP request while the LLM "thinks". For complex multi-turn loops, this could cause browser timeouts. I would move the agent execution to a background worker and return a `job_id` so the client can poll for status.

4. **Human-in-the-Loop (HITL) Auditing:** Using the `confidence` score currently returned in the schema, I would implement a routing rule: any extraction with a confidence score below `0.75` gets sent to a dead-letter queue for human review before being committed to the main database.

---

*Built with FastAPI · Anthropic SDK · Groq · Gemini · Pydantic v2*
Author 
Hardipsinh Jadeja

