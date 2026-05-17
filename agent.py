"""
Query Intelligence Agent
-------------------------
A Claude-powered agent that reasons over your queries using three tools:

  1. analyze_query   — extract intelligence from a new natural language query
  2. get_query       — retrieve a previously stored query by ID
  3. summarize_all   — find patterns across all stored queries

How it works (agentic loop):
  User sends a message → Agent thinks → decides which tool to call →
  gets tool result → thinks again → maybe calls another tool →
  eventually writes a final reply.

This is different from a normal API call because the agent DECIDES
what to do — it's not just running a fixed function. Give it
"compare the last two queries" and it'll call get_query twice,
then reason over both results before replying.

Provider fallback: Anthropic → Groq → Gemini (same as intelligence.py)
"""

import json
import os
from store import store
from intelligence import extract_intelligence

# ── Tool definitions (what the agent can do) ──────────────────────────────────
#
# Think of these as the agent's "hands". Each tool has:
#   - a name   (how the agent refers to it)
#   - a description  (how the agent decides WHEN to use it)
#   - an input_schema  (what inputs it needs)

AGENT_TOOLS = [
    {
        "name": "analyze_query",
        "description": (
            "Extract structured intelligence from a natural language research query. "
            "Use this when the user wants to analyze a new query. "
            "Returns intent, entities, filters, complexity, and suggested next steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language research query to analyze",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_query",
        "description": (
            "Retrieve a previously stored query record by its ID. "
            "Use this when the user asks about a specific past query or provides an ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query_id": {
                    "type": "string",
                    "description": "The UUID of the stored query",
                }
            },
            "required": ["query_id"],
        },
    },
    {
        "name": "summarize_all",
        "description": (
            "Summarize and find patterns across all stored queries. "
            "Use this when the user asks what queries exist, wants an overview, "
            "or asks for trends across multiple queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

AGENT_SYSTEM_PROMPT = """You are a Query Intelligence Agent for a corporate research platform.
You help users analyze research queries and retrieve insights from past queries.

You have three tools available:
- analyze_query: to extract structured intelligence from a new query
- get_query: to look up a specific past query by ID  
- summarize_all: to find patterns across all stored queries

Always be concise and specific. When you analyze a query, highlight the most
interesting intelligence — especially filters and suggested next steps.
If a query ID is not found, say so clearly."""


# ── Tool executor — runs the actual logic when agent picks a tool ──────────────

def _run_tool(tool_name: str, tool_input: dict) -> str:
    """
    Executes the tool the agent chose and returns the result as a string.
    The agent reads this string to decide what to do next.
    """
    if tool_name == "analyze_query":
        query_text = tool_input["query"]
        intelligence, provider = extract_intelligence(query_text)
        record = store.save(query_text, intelligence, provider)
        return json.dumps({
            "id": record.id,
            "intent": record.intelligence.intent,
            "entities": record.intelligence.entities,
            "filters": record.intelligence.filters,
            "complexity": record.intelligence.complexity,
            "suggested_next_steps": record.intelligence.suggested_next_steps,
            "confidence": record.intelligence.confidence,
            "provider": record.provider,
        }, indent=2)

    elif tool_name == "get_query":
        query_id = tool_input["query_id"]
        record = store.get(query_id)
        if not record:
            return json.dumps({"error": f"No query found with ID '{query_id}'"})
        return json.dumps({
            "id": record.id,
            "query": record.query,
            "intent": record.intelligence.intent,
            "entities": record.intelligence.entities,
            "filters": record.intelligence.filters,
            "complexity": record.intelligence.complexity,
            "suggested_next_steps": record.intelligence.suggested_next_steps,
            "created_at": record.created_at,
        }, indent=2)

    elif tool_name == "summarize_all":
        all_records = store.list_all()
        if not all_records:
            return json.dumps({"message": "No queries stored yet."})
        summary = [
            {
                "id": r.id,
                "query": r.query,
                "intent": r.intelligence.intent,
                "complexity": r.intelligence.complexity,
                "entity_count": len(r.intelligence.entities),
                "created_at": r.created_at,
            }
            for r in all_records
        ]
        return json.dumps({"total": len(summary), "queries": summary}, indent=2)

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Agentic loop for Anthropic ─────────────────────────────────────────────────

def _run_agent_anthropic(message: str) -> tuple[str, list[str]]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": message}]
    tools_used = []

    # Agentic loop — keeps going until the agent stops calling tools
    # Max 5 iterations so it can never spin forever
    for _ in range(5):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        # stop_reason = "end_turn"    → agent is done, has a final reply
        # stop_reason = "tool_use"    → agent wants to call a tool
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return final_text, tools_used

        # Agent chose to use a tool — find which one
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_block:
            break

        tools_used.append(tool_block.name)
        tool_result = _run_tool(tool_block.name, tool_block.input)

        # Feed the tool result back so the agent can reason over it
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": tool_result,
            }],
        })

    return "Agent reached maximum iterations without a final answer.", tools_used


# ── Simple single-turn agent for Groq / Gemini ────────────────────────────────
#
# Groq and Gemini also support tool/function calling, but to keep the fallback
# simple and dependency-light, we use a "plan then execute" approach:
#   1. Ask the model which tool to call and with what input (JSON)
#   2. Run the tool
#   3. Ask the model to write a final reply based on the result
#
# This is single-turn (one tool call max) vs Anthropic's multi-turn loop,
# which is fine for the vast majority of user requests.

_GROQ_PLAN_PROMPT = """You are a query intelligence agent. Based on the user's message,
decide which tool to call. Reply ONLY with a JSON object like:
{{"tool": "analyze_query", "input": {{"query": "..."}}}}
or
{{"tool": "get_query", "input": {{"query_id": "..."}}}}
or
{{"tool": "summarize_all", "input": {{}}}}

User message: {message}"""

_GROQ_REPLY_PROMPT = """You are a query intelligence agent. Based on the tool result below,
write a helpful, concise reply to the user.

User message: {message}
Tool used: {tool}
Tool result: {result}

Write your reply now:"""


def _run_agent_groq(message: str) -> tuple[str, list[str]]:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    # Step 1 — plan: which tool to call?
    plan_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": _GROQ_PLAN_PROMPT.format(message=message)}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    plan = json.loads(plan_response.choices[0].message.content)
    tool_name = plan.get("tool", "summarize_all")
    tool_input = plan.get("input", {})

    # Step 2 — execute the tool
    tool_result = _run_tool(tool_name, tool_input)

    # Step 3 — write a final reply
    reply_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": _GROQ_REPLY_PROMPT.format(
                message=message, tool=tool_name, result=tool_result
            ),
        }],
        temperature=0.3,
    )
    reply = reply_response.choices[0].message.content
    return reply, [tool_name]


def _run_agent_gemini(message: str) -> tuple[str, list[str]]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"},
    )

    # Step 1 — plan
    plan_response = model.generate_content(
        _GROQ_PLAN_PROMPT.format(message=message)
    )
    plan = json.loads(plan_response.text)
    tool_name = plan.get("tool", "summarize_all")
    tool_input = plan.get("input", {})

    # Step 2 — execute
    tool_result = _run_tool(tool_name, tool_input)

    # Step 3 — reply (plain text this time)
    reply_model = genai.GenerativeModel("gemini-1.5-flash")
    reply_response = reply_model.generate_content(
        _GROQ_REPLY_PROMPT.format(
            message=message, tool=tool_name, result=tool_result
        )
    )
    return reply_response.text, [tool_name]


# ── Public interface — same fallback chain as intelligence.py ──────────────────

AGENT_PROVIDERS = [
    ("Anthropic", _run_agent_anthropic, "anthropic/claude-haiku-4-5"),
    ("Groq",      _run_agent_groq,      "groq/llama-3.3-70b-versatile"),
    ("Gemini",    _run_agent_gemini,    "google/gemini-1.5-flash"),
]


def run_agent(message: str) -> tuple[str, list[str], str]:
    """
    Run the agent on a user message.
    Returns (reply_text, tools_used_list, provider_string).
    Tries Anthropic first, then Groq, then Gemini.
    """
    errors = []
    for name, fn, provider_tag in AGENT_PROVIDERS:
        try:
            reply, tools_used = fn(message)
            print(f"[agent] Used provider: {name}")
            return reply, tools_used, provider_tag
        except Exception as e:
            print(f"[agent] {name} failed: {e}")
            errors.append(f"{name}: {e}")

    raise RuntimeError("All agent providers failed:\n" + "\n".join(errors))