"""
Query Intelligence Engine
--------------------------
Extracts structured intelligence from natural language queries.

Provider fallback chain:
  1. Anthropic (claude-haiku-3-5)  — preferred, uses tool_use for guaranteed structure
  2. Groq (llama-3.3-70b)          — fast fallback, uses JSON mode
  3. Google Gemini (gemini-1.5-flash) — second fallback, uses JSON mode

The Anthropic path uses the SDK's tool_use / function-calling feature, which is
more reliable than asking the model to "return JSON please" — the API enforces
the schema at the protocol level, not just via prompt.
"""

import json
import os
from models import ExtractedIntelligence

# ── Schema shared across all providers ────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "description": "Primary intent (e.g. 'market research', 'competitive analysis', 'talent search')",
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key entities: companies, technologies, geographies, people, industries",
        },
        "filters": {
            "type": "object",
            "description": "Implied constraints like region, funding stage, date range, company size",
        },
        "complexity": {
            "type": "string",
            "enum": ["simple", "moderate", "complex"],
            "description": "How complex the query is to answer",
        },
        "suggested_next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Actionable next steps or data sources to answer this query",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Extraction confidence score",
        },
    },
    "required": ["intent", "entities", "filters", "complexity", "suggested_next_steps", "confidence"],
}

SYSTEM_PROMPT = """You are a query intelligence engine for a corporate research platform.
Your job is to deeply analyze natural language research queries and extract structured intelligence.
Be precise: pull out every implied constraint, not just the obvious ones.
For 'battery technology startups in Southeast Asia', the region filter is SEA,
the stage filter is startup/early-stage, and the domain is clean energy / advanced materials."""


# ── Provider 1: Anthropic (tool_use) ──────────────────────────────────────────

def _extract_with_anthropic(query: str) -> tuple[ExtractedIntelligence, str]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[
            {
                "name": "analyze_query",
                "description": "Extract structured intelligence from a research query",
                "input_schema": EXTRACTION_SCHEMA,
            }
        ],
        # Force tool use — no free-text fallback, schema is enforced at protocol level
        tool_choice={"type": "tool", "name": "analyze_query"},
        messages=[{"role": "user", "content": query}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return ExtractedIntelligence(**tool_block.input), "anthropic/claude-haiku-4-5"


# ── Provider 2: Groq (JSON mode) ──────────────────────────────────────────────

def _extract_with_groq(query: str) -> tuple[ExtractedIntelligence, str]:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    prompt = f"""{SYSTEM_PROMPT}

Analyze this query and return ONLY a valid JSON object matching this schema:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}

Query: {query}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    data = json.loads(response.choices[0].message.content)
    return ExtractedIntelligence(**data), "groq/llama-3.3-70b-versatile"


# ── Provider 3: Google Gemini (JSON mode) ─────────────────────────────────────

def _extract_with_gemini(query: str) -> tuple[ExtractedIntelligence, str]:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"},
    )

    prompt = f"""{SYSTEM_PROMPT}

Analyze this query and return ONLY a valid JSON object matching this schema:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}

Query: {query}"""

    response = model.generate_content(prompt)
    data = json.loads(response.text)
    return ExtractedIntelligence(**data), "google/gemini-1.5-flash"


# ── Public interface: tries providers in order ─────────────────────────────────

PROVIDERS = [
    ("Anthropic", _extract_with_anthropic),
    ("Groq",      _extract_with_groq),
    ("Gemini",    _extract_with_gemini),
]


def extract_intelligence(query: str) -> tuple[ExtractedIntelligence, str]:
    """
    Extract structured intelligence from a natural language query.

    Tries Anthropic first (tool_use, schema-enforced), then Groq, then Gemini.
    Returns (ExtractedIntelligence, provider_string).
    Raises RuntimeError if all providers fail.
    """
    errors: list[str] = []

    for name, fn in PROVIDERS:
        try:
            result, provider_tag = fn(query)
            print(f"[intelligence] Used provider: {name}")
            return result, provider_tag
        except Exception as e:
            print(f"[intelligence] {name} failed: {e}")
            errors.append(f"{name}: {e}")

    raise RuntimeError(
        f"All LLM providers failed. Errors:\n" + "\n".join(errors)
    )