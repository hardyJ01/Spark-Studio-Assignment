from pydantic import BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Natural language research query",
        examples=["find battery technology startups in Southeast Asia"],
    )


class ExtractedIntelligence(BaseModel):
    intent: str = Field(
        description="Primary intent behind the query (e.g. market research, competitive analysis)"
    )
    entities: list[str] = Field(
        description="Key entities mentioned — companies, technologies, geographies, people"
    )
    filters: dict[str, Any] = Field(
        description="Implied constraints extracted from the query (region, stage, date range, etc.)"
    )
    complexity: str = Field(
        description="Query complexity: simple | moderate | complex"
    )
    suggested_next_steps: list[str] = Field(
        description="Actionable next steps or tools that would help answer this query"
    )
    confidence: float = Field(
        description="Confidence score of the extraction, between 0.0 and 1.0"
    )


class QueryRecord(BaseModel):
    id: str
    query: str
    intelligence: ExtractedIntelligence
    provider: str = Field(description="LLM provider used for extraction")
    created_at: str


class QueryListItem(BaseModel):
    id: str
    query: str
    intent: str
    complexity: str
    created_at: str


# ── Agent models ───────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=3,
        description="A natural language message to the agent",
        examples=[
            "Analyze this query: find AI startups in Europe",
            "What queries have we stored so far?",
            "Retrieve query id abc-123",
        ],
    )


class AgentResponse(BaseModel):
    reply: str = Field(description="Agent's natural language response")
    tools_used: list[str] = Field(description="Names of tools the agent called")
    provider: str = Field(description="LLM provider that powered the agent")