"""
In-memory store for query records.

Design note: Intentionally kept as a plain class with a dict backend.
In production, swap _store for a SQLAlchemy async session or Redis client —
the interface stays identical. No ORM boilerplate needed for a 60-min assessment,
and it keeps the focus on the intelligence layer, not the plumbing.
"""

import uuid
from datetime import datetime, timezone
from models import QueryRecord, ExtractedIntelligence


class QueryStore:
    def __init__(self):
        self._store: dict[str, QueryRecord] = {}

    def save(self, query: str, intelligence: ExtractedIntelligence, provider: str) -> QueryRecord:
        record = QueryRecord(
            id=str(uuid.uuid4()),
            query=query,
            intelligence=intelligence,
            provider=provider,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store[record.id] = record
        return record

    def get(self, query_id: str) -> QueryRecord | None:
        return self._store.get(query_id)

    def list_all(self) -> list[QueryRecord]:
        return sorted(self._store.values(), key=lambda r: r.created_at, reverse=True)

    def count(self) -> int:
        return len(self._store)


# Singleton — shared across the app lifetime
store = QueryStore()