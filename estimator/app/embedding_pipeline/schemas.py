"""Pydantic models for the embedding pipeline (Session 7 pre-exercise).

Mirrors the JSON schema of normalised historical budgets (Session 6 output).
No persistence here — chunks and embeddings live in memory for the duration
of a single request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Complexity = Literal["low", "medium", "high"]
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]


class ClientMetadata(BaseModel):
    name: str
    sector: Sector
    country: str


class BudgetComponent(BaseModel):
    component_id: str
    name: str
    description: str
    tech_stack: list[str]
    estimated_hours: float
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    budget_id: str
    client_metadata: ClientMetadata
    project_summary: str
    main_technology: str
    year: int
    total_estimated_hours: float
    components: list[BudgetComponent]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    token_count: int


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget]


class IngestStats(BaseModel):
    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
