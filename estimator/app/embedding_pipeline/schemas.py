"""Pydantic models for the embedding pipeline (Session 7 pre-exercise).

Mirrors the JSON schema of normalised historical budgets (Session 6 output).
No persistence here — chunks and embeddings live in memory for the duration
of a single request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Complexity = Literal["low", "medium", "high"]
Sector = Literal[
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "media",
    "travel",
    "realestate",
    "energy",
    "publicsector",
    "hospitality",
]


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


class IngestDocumentRequest(BaseModel):
    source_path: str
    document_type: str
    content: Budget


class IngestDocumentResponse(BaseModel):
    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int


SearchMode = Literal["vector", "hybrid"]


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: SearchMode = "vector"
    rerank: bool = False
    # Recall depth fed to the reranker. Ignored when rerank is false.
    candidate_k: int = 50
    rrf_k: int = 60


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float | None = None
    metadata: dict
    lexical_rank_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    sources: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    k: int
    mode: SearchMode
    reranked: bool
    candidates_considered: int
    search_time_ms: int
    results: list[SearchResult]
