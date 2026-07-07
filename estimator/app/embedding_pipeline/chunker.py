"""Structural chunker for budget JSONs (Session 7 pre-exercise, Step 3).

One budget component = one chunk. No overlap, no fixed-size splitting: the
document structure already gives us a natural chunk boundary.
"""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, Chunk

_ENCODING = tiktoken.encoding_for_model("text-embedding-3-small")


class JSONStructuralChunker:
    """Splits budgets into one chunk per component, with parent context
    prepended so a component doesn't lose track of its client/sector."""

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                text = self._render_text(budget, component)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": budget.client_metadata.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=len(_ENCODING.encode(text)),
                    )
                )
        return chunks

    @staticmethod
    def _render_text(budget: Budget, component) -> str:
        return (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {budget.client_metadata.sector} | "
            f"Year: {budget.year} | Main tech: {budget.main_technology}]\n\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
