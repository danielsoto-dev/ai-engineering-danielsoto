"""RAG estimation with per-line, verifiable citation.

Closes the loop the earlier sessions left open: retrieval and generation were
two working halves that never met. Here the retrieved chunks become the prompt
context, their ids become the citation vocabulary, and the same id set is what
verification checks the model's citations against.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.prompts.loader import render_grounded_estimation_prompt
from app.retrieval.citations import CitationReport, verify_citations
from app.retrieval.context import AssembledContext, assemble_context
from app.schemas.estimation import GroundedEstimate
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()


@dataclass
class GroundedEstimationOutcome:
    estimate: GroundedEstimate
    citations: CitationReport
    context: AssembledContext
    meta: dict


class GroundedEstimationService:
    """Retrieve, generate with attribution, then verify the citations."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        embedder: OpenAIEmbedder | None = None,
        prompt_version: str = "v1",
        top_k: int = 5,
        use_rerank: bool = False,
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.embedder = embedder or OpenAIEmbedder()
        self.prompt_version = prompt_version
        self.top_k = top_k
        self.use_rerank = use_rerank

    async def estimate(
        self, session, description: str, *, request_id: str | None = None
    ) -> GroundedEstimationOutcome:
        context = await assemble_context(
            session,
            description,
            self.embedder.embed_one(description),
            top_k=self.top_k,
            use_rerank=self.use_rerank,
        )

        system_prompt, user_prompt = render_grounded_estimation_prompt(
            description=description, context=context.text, version=self.prompt_version
        )
        estimate, meta = self.llm_wrapper.complete_structured_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=GroundedEstimate,
        )

        # Verified against the ids actually rendered into the prompt, never a
        # fresh retrieval — a re-query could return a different set and bless a
        # citation the model was never shown.
        citations = verify_citations(estimate, context.chunk_ids, request_id=request_id)

        return GroundedEstimationOutcome(
            estimate=estimate, citations=citations, context=context, meta=meta
        )
