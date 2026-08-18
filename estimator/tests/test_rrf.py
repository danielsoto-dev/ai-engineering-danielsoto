"""Reciprocal Rank Fusion behaviour."""

from __future__ import annotations

from app.retrieval.hybrid import RetrievedChunk, reciprocal_rank_fusion


def _chunk(chunk_id: int, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        chunk_type="budget_component",
        content=f"chunk {chunk_id}",
        metadata={},
        sources=[source],
    )


def test_chunk_found_by_both_branches_outranks_single_branch_leaders():
    vector = [_chunk(1, "vector"), _chunk(2, "vector")]
    lexical = [_chunk(3, "lexical"), _chunk(2, "lexical")]

    fused = reciprocal_rank_fusion([vector, lexical])

    # Chunk 2 is second in both lists, so it accumulates two contributions and
    # beats chunks 1 and 3, each of which only leads one list.
    assert fused[0].chunk_id == 2
    assert sorted(fused[0].sources) == ["lexical", "vector"]


def test_fusion_is_rank_based_not_score_based():
    # Wildly different branch scores must not affect the outcome: only position
    # counts, which is what lets cosine distance and ts_rank_cd be merged.
    vector = [_chunk(1, "vector")]
    vector[0].vector_distance = 0.001
    lexical = [_chunk(2, "lexical")]
    lexical[0].lexical_rank_score = 999.0

    fused = reciprocal_rank_fusion([vector, lexical])

    assert {c.chunk_id for c in fused} == {1, 2}
    assert fused[0].fusion_score == fused[1].fusion_score


def test_smoothing_constant_flattens_differences_between_top_ranks():
    ranking = [_chunk(i, "vector") for i in range(1, 4)]

    tight = reciprocal_rank_fusion([ranking], k=1)
    loose = reciprocal_rank_fusion([ranking], k=1000)

    tight_gap = tight[0].fusion_score - tight[1].fusion_score
    loose_gap = loose[0].fusion_score - loose[1].fusion_score
    assert loose_gap < tight_gap
