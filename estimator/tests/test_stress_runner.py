"""Tests for the stress runner's pure helpers (Block 5).

We do NOT exercise the LLM path here — that's the real run's job. We check the
deterministic glue: scenario integrity, snapshot flattening, the PDF builder's
size calibration, and that the CSV schema matches what the report consumes.
"""

from __future__ import annotations

from evals.stress.fixtures.build_pdfs import _body_text
from evals.stress.run import CSV_FIELDS, _snapshot_text_fields
from evals.stress.scenarios import SCENARIOS, get_scenarios


def test_every_fact_is_in_its_own_transcript():
    # If a fact isn't even in the turn that introduces it, MemoryDrift can
    # never pass — the scenario would be unmeasurable.
    for scenario in SCENARIOS.values():
        for turn in scenario.turns:
            assert turn.fact_to_remember.lower() in turn.transcript.lower()


def test_get_scenarios_unknown_name_raises():
    import pytest

    with pytest.raises(KeyError):
        get_scenarios(["does_not_exist"])


def test_snapshot_text_fields_flattens_metadata():
    info = {
        "metadata": {
            "project_name": "Nimbus",
            "agreed_scope": "CRM with pipeline",
            "mentioned_technologies": ["React", "Postgres"],
        }
    }
    snap = _snapshot_text_fields(info)
    text = snap["metadata_text"].lower()
    assert "nimbus" in text and "react" in text and "postgres" in text


def test_snapshot_text_fields_handles_empty():
    snap = _snapshot_text_fields({})
    assert snap["metadata_text"] == ""
    assert snap["summary_text"] == "" and snap["anchors_text"] == ""


def test_body_text_hits_target_length():
    text = _body_text(5000)
    assert len(text) == 5000
    # Marker must survive so the content is greppable in a response.
    assert "Falcon" in text


def test_csv_fields_cover_the_thirteen_turn_observed_keys():
    # The 13 Block-1 fields must all be columns, else the report loses data.
    required = {
        "turn_index", "session_id", "enriched_transcript_chars",
        "attachments_total_chars", "messages_in_window", "anchors_count",
        "summary_chars", "tokens_in", "tokens_out", "cost_usd",
        "latency_ms", "cache_hit_kind", "last_resolved_tier",
    }
    assert required.issubset(set(CSV_FIELDS))
