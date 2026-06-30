# CAG Stress Test — Where Does It Break?

> **Status: template.** Numbers below are placeholders (`__`). Fill them after a
> real run:
>
> ```
> docker compose -f docker-compose.redis.yml up -d        # Redis Stack
> uv run uvicorn app.main:app --reload                     # backend on :8000
> uv run python -m evals.stress.run --http http://localhost:8000 \
>     --scenarios growing,pivot,contradiction \
>     --attachment-sizes 0,5,20,50,100 --repeats 3 \
>     --output evals/stress/results.csv
> ```
>
> Then read `results.csv` into the tables below. A full run is
> 3 scenarios × 5 sizes × 3 repeats × ~5 turns = **≥ 50 rows**.

## Setup

- **Single provider** (no cross-provider comparison): `PRIMARY_MODEL` only, so
  the curves are comparable. Pricing from `MODEL_COSTS` in `llm_wrapper.py`.
- **Budgets under test** (design contracts, in `evals/stress/run.py`):
  latency `4000 ms`, cost `0.01 USD/turn`.
- **Constants frozen** (the exercise measures, it does not optimise):
  `MAX_CONVERSATION_TURNS=6`, heuristic anchors, `MAX_ATTACHMENT_CHARS=60000`.

## 1. Summary table

One row per `scenario × attachment_size`, aggregated over repeats.

| scenario | attach KB | P50 latency (ms) | P95 latency (ms) | total cost (USD) | exact cache hit | semantic cache hit | mean fact recall |
|---|---|---|---|---|---|---|---|
| growing | 0 | __ | __ | __ | 0% | __% | __ |
| growing | 5 | __ | __ | __ | 0% | __% | __ |
| growing | 20 | __ | __ | __ | 0% | __% | __ |
| growing | 50 | __ | __ | __ | 0% | __% | __ |
| growing | 100 | __ | __ | __ | 0% | __% | __ |
| pivot | 0 | __ | __ | __ | 0% | __% | __ |
| pivot | … | … | … | … | … | … | … |
| contradiction | 0 | __ | __ | __ | 0% | __% | __ |
| contradiction | … | … | … | … | … | … | … |

> Note: the conversational path does **not** consult the caches
> (`cache_hit_kind` is always `none`), so exact-cache hit rate is `0%` by
> construction — every turn is context-dependent. Recorded for completeness.

## 2. The three curves (as tables)

### 2a. Latency vs tokens_in

| tokens_in (bucket) | mean latency (ms) | P95 latency (ms) |
|---|---|---|
| 0–1k | __ | __ |
| 1k–4k | __ | __ |
| 4k–16k | __ | __ |
| 16k+ | __ | __ |

### 2b. Cumulative cost vs turn_index (per scenario)

| turn_index | growing (USD) | pivot (USD) | contradiction (USD) |
|---|---|---|---|
| 1 | __ | __ | __ |
| 3 | __ | __ | __ |
| 6 | __ | __ | __ |
| 10 | __ | __ | __ |
| 20 | __ | — | — |

### 2c. Memory recall vs N (history length)

`MemoryDriftMetric` pass-rate for the turn-1 fact, checked at later turns.

| checked at turn N | growing (project_name recall) | pivot (stack recall) | contradiction (budget recall) |
|---|---|---|---|
| 3 | __ | __ | __ |
| 6 | __ | __ | __ |
| 10 | __ | __ | __ |
| 20 | __ | — | — |

## 3. Reading — where does my CAG break, and why?

> _Two paragraphs, ≥ 1 concrete quantitative claim each. Fill from the data._

**Paragraph 1 — the dominant degradation.**
TODO. e.g. _"From turn N=__ the project_name recall drops below 60%, while
latency stays within the 4000 ms budget until tokens_in exceeds ~__k around the
50 KB attachment. The system degrades through **memory loss before latency** —
the silent failure (recall 90→__%) that no red test catches is the dangerous
one."_

**Paragraph 2 — the limit that justifies RAG.**
TODO. e.g. _"The turn-20 cost is __× the turn-1 cost in the growing scenario
because every turn re-sends the full window + summary + metadata; at the 100 KB
attachment the transcript is truncated at MAX_ATTACHMENT_CHARS=60000 and the
attachment's content recall falls to __%. That truncation point — full context
by construction, content silently dropped — is exactly where CAG stops scaling
and retrieval (RAG) becomes the architectural answer."_

## Appendix — how the data is produced

- **Block 1**: `estimate_conversational()` emits one `turn_observed` event
  (13 fields) per turn; the last one is exposed on `GET /sessions/{id}` as
  `last_turn`.
- **Block 2**: `evals/stress/scenarios.py` — growing / pivot / contradiction,
  each turn carrying the fact later turns should recall.
- **Block 3**: `evals/stress/fixtures/build_pdfs.py` — deterministic PDFs at
  5/20/50/100 KB of extractable text (regenerated, not committed).
- **Block 4**: `evals/stress/metrics.py` — `LatencyBudgetMetric`,
  `CostBudgetMetric`, `MemoryDriftMetric` (reuse `MetricResult`).
- **Block 5**: `evals/stress/run.py` — drives the API, writes `results.csv`.
