"""RAGAS baseline for the grounded estimator.

    uv run python scripts/evaluate_generation.py

For each golden query: run the real RAG pipeline (hybrid retrieval → grounded
generation → citation verification), then score the result with the four RAGAS
metrics against the reference estimate.

``contexts`` are the chunks the generator actually received, so the retrieval
metrics describe the pipeline that produced the answer rather than a separate
lookup done for the benefit of the evaluator.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing settings-reading modules. The script runs on the
# host, so rewrite the compose hostname to the published port.
load_dotenv()
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://estimator:estimator@localhost:5433/estimator"
)
if "@postgres:" in os.environ.get("DATABASE_URL", ""):
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL"].replace(
        "@postgres:5432", "@localhost:5433"
    )

from app.db.session import get_sessionmaker  # noqa: E402
from app.dependencies import get_llm_wrapper  # noqa: E402
from app.services.grounded_estimation import GroundedEstimationService  # noqa: E402

GOLDEN_SET_PATH = Path("evals/retrieval_golden_set.json")
REPORT_PATH = Path("evals/generation_results.md")
RAW_PATH = Path("evals/generation_raw.json")

JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Everything from this heading onward is written by hand and survives a rerun.
NOTE_HEADING = "## Nota sobre los n\u00fameros"


async def run_pipeline() -> list[dict]:
    """Generate one grounded estimate per golden query."""
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    service = GroundedEstimationService(llm_wrapper=get_llm_wrapper())

    rows: list[dict] = []
    async with get_sessionmaker()() as session:
        for case in golden["queries"]:
            outcome = await service.estimate(session, case["query"], request_id=case["id"])
            report = outcome.citations

            rows.append(
                {
                    "query_id": case["id"],
                    "question": case["query"],
                    "answer": outcome.estimate.as_text(),
                    "contexts": outcome.context.contexts,
                    "ground_truth": case["ground_truth"],
                    "retrieved_chunk_ids": outcome.context.sorted_chunk_ids,
                    "citations": report.as_dict(),
                    "lines": [
                        {
                            "component": verdict.component,
                            "status": verdict.status,
                            "cited": verdict.cited_chunk_ids,
                            "dangling": verdict.dangling_chunk_ids,
                        }
                        for verdict in report.verdicts
                    ],
                    "total_hours": outcome.estimate.total_hours,
                    "ground_truth_hours": case["ground_truth_hours"],
                }
            )
            print(
                f"{case['id']}  lines={report.as_dict()['lines']}  "
                f"grounded={report.as_dict()['grounded']}  "
                f"dangling={report.as_dict()['dangling']}  "
                f"insufficient={report.as_dict()['insufficient_context']}  "
                f"hours={outcome.estimate.total_hours:g} (ref {case['ground_truth_hours']})"
            )
    return rows


def score(rows: list[dict]) -> dict[str, list[float]]:
    """Run RAGAS over the generated rows."""
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
            for row in rows
        ]
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ChatOpenAI(model=JUDGE_MODEL, temperature=0),
        embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    scores = result.to_pandas()
    return {name: [float(v) for v in scores[name]] for name in METRIC_NAMES}


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"  # NaN when a metric abstains


def write_report(rows: list[dict], metrics: dict[str, list[float]]) -> None:
    header = (
        "| Query | Faithfulness | Answer relevancy | Context precision | Context recall |\n"
        "| --- | --- | --- | --- | --- |"
    )
    lines = [
        "# Generation evaluation (RAGAS) — Session 11",
        "",
        (
            f"Judge: `{JUDGE_MODEL}`. Embeddings: `{EMBEDDING_MODEL}`. "
            f"Retrieval: hybrid (vector + lexical, RRF), top-k 5, no reranking — "
            f"config B, the best P@5 on the Session 10 golden set."
        ),
        "",
        "## Metrics",
        "",
        header,
    ]

    for index, row in enumerate(rows):
        cells = " | ".join(_fmt(metrics[name][index]) for name in METRIC_NAMES)
        lines.append(f"| {row['query_id']} | {cells} |")

    means = {
        name: statistics.mean([v for v in metrics[name] if v == v]) or 0.0 for name in METRIC_NAMES
    }
    mean_cells = " | ".join(_fmt(means[name]) for name in METRIC_NAMES)
    lines.append(f"| **Media** | {mean_cells} |")

    lines += [
        "",
        "## Citation verification",
        "",
        "| Query | Líneas | Fundamentadas | Colgantes | Sin datos | Horas | Horas ref. |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        c = row["citations"]
        lines.append(
            f"| {row['query_id']} | {c['lines']} | {c['grounded']} | {c['dangling']} "
            f"| {c['insufficient_context']} | {row['total_hours']:g} "
            f"| {row['ground_truth_hours']} |"
        )

    # The hand-written note is a deliverable; carry it over so a rerun refreshes
    # the numbers without deleting the analysis of them.
    if REPORT_PATH.exists():
        _, marker, note = REPORT_PATH.read_text(encoding="utf-8").partition(NOTE_HEADING)
        if marker:
            lines += ["", marker + note.rstrip("\n")]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> int:
    rows = await run_pipeline()
    print("\nscoring with RAGAS...")
    metrics = score(rows)

    for index, row in enumerate(rows):
        row["metrics"] = {name: metrics[name][index] for name in METRIC_NAMES}
    RAW_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_report(rows, metrics)
    print(f"report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
