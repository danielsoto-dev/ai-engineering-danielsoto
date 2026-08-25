"""Citation verification on a real estimate, with an injected dangling citation.

    uv run python -m scripts.verify_citations_demo

Runs the pipeline for one golden query, prints the verification report, then
tampers with the estimate by pointing one line at a chunk_id that was never
retrieved and re-runs verification to show the failure surfacing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

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
from app.retrieval.citations import verify_citations  # noqa: E402
from app.services.grounded_estimation import GroundedEstimationService  # noqa: E402

FABRICATED_CHUNK_ID = "9999"
REPORT_PATH = Path("evals/citation_report.md")


def _render(title: str, report, chunk_ids: set[str]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"Chunks entregados al LLM: `{sorted(chunk_ids, key=int)}`",
        "",
        "| Línea | Estado | Cita | No recuperado |",
        "| --- | --- | --- | --- |",
    ]
    for verdict in report.verdicts:
        lines.append(
            f"| {verdict.component} | `{verdict.status}` "
            f"| {', '.join(verdict.cited_chunk_ids) or '—'} "
            f"| {', '.join(verdict.dangling_chunk_ids) or '—'} |"
        )
    lines += ["", f"Resumen: `{report.as_dict()}`", ""]
    return lines


async def main() -> int:
    case = json.loads(Path("evals/retrieval_golden_set.json").read_text(encoding="utf-8"))[
        "queries"
    ][0]
    service = GroundedEstimationService(llm_wrapper=get_llm_wrapper())

    async with get_sessionmaker()() as session:
        outcome = await service.estimate(session, case["query"], request_id=case["id"])

    out = [
        "# Citation verification report",
        "",
        f"Consulta `{case['id']}`: {case['query']}",
        "",
        "## Estimación generada",
        "",
        "```",
        outcome.estimate.as_text(),
        "```",
        "",
    ]
    out += _render("Verificación real", outcome.citations, outcome.context.chunk_ids)

    # Tamper: repoint the first grounded line at a chunk that was never retrieved.
    tampered = outcome.estimate.model_copy(deep=True)
    for item in tampered.line_items:
        if item.grounded:
            item.sources[0].chunk_id = FABRICATED_CHUNK_ID
            break

    tampered_report = verify_citations(
        tampered, outcome.context.chunk_ids, request_id=f"{case['id']}-TAMPERED"
    )
    out += _render(
        f"Verificación metiendo a mano una cita a un chunk inexistente (`{FABRICATED_CHUNK_ID}`)",
        tampered_report,
        outcome.context.chunk_ids,
    )

    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nreport written to {REPORT_PATH}")
    return 0 if outcome.citations.is_valid and not tampered_report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
