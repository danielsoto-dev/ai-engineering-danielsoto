#!/usr/bin/env python3
"""Run the persisted Session 13 graph and save its complete node trace."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from app.agent.schemas import GraphEstimationResult  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.graph.runner import EstimationGraphRunner  # noqa: E402
from app.graph.state import EstimationGraphContext  # noqa: E402
from app.observability import configure_observability  # noqa: E402


def render_result(result: GraphEstimationResult) -> str:
    lines = [
        "=" * 78,
        "LANGGRAPH EXECUTION TRACE",
        "=" * 78,
        f"thread_id: {result.estimation_id}",
        f"status: {result.status}",
        "",
    ]
    for step in result.trace.steps:
        lines.extend(
            [
                f"SPAN {step.step}: node: {step.action}",
                f"  reasoning: {step.reasoning}",
                f"  arguments: {step.arguments}",
                f"  observation: {step.observation}",
                "",
            ]
        )
    lines.extend(
        [
            "=" * 78,
            "FINAL ESTIMATE",
            "=" * 78,
            result.estimate.model_dump_json(indent=2),
        ]
    )
    if result.errors:
        lines.extend(["", "ERRORS", *[f"- {error}" for error in result.errors]])
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        print("OPENAI_API_KEY is required.", file=sys.stderr)
        return 1

    configure_observability()
    estimation_id = args.estimation_id or str(uuid4())
    result = await EstimationGraphRunner(database_url=settings.DATABASE_URL).run(
        transcript=transcript_path.read_text(encoding="utf-8"),
        estimation_id=estimation_id,
        context=EstimationGraphContext(
            client=AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
            model=args.model,
            reasoning_effort=args.effort,
        ),
    )
    rendered = render_result(result)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nTrace written to {output_path}")
    return 0 if result.status == "validated" else 2


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Session 13 estimation graph.")
    parser.add_argument("transcript", help="Path to a meeting transcript.")
    parser.add_argument("--model", default=settings.AGENT_MODEL)
    parser.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high"],
        default=settings.AGENT_REASONING_EFFORT,
    )
    parser.add_argument("--estimation-id", help="Optional persistent LangGraph thread ID.")
    parser.add_argument("--output", help="Write the complete node trace to this file.")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
