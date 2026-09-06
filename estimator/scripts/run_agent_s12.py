#!/usr/bin/env python3
"""Run the Session 12 estimation agent and print its full trace."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.loop import run_estimation_agent  # noqa: E402
from app.agent.schemas import (  # noqa: E402
    AgentRunResult,
    HistoricalBudgetItem,
    SearchBudgetsArgs,
)
from app.config import get_settings  # noqa: E402

STUB_PATH = REPO_ROOT / "exercises" / "session-12" / "reference_retrieval.py"


def load_stub_backend():
    """Load the exercise fallback without adding it to the application package."""
    spec = importlib.util.spec_from_file_location("session_12_reference_retrieval", STUB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load retrieval stub from {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def stub_backend(args: SearchBudgetsArgs) -> list[HistoricalBudgetItem]:
        filters = args.filters.model_dump() if args.filters else None
        raw_items = module.search_budgets_stub(args.query, filters)
        return [HistoricalBudgetItem.model_validate(item) for item in raw_items]

    return stub_backend


def render_result(result: AgentRunResult) -> str:
    lines = [
        "=" * 78,
        "AGENT TRACE",
        "=" * 78,
        result.trace.render(),
        "",
        "=" * 78,
        f"FINAL ESTIMATE (iterations={result.iterations}, stop={result.stop_reason})",
        "=" * 78,
    ]
    if result.estimate is None:
        lines.append("No valid structured estimate was produced.")
    else:
        lines.append(result.estimate.model_dump_json(indent=2))
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"Transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        print("OPENAI_API_KEY is required for the Responses API.", file=sys.stderr)
        return 1

    transcript = transcript_path.read_text(encoding="utf-8")
    backend = load_stub_backend() if args.stub else None
    run_options = {
        "client": AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        "model": args.model,
        "reasoning_effort": args.effort,
        "max_iterations": args.max_iterations,
    }
    if backend is not None:
        run_options["retrieval_backend"] = backend

    result = await run_estimation_agent(transcript, **run_options)
    rendered = render_result(result)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nTrace written to {output_path}")

    return 0 if result.stop_reason == "completed" else 1


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Session 12 estimation agent.")
    parser.add_argument("transcript", help="Path to a meeting transcript.")
    parser.add_argument("--model", default=settings.AGENT_MODEL)
    parser.add_argument(
        "--effort",
        choices=["minimal", "low", "medium", "high"],
        default=settings.AGENT_REASONING_EFFORT,
    )
    parser.add_argument("--max-iterations", type=int, default=settings.AGENT_MAX_ITERATIONS)
    parser.add_argument("--stub", action="store_true", help="Use the fallback retrieval data.")
    parser.add_argument("--output", help="Write the trace and estimate to this file.")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
