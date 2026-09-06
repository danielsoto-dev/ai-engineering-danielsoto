"""Manual Responses API loop for the Session 12 estimation agent."""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import ValidationError

from app.agent.schemas import AgentEstimate, AgentRunResult, AgentStep, AgentTrace
from app.agent.tools import (
    TOOL_SCHEMAS,
    RetrievalBackend,
    dispatch_tool,
    retrieve_historical_budgets,
)

log = structlog.get_logger()

SYSTEM_PROMPT = """
You are a senior software estimator working from meeting transcripts and historical budgets.

Follow this method:
1. Read the transcript and identify every distinct component that must be estimated.
2. Call search_budgets separately for each component. Use focused queries and never send the whole
   transcript as one search.
3. Use only estimated_hours returned by search_budgets as reference_amounts. If a component has no
   useful references, search again with a better query. Never invent a historical amount.
4. After every component has references, call calculate_estimate once with the complete component
   list.
5. Return a final structured estimate. Its component hours and total must exactly match the output
   from calculate_estimate. Include the chunk ids used for each component and state assumptions.

Do not merge unrelated components to reduce the number of searches. Stop calling tools once the
complete estimate has been calculated.
""".strip()

FINAL_OUTPUT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "agent_estimate",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "estimated_hours": {"type": "number"},
                        "reference_chunk_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "estimated_hours",
                        "reference_chunk_ids",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
            "total_hours": {"type": "number"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["components", "total_hours", "assumptions", "confidence"],
        "additionalProperties": False,
    },
}


def _reasoning_summary(output: list[Any]) -> str:
    summaries: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", []) or []:
            text = getattr(summary, "text", None)
            if text:
                summaries.append(text)
    return " ".join(summaries) or "No reasoning summary was emitted for this response."


def _function_calls(output: list[Any]) -> list[Any]:
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def _parse_final_estimate(response: Any) -> AgentEstimate | None:
    output_text = getattr(response, "output_text", "")
    if not output_text:
        return None
    try:
        return AgentEstimate.model_validate_json(output_text)
    except ValidationError:
        log.exception("agent_final_output_invalid")
        return None


async def run_estimation_agent(
    transcript: str,
    *,
    client: Any,
    model: str = "gpt-5",
    reasoning_effort: str = "medium",
    max_iterations: int = 12,
    retrieval_backend: RetrievalBackend = retrieve_historical_budgets,
) -> AgentRunResult:
    """Run the reason, act, observe loop until the model returns its estimate."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    trace = AgentTrace()
    response = await client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=transcript,
        tools=TOOL_SCHEMAS,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        text={"format": FINAL_OUTPUT_FORMAT},
        parallel_tool_calls=True,
        store=True,
    )
    iterations = 1

    while calls := _function_calls(response.output):
        if iterations >= max_iterations:
            return AgentRunResult(
                estimate=None,
                trace=trace,
                iterations=iterations,
                stop_reason="max_iterations",
            )

        reasoning = _reasoning_summary(response.output)
        tool_outputs: list[dict[str, str]] = []
        for call in calls:
            arguments: dict[str, Any] = {}
            try:
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                result = await dispatch_tool(
                    call.name,
                    arguments,
                    retrieval_backend=retrieval_backend,
                )
            except Exception as exc:  # noqa: BLE001
                result = {"error": f"{type(exc).__name__}: {exc}"}
                log.warning("agent_tool_failed", tool=call.name, error=str(exc)[:200])

            observation = result.get("summary") or result.get("error") or json.dumps(result)[:240]
            trace.steps.append(
                AgentStep(
                    step=len(trace.steps) + 1,
                    reasoning=reasoning,
                    action=call.name,
                    arguments=arguments,
                    observation=observation,
                )
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = await client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOL_SCHEMAS,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            text={"format": FINAL_OUTPUT_FORMAT},
            parallel_tool_calls=True,
            store=True,
        )
        iterations += 1

    estimate = _parse_final_estimate(response)
    stop_reason = "completed" if estimate is not None else "invalid_final_output"
    log.info(
        "estimation_agent_finished",
        iterations=iterations,
        tool_calls=len(trace.steps),
        stop_reason=stop_reason,
    )
    return AgentRunResult(
        estimate=estimate,
        trace=trace,
        iterations=iterations,
        stop_reason=stop_reason,
    )
