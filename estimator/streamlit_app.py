"""Streamlit UI for the estimator — conversational (Session 5/6) frontend.

Talks to the FastAPI service over the /sessions API:

    POST /sessions                       -> open a multi-turn session
    POST /sessions/{id}/estimate         -> one turn (memory + attachments)
    POST /sessions/{id}/estimate-acb     -> same, with Actor-Critic-Boss review
    GET  /sessions/{id}                  -> session state + last-turn telemetry

The session_id is held in st.session_state so the conversation persists across
reruns. The endpoint URL comes from ESTIMATOR_API_BASE_URL (same .env as the
API), so the UI works against local uvicorn or docker-compose.
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
SESSIONS_URL = f"{API_BASE_URL.rstrip('/')}/sessions"

_GUARDRAIL_REASONS = {
    "moderation": "Content was flagged by moderation.",
    "prompt_injection": "The text looks like a prompt-injection attempt.",
    "pii": "The description contains personal data (email, phone or IBAN).",
}

st.set_page_config(page_title="Software Estimator — Conversational", page_icon="💬")
st.title("Software Estimator 💬")
st.caption("Multi-turn estimation with memory, attachments and optional Actor-Critic-Boss review.")


# --- session lifecycle -------------------------------------------------------


def _new_session() -> str:
    resp = httpx.post(SESSIONS_URL, timeout=10.0)
    resp.raise_for_status()
    return resp.json()["session_id"]


def _ensure_session() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = _new_session()
        st.session_state.turns = []  # list of (user_text, result_dict, acb_or_none)
    return st.session_state.session_id


def _reset_session() -> None:
    for key in ("session_id", "turns"):
        st.session_state.pop(key, None)


# --- sidebar: session state + telemetry --------------------------------------

with st.sidebar:
    st.header("Session")
    if st.button("🔄 New conversation", use_container_width=True):
        _reset_session()
        st.rerun()

    sid = _ensure_session()
    st.code(sid, language="text")

    try:
        info = httpx.get(f"{SESSIONS_URL}/{sid}", timeout=10.0).json()
    except httpx.HTTPError:
        info = {}

    meta = info.get("metadata") or {}
    st.subheader("Project memory")
    st.markdown(f"**Name:** {meta.get('project_name') or '—'}")
    st.markdown(f"**Team size:** {meta.get('assumed_team_size') or '—'}")
    techs = meta.get("mentioned_technologies") or []
    st.markdown(f"**Tech:** {', '.join(techs) if techs else '—'}")
    if meta.get("agreed_scope"):
        st.caption(meta["agreed_scope"])

    st.subheader("Context window")
    c1, c2 = st.columns(2)
    c1.metric("Messages", info.get("message_count", 0))
    c2.metric("Anchors", info.get("anchors_count", 0))
    st.caption(
        f"Summary chars: {info.get('summary_chars', 0)} · Tier: {info.get('last_resolved_tier') or '—'}"
    )

    last = info.get("last_turn")
    if last:
        st.subheader("Last turn telemetry")
        st.caption(
            f"turn #{last['turn_index']} · {last['latency_ms']} ms · "
            f"{last['tokens_in']}→{last['tokens_out']} tok · "
            f"${last['cost_usd']:.4f} · cache={last['cache_hit_kind']}"
        )


# --- conversation history (above the form) -----------------------------------


def _render_result(result: dict, acb: dict | None) -> None:
    if acb:
        decision = acb.get("final_decision", "?")
        runs = acb.get("iterations_run", 0)
        st.info(f"🤝 Actor-Critic-Boss: **{decision}** after {runs} iteration(s)")

    st.write(result["summary"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Weeks", result["total_duration_weeks"])
    col2.metric("Cost (EUR)", f"€{result['total_cost_eur']:,}")
    col3.metric("Confidence", f"{result['confidence_pct']}%")
    st.table(result["phases"])

    if acb and acb.get("iterations"):
        with st.expander("🔍 Critic trace"):
            for it in acb["iterations"]:
                st.markdown(
                    f"**Iter {it['iteration']}** → decision `{it['decision_after']}` · "
                    f"verdict `{it['critic_verdict']}` ({it['critic_confidence']}%)"
                )
                for issue in it.get("issue_summary", []):
                    st.markdown(f"- {issue}")


def _render_agent_result(body: dict) -> None:
    estimate = body.get("estimate")
    if estimate is None:
        st.warning(f"Agent stopped without an estimate: {body.get('stop_reason', 'unknown')}")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total hours", estimate["total_hours"])
    col2.metric("Components", len(estimate["components"]))
    col3.metric("Iterations", body["iterations"])
    st.caption(f"Confidence: {estimate['confidence']} · Stop reason: {body['stop_reason']}")

    rows = [
        {
            "Component": component["name"],
            "Hours": component["estimated_hours"],
            "Reference chunks": ", ".join(
                str(chunk_id) for chunk_id in component["reference_chunk_ids"]
            ),
            "Rationale": component["rationale"],
        }
        for component in estimate["components"]
    ]
    st.table(rows)

    if estimate["assumptions"]:
        st.markdown("**Assumptions**")
        for assumption in estimate["assumptions"]:
            st.markdown(f"- {assumption}")

    steps = body.get("trace", {}).get("steps", [])
    with st.expander(f"Agent tool trace ({len(steps)} calls)"):
        if not steps:
            st.caption("No tools were called.")
        for step in steps:
            st.markdown(f"**Step {step['step']}: `{step['action']}`**")
            st.write(step["reasoning"])
            st.code(json.dumps(step["arguments"], indent=2), language="json")
            st.caption(f"Observation: {step['observation']}")


for turn in st.session_state.get("turns", []):
    user_text, result, acb, *mode_flag = turn
    is_agent = bool(mode_flag and mode_flag[0])
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        if is_agent:
            _render_agent_result(result)
        else:
            _render_result(result, acb)


# --- input form --------------------------------------------------------------

mode = st.radio(
    "Estimation mode",
    ["Conversational", "Session 12 agent"],
    horizontal=True,
)
use_agent = mode == "Session 12 agent"

with st.form("turn_form", clear_on_submit=True):
    transcript = st.text_area(
        "Describe the project (or add to the conversation)",
        height=150,
        placeholder="Each message is a turn — the service remembers the previous ones.",
        help="Between 20 and 80000 characters.",
    )
    if use_agent:
        st.caption("The agent searches historical budgets and calculates the result in hours.")
        attachment = None
        use_acb = False
        project_type = detail_level = output_format = None
    else:
        col_a, col_b, col_c = st.columns(3)
        project_type = col_a.selectbox("Project type", [t.value for t in ProjectType], index=1)
        detail_level = col_b.selectbox("Detail", [d.value for d in DetailLevel], index=1)
        output_format = col_c.selectbox("Format", [f.value for f in OutputFormat], index=0)
        attachment = st.file_uploader("Attachment (optional)", type=["pdf", "docx"])
        use_acb = st.toggle("Use Actor-Critic-Boss (slower, higher quality)", value=False)
    submitted = st.form_submit_button("Send turn", type="primary")


if submitted:
    if len(transcript.strip()) < 20:
        st.error("The description must be at least 20 characters long.")
    else:
        if use_agent:
            endpoint = f"{API_BASE_URL.rstrip('/')}/api/v1/agent/estimate"
            request_options = {"json": {"transcript": transcript.strip()}}
        else:
            endpoint = (
                f"{SESSIONS_URL}/{sid}/estimate-acb"
                if use_acb
                else f"{SESSIONS_URL}/{sid}/estimate"
            )
            data = {
                "transcript": transcript.strip(),
                "project_type": project_type,
                "detail_level": detail_level,
                "output_format": output_format,
            }
            files = None
            if attachment is not None:
                files = {"attachments": (attachment.name, attachment.getvalue(), attachment.type)}
            request_options = {"data": data, "files": files}

        with st.spinner("Estimating…"):
            try:
                resp = httpx.post(
                    endpoint,
                    **request_options,
                    timeout=httpx.Timeout(180.0, connect=10.0),
                )
                resp.raise_for_status()
                body = resp.json()
            except httpx.HTTPStatusError as exc:
                detail = None
                try:
                    detail = exc.response.json().get("detail")
                except Exception:  # noqa: BLE001
                    pass
                if isinstance(detail, dict) and "reason" in detail:
                    reason = detail["reason"]
                    st.error(
                        f"Rejected ({reason}): {detail.get('message', '')}\n\n"
                        f"{_GUARDRAIL_REASONS.get(reason, '')}"
                    )
                else:
                    st.error(f"Service returned {exc.response.status_code}: {exc.response.text}")
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{endpoint}`: {exc}")
            else:
                if use_agent:
                    st.session_state.turns.append((transcript.strip(), body, None, True))
                else:
                    st.session_state.turns.append(
                        (transcript.strip(), body["result"], body.get("acb"), False)
                    )
                st.rerun()
