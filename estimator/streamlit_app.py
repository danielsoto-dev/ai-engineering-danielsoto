"""Streamlit form UI for the estimator.

Streamlit acts as an HTTP client of the FastAPI service: it submits a typed
``EstimationRequest`` to ``POST /api/v1/estimate`` and renders the structured
``EstimationResult``. The endpoint URL is read from ``ESTIMATOR_API_BASE_URL``
(loaded from the same ``.env`` as the API), so the same UI works against a
local uvicorn or against docker-compose.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

# Human-readable labels for the guardrail rejection reasons the API returns.
_GUARDRAIL_REASONS = {
    "moderation": "Content was flagged by moderation.",
    "prompt_injection": "The text looks like a prompt-injection attempt.",
    "pii": "The description contains personal data (email, phone or IBAN).",
}

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Describe the project and get a structured, validated estimation.")


with st.form("estimation_form", clear_on_submit=False):
    description = st.text_area(
        "Project description",
        height=200,
        placeholder="Describe the project: goals, key features, constraints…",
        help="Between 20 and 2000 characters.",
    )
    project_type = st.selectbox(
        "Project type",
        options=[t.value for t in ProjectType],
        index=1,
    )
    detail_level = st.radio(
        "Detail level",
        options=[d.value for d in DetailLevel],
        index=1,
        horizontal=True,
    )
    output_format = st.selectbox(
        "Output format",
        options=[f.value for f in OutputFormat],
        index=0,
    )
    submitted = st.form_submit_button("Generate estimation", type="primary")


if submitted:
    if len(description.strip()) < 20:
        st.error("The description must be at least 20 characters long.")
    else:
        payload = {
            "description": description.strip(),
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }
        with st.spinner("Calling the estimator service…"):
            try:
                response = httpx.post(
                    ESTIMATE_ENDPOINT,
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as exc:
                # Guardrail violations come back as 400 with {detail: {reason, message}}.
                detail = None
                try:
                    detail = exc.response.json().get("detail")
                except Exception:  # noqa: BLE001 — non-JSON error body
                    pass
                if isinstance(detail, dict):
                    reason = detail.get("reason", "")
                    hint = _GUARDRAIL_REASONS.get(reason, "")
                    st.error(f"Rejected ({reason}): {detail.get('message', '')}\n\n{hint}")
                else:
                    st.error(f"Service returned {exc.response.status_code}: {exc.response.text}")
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
            else:
                result = body["result"]

                if body.get("cached"):
                    st.success("⚡ Served from cache (no LLM call).")

                st.subheader("Summary")
                st.write(result["summary"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Total weeks", result["total_duration_weeks"])
                col2.metric("Total cost (EUR)", f"€{result['total_cost_eur']:,}")
                col3.metric("Confidence", f"{result['confidence_pct']}%")

                st.subheader("Phases")
                st.table(result["phases"])

                st.caption(f"Prompt version: {body['prompt_version']}")


with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{os.getenv('PRIMARY_MODEL', 'gpt-4o-mini')}`")
    st.markdown(f"**Fallback model:** `{os.getenv('FALLBACK_MODEL', 'claude-haiku-4-5-20251001')}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")
    st.markdown(f"**Semantic cache:** `{os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.85')}` threshold")
