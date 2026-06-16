"""Streamlit form UI for the estimator.

The UI collects an EstimationRequest through a form, POSTs it to the FastAPI
service, and renders the structured EstimationResult.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)

load_dotenv()

API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Describe the project and get a structured estimation.")


with st.form("estimation_form"):
    description = st.text_area(
        "Project description",
        max_chars=2000,
        height=150,
    )
    project_type = st.selectbox(
        "Project type",
        options=[e.value for e in ProjectType],
    )
    detail_level = st.selectbox(
        "Detail level",
        options=[e.value for e in DetailLevel],
    )
    output_format = st.selectbox(
        "Output format",
        options=[e.value for e in OutputFormat],
    )
    submitted = st.form_submit_button("Estimate")

if submitted:
    if not description or len(description) < 20:
        st.error("Please provide a description of at least 20 characters.")
    else:
        payload = EstimationRequest(
            description=description,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        ).model_dump()

        with st.spinner("Estimating..."):
            try:
                response = httpx.post(
                    ESTIMATE_ENDPOINT,
                    json=payload,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
            else:
                data = response.json()
                result = data["result"]

                st.subheader("Summary")
                st.write(result["summary"])

                col1, col2, col3 = st.columns(3)
                col1.metric("Total weeks", result["total_duration_weeks"])
                col2.metric("Total cost (EUR)", f"€{result['total_cost_eur']:,}")
                col3.metric("Confidence", f"{result['confidence_pct']}%")

                st.subheader("Phases")
                st.table(result["phases"])

                st.caption(f"Prompt version: {data['prompt_version']}")
