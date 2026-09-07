"""Logfire configuration shared by FastAPI and command-line graph runs."""

from __future__ import annotations

import logfire
from fastapi import FastAPI

from app.config import get_settings

_core_instrumented = False
_instrumented_apps: set[int] = set()


def configure_observability(app: FastAPI | None = None) -> None:
    """Configure local traces and export them when a Logfire token is available."""
    global _core_instrumented

    if not _core_instrumented:
        settings = get_settings()
        logfire.configure(
            send_to_logfire="if-token-present",
            service_name="estimator",
            environment=settings.APP_ENV,
        )
        logfire.instrument_httpx()
        logfire.instrument_asyncpg()
        logfire.instrument_psycopg()
        _core_instrumented = True

    if app is not None and id(app) not in _instrumented_apps:
        logfire.instrument_fastapi(app)
        _instrumented_apps.add(id(app))
