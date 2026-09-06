"""Fallback retrieval data for debugging the Session 12 agent loop."""

from __future__ import annotations

from typing import Any

_CORPUS: list[dict[str, Any]] = [
    {
        "keywords": ("oauth", "auth", "autenticación", "jwt", "login", "token", "sso"),
        "id": 1001,
        "content_preview": "Backend de autenticación OAuth2 con JWT y multi-tenant.",
        "sector": "finance",
        "budget_id": "BUD-AUTH-2023-07",
        "estimated_hours": 420.0,
    },
    {
        "keywords": ("backend", "api", "pedidos", "rutas", "tarifas", "núcleo"),
        "id": 2001,
        "content_preview": "Backend de pedidos, rutas y tarifas con API REST.",
        "sector": "logistics",
        "budget_id": "BUD-CORE-2023-02",
        "estimated_hours": 1150.0,
    },
    {
        "keywords": ("erp", "sap", "integración", "idoc", "middleware", "facturación"),
        "id": 3001,
        "content_preview": "Integración SAP vía IDocs con sincronización y reintentos.",
        "sector": "industrial",
        "budget_id": "BUD-SAP-2023-05",
        "estimated_hours": 860.0,
    },
    {
        "keywords": ("móvil", "mobile", "app", "android", "ios", "offline", "repartidor"),
        "id": 4001,
        "content_preview": "App de reparto Android/iOS con firma, foto y modo offline.",
        "sector": "logistics",
        "budget_id": "BUD-APP-2023-03",
        "estimated_hours": 780.0,
    },
    {
        "keywords": ("analítica", "analytics", "panel", "dashboard", "kpi", "alertas"),
        "id": 5001,
        "content_preview": "Panel de KPIs logísticos con filtros y alertas.",
        "sector": "logistics",
        "budget_id": "BUD-BI-2023-01",
        "estimated_hours": 560.0,
    },
]


def search_budgets_stub(query: str, filters: dict | None = None) -> list[dict[str, Any]]:
    """Return canned historical items whose keywords appear in ``query``."""
    normalized_query = query.lower()
    sectors = None
    if filters and filters.get("sectors"):
        sectors = {sector.lower() for sector in filters["sectors"]}

    hits: list[dict[str, Any]] = []
    for entry in _CORPUS:
        matches = sum(keyword in normalized_query for keyword in entry["keywords"])
        if matches == 0 or (sectors and entry["sector"].lower() not in sectors):
            continue
        hits.append(
            {
                "id": entry["id"],
                "content_preview": entry["content_preview"],
                "sector": entry["sector"],
                "budget_id": entry["budget_id"],
                "estimated_hours": entry["estimated_hours"],
                "distance": round(max(0.05, 0.6 - 0.1 * matches), 4),
            }
        )

    return sorted(hits, key=lambda hit: hit["distance"])[:5]
