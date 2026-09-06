"""Starting point for the deterministic ``calculate_estimate`` tool."""

from __future__ import annotations

import statistics
from typing import Any

CONTINGENCY_FACTOR = 0.15


def calculate_estimate(args: dict[str, Any]) -> dict[str, Any]:
    """Cost each component from its historical reference amounts, then total."""
    components = args["components"]
    breakdown: list[dict[str, Any]] = []
    total = 0.0

    for component in components:
        name = component["name"]
        refs = component.get("reference_amounts", [])

        if refs:
            central = statistics.median(refs)
            hours = round(central * (1 + CONTINGENCY_FACTOR), 1)
            unbudgeted = False
        else:
            hours = 0.0
            unbudgeted = True

        total += hours
        breakdown.append(
            {
                "name": name,
                "reference_count": len(refs),
                "estimated_hours": hours,
                "unbudgeted": unbudgeted,
            }
        )

    total = round(total, 1)
    return {
        "components": breakdown,
        "total_hours": total,
        "summary": f"total={total}h across {len(breakdown)} components",
    }
