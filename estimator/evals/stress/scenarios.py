"""Synthetic multi-turn scenarios for the CAG stress test (Block 2).

Each scenario is a list of ``Turn(turn_index, transcript, fact_to_remember)``.
``fact_to_remember`` is the durable claim introduced at that turn that LATER
turns should still recall — it feeds ``MemoryDriftMetric`` (Block 4), which
looks for the fact in the session snapshot (summary / anchors / metadata).

Three profiles, each designed to force a specific behaviour:

- ``growing``: coherent requirements pile up turn by turn. Does the original
  project_name survive to turn 20? How does cost scale?
- ``pivot``: turn 5 swaps the stack (React -> Flutter). Does metadata update
  cleanly, or does mentioned_technologies accumulate both?
- ``contradiction``: turn 3 says "budget 30k", turn 8 says "budget 80k".
  Which one is preserved, promoted to an anchor, or folded into the summary?

The fact strings are short and match-friendly (case-insensitive substring)
because the metric is deterministic by design — no embeddings, no LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    turn_index: int
    transcript: str
    fact_to_remember: str


@dataclass(frozen=True)
class Scenario:
    name: str
    project_type: str
    detail_level: str
    output_format: str
    turns: list[Turn]


# --- growing: a project that accretes coherent requirements ------------------

_GROWING = Scenario(
    name="growing",
    project_type="web_saas",
    detail_level="medium",
    output_format="phases_table",
    turns=[
        Turn(1, "We are building Nimbus, a B2B SaaS CRM for small agencies. "
                "First we need contact management and a deal pipeline.",
             "Nimbus"),
        Turn(3, "Add user authentication with email/password and Google SSO to Nimbus.",
             "authentication"),
        Turn(6, "Nimbus also needs multi-tenant isolation — each agency is a "
                "separate tenant with its own data.",
             "multi-tenant"),
        Turn(10, "We need an audit log on Nimbus recording every change to deals "
                 "and contacts, retained for 12 months.",
             "audit log"),
        Turn(20, "Finally, Nimbus must support CSV export of contacts and deals "
                  "for the reporting team.",
             "CSV export"),
    ],
)


# --- pivot: the stack changes mid-conversation -------------------------------

_PIVOT = Scenario(
    name="pivot",
    project_type="mobile_app",
    detail_level="medium",
    output_format="phases_table",
    turns=[
        Turn(1, "We want a mobile loyalty app called Orbit, built with React Native, "
                "with a points wallet and push notifications.",
             "React Native"),
        Turn(3, "Orbit needs a rewards catalogue and QR-code redemption in store.",
             "QR-code redemption"),
        Turn(5, "Change of plan: we are dropping React Native and building Orbit "
                "natively with Flutter instead. The rest of the scope stays.",
             "Flutter"),
        Turn(8, "Orbit on Flutter should also support offline mode for the wallet.",
             "offline mode"),
        Turn(12, "Add Apple Wallet and Google Wallet pass integration to Orbit.",
             "Apple Wallet"),
    ],
)


# --- contradiction: a fact is overwritten by a later turn --------------------

_CONTRADICTION = Scenario(
    name="contradiction",
    project_type="internal_tool",
    detail_level="medium",
    output_format="phases_table",
    turns=[
        Turn(1, "We need an internal HR onboarding tool called Atlas for a "
                "150-person company.",
             "Atlas"),
        Turn(3, "The budget for Atlas is locked at 30000 EUR. Keep it lean.",
             "30000 EUR"),
        Turn(5, "Atlas should manage equipment assignment and document signing.",
             "document signing"),
        Turn(8, "Update: the budget for Atlas has been raised and is now "
                "locked at 80000 EUR. We can be more ambitious.",
             "80000 EUR"),
        Turn(12, "With the larger budget, add an analytics dashboard to Atlas.",
             "analytics dashboard"),
    ],
)


SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (_GROWING, _PIVOT, _CONTRADICTION)
}


def get_scenarios(names: list[str] | None = None) -> list[Scenario]:
    """Return the requested scenarios (all of them when ``names`` is None).

    Raises ``KeyError`` for an unknown name so the runner fails loudly rather
    than silently skipping a scenario the user asked for.
    """
    if not names:
        return list(SCENARIOS.values())
    return [SCENARIOS[n] for n in names]


if __name__ == "__main__":  # pragma: no cover - smoke check
    # ponytail: cheapest possible self-check — every fact must be a substring
    # of its own transcript, else MemoryDriftMetric can never pass even at the
    # turn the fact is introduced.
    for scenario in SCENARIOS.values():
        for turn in scenario.turns:
            assert turn.fact_to_remember.lower() in turn.transcript.lower(), (
                f"{scenario.name} turn {turn.turn_index}: fact "
                f"{turn.fact_to_remember!r} not in its own transcript"
            )
    print(f"OK: {len(SCENARIOS)} scenarios, "
          f"{sum(len(s.turns) for s in SCENARIOS.values())} turns")
