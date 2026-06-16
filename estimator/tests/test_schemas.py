import pytest

from app.schemas.estimation import EstimationResult, Phase


def test_estimation_result_accepts_matching_totals() -> None:
    result = EstimationResult(
        summary="Valid estimate",
        total_duration_weeks=6,
        total_cost_eur=12000,
        confidence_pct=80,
        phases=[
            Phase(
                name="Design", duration_weeks=2, cost_eur=4000, confidence_pct=90, assumptions=[]
            ),
            Phase(name="Build", duration_weeks=4, cost_eur=8000, confidence_pct=70, assumptions=[]),
        ],
    )
    assert result.total_duration_weeks == 6
    assert result.total_cost_eur == 12000


def test_estimation_result_rejects_mismatched_weeks() -> None:
    with pytest.raises(ValueError, match="total_duration_weeks"):
        EstimationResult(
            summary="Invalid weeks",
            total_duration_weeks=10,
            total_cost_eur=12000,
            confidence_pct=80,
            phases=[
                Phase(
                    name="Design",
                    duration_weeks=2,
                    cost_eur=4000,
                    confidence_pct=90,
                    assumptions=[],
                ),
                Phase(
                    name="Build", duration_weeks=4, cost_eur=8000, confidence_pct=70, assumptions=[]
                ),
            ],
        )


def test_estimation_result_rejects_mismatched_cost() -> None:
    with pytest.raises(ValueError, match="total_cost_eur"):
        EstimationResult(
            summary="Invalid cost",
            total_duration_weeks=6,
            total_cost_eur=10000,
            confidence_pct=80,
            phases=[
                Phase(
                    name="Design",
                    duration_weeks=2,
                    cost_eur=4000,
                    confidence_pct=90,
                    assumptions=[],
                ),
                Phase(
                    name="Build", duration_weeks=4, cost_eur=8000, confidence_pct=70, assumptions=[]
                ),
            ],
        )
