"""Capacity-sweep aggregation and rendering (no allocation, no training)."""

from app.capacity_sweep import RATIOS, VARIANTS, aggregate, render


def test_aggregation_reports_the_crossover_direction():
    """Synthetic rows where RL is worse at slack capacity and better at tight capacity."""
    rows = []
    for ratio in RATIOS:
        for label, _algorithm, _options in VARIANTS:
            objective = 1000.0 + (300.0 * (ratio - 1.0) if label == "rl[+polish]" else 0.0)
            rows.append(
                {
                    "ratio": ratio,
                    "seed": 11,
                    "variant": label,
                    "algorithm": "x",
                    "objective": objective,
                }
            )

    table = aggregate(rows)
    assert len(table) == len(RATIOS)
    leads = {entry["ratio"]: entry["rl_lead_pct"] for entry in table}
    assert leads[1.50] < 0.0, "RL should be worse at 1.5x capacity in this fixture"
    assert leads[0.85] > 0.0, "RL should be better at 0.85x capacity in this fixture"
    # the lead must grow monotonically as capacity tightens
    ordered = [leads[ratio] for ratio in sorted(RATIOS, reverse=True)]
    assert ordered == sorted(ordered)


def test_render_mentions_the_crossover():
    rows = [
        {
            "ratio": ratio,
            "greedy[priority]": 1200.0,
            "hungarian[+lateness]": 1500.0,
            "regret[+polish]": 1000.0,
            "rl[+polish]": 500.0,
            "rl_lead_pct": 50.0,
        }
        for ratio in RATIOS
    ]
    text = render(rows, 1.0)
    assert "# Capacity-pressure sweep" in text
    assert "overtakes" in text
    assert "capacity/demand" in text
    assert "0.85x" in text
