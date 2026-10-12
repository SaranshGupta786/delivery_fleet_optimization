"""Capacity-pressure sweep: how the strategy ranking moves as the fleet tightens.

The problem statement the engine is built around - "fleet capacity is the scarce
resource" - is only interesting if the ranking *changes* with capacity pressure. This
sweep fixes the book size (the ``medium`` preset) and varies the fleet's
capacity/demand ratio, recording each strategy's mean objective across seeds. The
resulting table shows the learned policy's lead growing as the fleet becomes
over-subscribed.

    python -m app.capacity_sweep --out CAPACITY_SWEEP.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

from .algorithms import get_allocator
from .metrics import DEFAULT_OBJECTIVE
from .scenario import generate_scenario

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "CAPACITY_SWEEP.md"
)

PRESET = "medium"
RATIOS = (1.50, 1.35, 1.20, 1.10, 1.00, 0.90, 0.85)
SEEDS = (11, 23, 37)

#: label, algorithm, options
VARIANTS = (
    ("greedy[priority]", "greedy", {"ordering": "priority"}),
    ("hungarian[+lateness]", "hungarian", {"lateness_weight": 2.0}),
    ("regret[+polish]", "regret", {"local_search": True}),
    ("rl[+polish]", "rl", {"local_search": True}),
)


def run_sweep(progress=None) -> List[dict]:
    """One objective per (ratio, seed, variant)."""
    rows: List[dict] = []
    for ratio in RATIOS:
        for seed in SEEDS:
            scenario = generate_scenario(PRESET, seed=seed, capacity_ratio=ratio)
            for label, algorithm, options in VARIANTS:
                plan = get_allocator(algorithm).allocate(scenario, DEFAULT_OBJECTIVE, **options)
                rows.append(
                    {
                        "ratio": ratio,
                        "seed": seed,
                        "variant": label,
                        "algorithm": algorithm,
                        "objective": plan.metrics["objective"],
                        "coverage_pct": plan.metrics["coverage_pct"],
                        "unassigned": plan.metrics["requests_unassigned"],
                        "runtime_ms": plan.runtime_ms,
                    }
                )
                if progress is not None:
                    progress(ratio, seed, label, plan.metrics["objective"])
    return rows


def aggregate(rows: Sequence[dict]) -> List[dict]:
    """Per (ratio, variant) mean objective and the learned policy's lead."""
    grouped: Dict[tuple, List[dict]] = {}
    for row in rows:
        grouped.setdefault((row["ratio"], row["variant"]), []).append(row)

    table: List[dict] = []
    for ratio in RATIOS:
        entry = {"ratio": ratio}
        for label, _algorithm, _options in VARIANTS:
            values = [row["objective"] for row in grouped.get((ratio, label), [])]
            if values:
                entry[label] = sum(values) / len(values)
        regret = entry.get("regret[+polish]")
        learned = entry.get("rl[+polish]")
        # positive = RL is better; negative = regret is better
        entry["rl_lead_pct"] = (regret - learned) / regret * 100.0 if regret and learned else 0.0
        table.append(entry)
    return table


def render(rows: Sequence[dict], elapsed: float) -> str:
    parts: List[str] = []
    parts.append("# Capacity-pressure sweep\n")
    parts.append(
        "Book size fixed at the `medium` preset (14 vehicles, 140 requests); only the "
        "fleet's capacity/demand ratio changes. Each cell is the mean objective over "
        "seeds %s (lower is better).\n" % ", ".join(str(seed) for seed in SEEDS)
    )
    headers = ["capacity/demand", "greedy", "hungarian", "regret", "rl", "rl vs regret"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for entry in rows:
        learned_better = entry["rl_lead_pct"] > 0.0
        lines.append(
            "| %.2fx | %.0f | %.0f | %.0f | %.0f | %s%.1f%%%s |"
            % (
                entry["ratio"],
                entry["greedy[priority]"],
                entry["hungarian[+lateness]"],
                entry["regret[+polish]"],
                entry["rl[+polish]"],
                "**+" if learned_better else "",
                entry["rl_lead_pct"],
                "**" if learned_better else "",
            )
        )
    parts.append("\n".join(lines) + "\n")

    crossovers = [
        (entry["ratio"], entry["rl_lead_pct"])
        for entry in rows
        if entry["rl_lead_pct"] > 0.0
    ]
    if crossovers:
        first = crossovers[0]
        best = max(crossovers, key=lambda item: item[1])
        parts.append(
            "The learned policy first overtakes the hand-tuned regret rule at **%.2fx** "
            "capacity and leads by **%.1f%%** at the tightest point (%.2fx).\n"
            % (first[0], best[1], best[0])
        )
    else:  # pragma: no cover
        parts.append("The learned policy does not overtake regret within this sweep.\n")

    parts.append(
        "\nGenerated in %.1f s by `make capacity` (`python -m app.capacity_sweep`).\n" % elapsed
    )
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sweep capacity pressure and write the table.")
    parser.add_argument("--out", default=os.environ.get("BENCH_OUT", DEFAULT_OUTPUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def progress(ratio, seed, label, objective):
        if not args.quiet:
            print(
                "  ratio %.2f  seed %-3d %-20s objective %10.1f" % (ratio, seed, label, objective),
                flush=True,
            )

    started = time.perf_counter()
    rows = run_sweep(None if args.quiet else progress)
    table = aggregate(rows)
    elapsed = time.perf_counter() - started

    target = os.path.abspath(args.out)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(render(table, elapsed))
    json_target = os.path.splitext(target)[0] + ".json"
    with open(json_target, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "preset": PRESET,
                "ratios": list(RATIOS),
                "seeds": list(SEEDS),
                "table": table,
                "rows": rows,
                "elapsed_seconds": round(elapsed, 1),
            },
            handle,
            indent=1,
            sort_keys=True,
        )
    print("wrote %s" % target)
    print("wrote %s" % json_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
