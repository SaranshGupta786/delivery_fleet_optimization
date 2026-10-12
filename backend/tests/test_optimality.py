"""Optimality tests: the exhaustive solver is the ground truth.

Two claims are checked here:

1. the exhaustive solver's plan is never worse than any heuristic's plan on the
   same composite objective;
2. when capacity forces exactly one request per vehicle, the Hungarian strategy
   reproduces the optimal assignment (computed here by brute force over
   permutations), which is the strongest statement that can be made about it -
   the overall plan is optimal per round, not globally, and the README says so.
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.algorithms.bruteforce import BruteForceAllocator, BruteForceTooLarge, solve
from app.algorithms.greedy import GreedyAllocator
from app.algorithms.heuristic import RegretAllocator
from app.algorithms.hungarian import HungarianAllocator
from app.domain import Point
from app.geo import haversine_km
from app.metrics import DEFAULT_OBJECTIVE
from tests.conftest import assert_plan_is_consistent
from tests.factories import make_request, make_scenario, make_vehicle

HEURISTICS = (GreedyAllocator(), HungarianAllocator(), RegretAllocator())


def tiny_instance(rng: random.Random, requests: int = 5, vehicles: int = 2):
    """A small instance whose optimum can be enumerated exhaustively."""
    fleet = [
        make_vehicle(
            "V%d" % index,
            depot=Point(12.93 + rng.random() * 0.04, 77.56 + rng.random() * 0.04),
            capacity_kg=rng.choice([40.0, 80.0, 150.0]),
            capacity_m3=4.0,
            max_route_minutes=420.0,
            speed_kmh=rng.choice([20.0, 30.0]),
            cost_per_km=rng.choice([0.8, 1.0, 1.4]),
            capabilities=frozenset(),
        )
        for index in range(vehicles)
    ]
    demand = [
        make_request(
            "R%d" % index,
            location=Point(12.93 + rng.random() * 0.04, 77.56 + rng.random() * 0.04),
            demand_kg=rng.choice([5.0, 15.0, 30.0]),
            volume_m3=0.3,
            service_minutes=5.0,
            window_open_min=540.0,
            window_close_min=540.0 + rng.choice([60.0, 120.0, 240.0]),
            priority=rng.randint(1, 5),
            required_capabilities=frozenset(),
        )
        for index in range(requests)
    ]
    return make_scenario(fleet, demand, grace_minutes=30.0)


def test_exact_reference_is_never_beaten_by_a_heuristic():
    rng = random.Random(1234)
    for _ in range(8):
        scenario = tiny_instance(rng, requests=5, vehicles=2)
        optimum = solve(scenario).objective
        for allocator in HEURISTICS:
            plan = allocator.allocate(scenario)
            # 0.01 of slack: the reported objective is rounded to 3 decimals, so a
            # heuristic can appear up to ~0.0005 below the exact solver's raw value.
            assert optimum <= plan.metrics["objective"] + 0.01, (
                "%s beat the exhaustive optimum on %s (%.6f vs %.6f)"
                % (allocator.name, scenario.name, plan.metrics["objective"], optimum)
            )


def test_exact_plan_refuses_a_request_nobody_can_serve():
    vehicle = make_vehicle("V1")
    impossible = make_request(
        "IMPOSSIBLE",
        location=Point(30.0, 100.0),
        window_open_min=480.0,
        window_close_min=485.0,
    )
    scenario = make_scenario([vehicle], [impossible])
    solution = solve(scenario)
    assert solution.unassigned_ids == ("IMPOSSIBLE",)
    assert solution.objective == pytest.approx(DEFAULT_OBJECTIVE.unassigned_cost(impossible))
    assert solution.evaluated > 0


def test_hungarian_reproduces_the_optimal_assignment():
    """Capacity forces one request per vehicle, so the batch optimum is the optimum."""
    costs = (1.0, 1.5, 0.8)
    fleet = [
        make_vehicle(
            "V%d" % index,
            depot=Point(12.95 + 0.01 * index, 77.60),
            capacity_kg=100.0,
            capacity_m3=10.0,
            speed_kmh=25.0,
            cost_per_km=tariff,
            capabilities=frozenset(),
        )
        for index, tariff in enumerate(costs)
    ]
    demand = [
        make_request(
            "R%d" % index,
            location=Point(12.96 + 0.005 * index, 77.61 + 0.005 * index),
            demand_kg=100.0,
            volume_m3=0.5,
            required_capabilities=frozenset(),
        )
        for index in range(3)
    ]
    scenario = make_scenario(fleet, demand)

    plan = HungarianAllocator().allocate(scenario)
    assert plan.metrics["coverage_pct"] == 100.0
    assert plan.metrics["lateness_minutes"] == 0.0

    best_matching = min(
        sum(
            haversine_km(fleet[vehicle_index].depot, demand[request_index].location)
            * fleet[vehicle_index].cost_per_km
            for vehicle_index, request_index in enumerate(order)
        )
        for order in itertools.permutations(range(len(demand)))
    )
    assert plan.metrics["objective_components"]["distance_cost"] == pytest.approx(
        best_matching, abs=1e-3
    )


def test_exact_strategy_reports_its_own_search_size():
    rng = random.Random(5)
    scenario = tiny_instance(rng, requests=4, vehicles=2)
    plan = BruteForceAllocator().allocate(scenario)
    assert_plan_is_consistent(plan, scenario)
    assert any("candidate plans evaluated" in note for note in plan.notes)


def test_exact_strategy_refuses_instances_beyond_its_guard(small_scenario):
    with pytest.raises(BruteForceTooLarge):
        BruteForceAllocator().allocate(small_scenario)
    with pytest.raises(ValueError):
        solve(small_scenario, max_requests=3, max_vehicles=1)


@pytest.mark.slow
def test_exact_handles_the_largest_allowed_instance():
    rng = random.Random(7)
    scenario = tiny_instance(rng, requests=6, vehicles=3)
    plan = BruteForceAllocator().allocate(scenario)
    assert_plan_is_consistent(plan, scenario)
