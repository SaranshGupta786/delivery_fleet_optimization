"""Shared fixtures and helpers for the engine's test suite."""

from __future__ import annotations

import pytest

from app.algorithms import algorithm_names, get_allocator
from app.feasibility import evaluate_route
from app.scenario import generate_scenario


@pytest.fixture(scope="session")
def small_scenario():
    """A 6-vehicle / 40-request book: fast enough for property-style checks."""
    return generate_scenario("small", seed=42)


@pytest.fixture(scope="session")
def medium_scenario():
    return generate_scenario("medium", seed=42)


@pytest.fixture(scope="session")
def tight_scenario():
    return generate_scenario("tight", seed=7)


@pytest.fixture(scope="session")
def available_algorithms():
    return algorithm_names()


@pytest.fixture(scope="session")
def classical_allocators():
    """Classical strategies only - always available, artifact or not."""
    return [get_allocator(name) for name in ("greedy", "hungarian", "regret")]


def assert_plan_is_consistent(plan, scenario):
    """Every plan must account for every request exactly once and stay feasible."""
    metrics = plan.metrics
    assert metrics["valid"], metrics["hard_violations"]
    assert metrics["requests_assigned"] + metrics["requests_unassigned"] == metrics["requests_total"]
    assert metrics["requests_total"] == len(scenario.requests)
    assert len(plan.assignments) == metrics["requests_assigned"]
    assert len(plan.unassigned) == metrics["requests_unassigned"]

    assigned = [assignment.request_id for assignment in plan.assignments]
    assert len(assigned) == len(set(assigned)), "a request was served twice"
    for request in scenario.requests:
        assert request.id in assigned or any(
            unassigned.request_id == request.id for unassigned in plan.unassigned
        )

    for vehicle_id, request_ids in plan.routes.items():
        vehicle = scenario.vehicle(vehicle_id)
        evaluation = evaluate_route(
            vehicle, [scenario.request(request_id) for request_id in request_ids], scenario.grace_minutes
        )
        assert evaluation.feasible, evaluation.violations

    for assignment in plan.assignments:
        assert assignment.reason, "every assignment needs an explanation"
    for unassigned in plan.unassigned:
        assert unassigned.reason, "every refusal needs an explanation"
