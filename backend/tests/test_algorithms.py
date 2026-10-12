"""Strategy behaviour: feasibility, determinism, explanations, options."""

from __future__ import annotations

import pytest

from app.algorithms.bruteforce import BruteForceAllocator, BruteForceTooLarge
from app.algorithms.greedy import GreedyAllocator
from app.algorithms.heuristic import RegretAllocator
from app.algorithms.hungarian import HungarianAllocator
from app.metrics import ObjectiveConfig
from tests.conftest import assert_plan_is_consistent

STRATEGIES = (GreedyAllocator(), HungarianAllocator(), RegretAllocator())
IDS = [allocator.name for allocator in STRATEGIES]


def priced_cost(plan):
    """Distance cost plus priced lateness - what the relocate pass minimises."""
    return (
        plan.metrics["objective_components"]["distance_cost"]
        + 2.0 * plan.metrics["lateness_minutes"]
    )


@pytest.mark.parametrize("allocator", STRATEGIES, ids=IDS)
def test_plans_are_feasible_and_fully_accounted(allocator, small_scenario):
    plan = allocator.allocate(small_scenario)
    assert_plan_is_consistent(plan, small_scenario)
    assert plan.algorithm == allocator.name
    assert plan.scenario_name == small_scenario.name
    assert plan.runtime_ms > 0
    assert plan.notes
    assert plan.metrics["objective"] > 0


@pytest.mark.parametrize("allocator", STRATEGIES, ids=IDS)
def test_strategies_are_deterministic(allocator, small_scenario):
    first = allocator.allocate(small_scenario)
    second = allocator.allocate(small_scenario)
    assert first.routes == second.routes
    assert first.metrics["objective"] == second.metrics["objective"]
    assert [a.reason for a in first.assignments] == [a.reason for a in second.assignments]


@pytest.mark.parametrize("allocator", STRATEGIES, ids=IDS)
def test_every_assignment_is_explained_and_ordered(allocator, small_scenario):
    plan = allocator.allocate(small_scenario)
    for vehicle_id, request_ids in plan.routes.items():
        ordered = [a.request_id for a in plan.assignments if a.vehicle_id == vehicle_id]
        assert ordered == list(request_ids), "route order must match the assignment sequence"
        sequences = [a.sequence for a in plan.assignments if a.vehicle_id == vehicle_id]
        assert sequences == list(range(len(sequences)))
    for assignment in plan.assignments:
        assert assignment.reason.startswith(
            "%s serves %s" % (assignment.vehicle_id, assignment.request_id)
        )
        assert 0.0 < assignment.load_pct
        assert assignment.to_dict()["arrival_clock"] != "--:--"
    for unassigned in plan.unassigned:
        assert 0 < unassigned.priority <= 5
        assert unassigned.reason


@pytest.mark.parametrize("allocator", STRATEGIES, ids=IDS)
def test_plans_stay_feasible_on_a_constrained_book(allocator, tight_scenario):
    plan = allocator.allocate(tight_scenario)
    assert_plan_is_consistent(plan, tight_scenario)


@pytest.mark.parametrize("allocator", STRATEGIES, ids=IDS)
def test_plans_handle_an_empty_book(allocator):
    from tests.factories import make_scenario, make_vehicle

    empty = make_scenario([make_vehicle("V1")], [])
    plan = allocator.allocate(empty)
    assert plan.assignments == ()
    assert plan.unassigned == ()
    assert plan.metrics["requests_total"] == 0
    assert plan.metrics["objective"] == 0.0


def test_greedy_orderings_are_honoured(small_scenario):
    for ordering in ("deadline", "priority", "demand"):
        plan = GreedyAllocator().allocate(small_scenario, ordering=ordering)
        assert any(ordering in note for note in plan.notes)
    with pytest.raises(ValueError):
        GreedyAllocator().allocate(small_scenario, ordering="bogus")


def test_greedy_ordering_changes_the_plan(small_scenario):
    by_deadline = GreedyAllocator().allocate(small_scenario, ordering="deadline")
    by_priority = GreedyAllocator().allocate(small_scenario, ordering="priority")
    assert by_deadline.routes != by_priority.routes or (
        by_deadline.metrics["objective"] != by_priority.metrics["objective"]
    )


def test_local_search_never_makes_the_priced_cost_worse(small_scenario):
    with_search = RegretAllocator().allocate(small_scenario, local_search=True)
    without_search = RegretAllocator().allocate(small_scenario, local_search=False)
    assert priced_cost(with_search) <= priced_cost(without_search) + 1e-6
    assert any("local search: on" in note for note in with_search.notes)
    assert any("local search: off" in note for note in without_search.notes)


def test_regret_margins_are_reported(small_scenario):
    plan = RegretAllocator().allocate(small_scenario)
    assert any("Regret rule" in assignment.reason for assignment in plan.assignments)


def test_hungarian_respects_max_rounds(small_scenario):
    single_round = HungarianAllocator().allocate(small_scenario, max_rounds=1)
    assert single_round.metrics["valid"]
    assert any("1 batch rounds" in note for note in single_round.notes)
    full = HungarianAllocator().allocate(small_scenario)
    assert full.metrics["coverage_pct"] >= single_round.metrics["coverage_pct"]


def test_objective_overrides_change_the_reported_objective(small_scenario):
    free_lateness = RegretAllocator().allocate(
        small_scenario, ObjectiveConfig(lateness_cost_per_min=0.0)
    )
    priced_lateness = RegretAllocator().allocate(
        small_scenario, ObjectiveConfig(lateness_cost_per_min=10.0)
    )
    assert free_lateness.metrics["objective_config"]["lateness_cost_per_min"] == 0.0
    if priced_lateness.metrics["lateness_minutes"] > 0:
        assert priced_lateness.metrics["objective"] > free_lateness.metrics["objective"]


def test_exact_strategy_guards_its_instance_size(small_scenario):
    with pytest.raises(BruteForceTooLarge):
        BruteForceAllocator().allocate(small_scenario)


def test_regret_is_at_least_as_good_as_the_naive_baselines(small_scenario):
    """On the default book the hand-tuned heuristic should lead."""
    scores = {
        allocator.name: allocator.allocate(small_scenario).metrics["objective"]
        for allocator in STRATEGIES
    }
    assert scores["regret"] <= scores["greedy"]
    assert scores["regret"] <= scores["hungarian"]


def test_plan_serialisation_is_json_ready(small_scenario):
    import json

    payload = RegretAllocator().allocate(small_scenario).to_dict()
    encoded = json.dumps(payload)  # must not raise
    assert len(encoded) > 500
    assert payload["metrics"]["valid"] is True
    assert payload["routes"]
    assert payload["assignments"][0]["reason"]
    assert payload["assignments"][0]["arrival_clock"]
