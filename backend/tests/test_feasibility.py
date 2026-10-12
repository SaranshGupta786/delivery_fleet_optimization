"""Hard-constraint semantics and the fast-insertion/full-simulation equivalence.

The equivalence test is the important one: ``best_insertion`` reasons about
insertions in O(1) per position (closed-form distance and capacity deltas plus
forward time slack), while ``evaluate_route`` re-simulates a whole route. Any
disagreement would silently corrupt every strategy's decisions, so the test walks
every candidate position of many random instances and demands identical
feasibility, distance, finish time, lateness and waiting.
"""

from __future__ import annotations

import random

import pytest

from app.domain import Point
from app.feasibility import best_insertion, evaluate_route, route_distance_cost
from tests.factories import make_request, make_scenario, make_vehicle


def random_instance(rng: random.Random):
    """A small, deliberately awkward instance (tight windows, mixed capacities)."""
    vehicles = [
        make_vehicle(
            "V%d" % (index + 1),
            depot=Point(12.90 + rng.random() * 0.15, 77.50 + rng.random() * 0.20),
            capacity_kg=rng.choice([150.0, 400.0, 900.0]),
            capacity_m3=rng.choice([1.0, 3.0, 8.0]),
            shift_end_min=480.0 + rng.choice([300.0, 480.0, 600.0]),
            max_route_minutes=rng.choice([180.0, 300.0, 480.0]),
            speed_kmh=rng.choice([18.0, 25.0, 35.0]),
            cost_per_km=rng.choice([0.5, 0.9, 1.2]),
            capabilities=frozenset(["cold_chain"]) if rng.random() < 0.5 else frozenset(),
        )
        for index in range(3)
    ]
    requests = []
    for index in range(8):
        window_open = rng.choice([480.0, 540.0, 600.0, 660.0])
        requests.append(
            make_request(
                "R%02d" % index,
                location=Point(12.90 + rng.random() * 0.15, 77.50 + rng.random() * 0.20),
                demand_kg=rng.choice([5.0, 20.0, 60.0, 120.0]),
                volume_m3=rng.choice([0.1, 0.5, 1.5]),
                service_minutes=rng.choice([5.0, 10.0, 15.0]),
                window_open_min=window_open,
                window_close_min=window_open + rng.choice([60.0, 120.0, 240.0, 360.0]),
                priority=rng.randint(1, 5),
                required_capabilities=(
                    frozenset() if rng.random() < 0.7 else frozenset({"cold_chain"})
                ),
            )
        )
    return make_scenario(vehicles, requests, grace_minutes=30.0)


def test_empty_route_is_feasible_and_free():
    vehicle = make_vehicle()
    evaluation = evaluate_route(vehicle, [], 30.0)
    assert evaluation.feasible
    assert evaluation.stop_count == 0
    assert evaluation.distance_km == 0.0
    assert evaluation.finish_min == vehicle.shift_start_min
    assert evaluation.busy_minutes == 0.0
    assert route_distance_cost(vehicle, evaluation) == 0.0


def test_capability_constraint_is_hard():
    vehicle = make_vehicle(capabilities=frozenset())
    request = make_request(required_capabilities=frozenset({"cold_chain"}))
    assert not vehicle.supports(request)
    evaluation = evaluate_route(vehicle, [request], 30.0)
    assert not evaluation.feasible
    assert any(violation.startswith("capability:") for violation in evaluation.violations)
    assert best_insertion(vehicle, [], request, 30.0) is None


def test_capacity_constraint_counts_the_whole_route():
    vehicle = make_vehicle(capacity_kg=100.0, capacity_m3=10.0)
    first = make_request("R1", demand_kg=60.0, volume_m3=0.5)
    second = make_request("R2", demand_kg=60.0, volume_m3=0.5)
    assert evaluate_route(vehicle, [first], 30.0).feasible
    evaluation = evaluate_route(vehicle, [first, second], 30.0)
    assert not evaluation.feasible
    assert any(violation.startswith("capacity_kg:") for violation in evaluation.violations)
    # 60 + 60 kg cannot fit together, so the second request must land elsewhere.
    assert best_insertion(vehicle, [first], second, 30.0) is None


def test_time_window_grace_and_hard_limit():
    vehicle = make_vehicle(speed_kmh=20.0)
    near = make_request("NEAR", location=Point(12.951, 77.601))
    # ~0.01 degrees of latitude is about 1.1 km -> ~3.3 minutes at 20 km/h
    within_grace = make_request(
        "GRACE", location=Point(12.96, 77.61), window_open_min=480.0, window_close_min=480.0
    )
    evaluation = evaluate_route(vehicle, [within_grace], 30.0)
    assert evaluation.feasible
    assert evaluation.lateness_minutes > 0.0

    far = make_request(
        "FAR", location=Point(13.20, 77.90), window_open_min=480.0, window_close_min=480.0
    )
    hard = evaluate_route(vehicle, [far], 30.0)
    assert not hard.feasible
    assert any(violation.startswith("window:") for violation in hard.violations)
    assert best_insertion(vehicle, [], far, 30.0) is None
    assert best_insertion(vehicle, [near], far, 30.0) is None


def test_waiting_for_the_window_is_allowed_and_counted():
    vehicle = make_vehicle(speed_kmh=30.0)
    request = make_request(
        "EARLY", location=Point(12.96, 77.61), window_open_min=600.0, window_close_min=720.0
    )
    evaluation = evaluate_route(vehicle, [request], 30.0)
    assert evaluation.feasible
    stop = evaluation.stops[0]
    assert stop.arrival_min < 600.0
    assert stop.service_start_min == pytest.approx(600.0)
    assert stop.waiting_min == pytest.approx(600.0 - stop.arrival_min)
    assert evaluation.busy_minutes == pytest.approx(
        evaluation.travel_minutes + evaluation.waiting_minutes + evaluation.service_minutes
    )


def test_shift_end_and_route_length_are_hard_constraints():
    slow = make_request("SLOW", location=Point(13.30, 78.20), window_close_min=1200.0)

    shift_vehicle = make_vehicle(shift_end_min=500.0, max_route_minutes=1000.0)
    evaluation = evaluate_route(shift_vehicle, [slow], 300.0)
    assert not evaluation.feasible
    assert "shift_end" in evaluation.violations

    short_route_vehicle = make_vehicle(max_route_minutes=5.0, shift_end_min=1200.0)
    evaluation = evaluate_route(short_route_vehicle, [slow], 300.0)
    assert not evaluation.feasible
    assert "route_length" in evaluation.violations


def test_best_insertion_respects_a_custom_ranking_key():
    vehicle = make_vehicle("V1")
    route = [make_request("R1", location=Point(12.96, 77.61))]
    request = make_request("R2", location=Point(12.97, 77.62))
    last = best_insertion(
        vehicle,
        route,
        request,
        30.0,
        key=lambda insertion: (-insertion.position, insertion.cost),
    )
    assert last is not None
    assert last.position == 1


def test_best_insertion_is_equivalent_to_full_simulation():
    rng = random.Random(20261011)
    positions_checked = 0
    feasible_positions = 0

    for _ in range(30):
        scenario = random_instance(rng)
        for vehicle in scenario.vehicles:
            route: list = []
            for request in rng.sample(list(scenario.requests), rng.randint(0, 4)):
                insertion = best_insertion(vehicle, route, request, scenario.grace_minutes)
                if insertion is not None:
                    route.insert(insertion.position, request)
            base = evaluate_route(vehicle, route, scenario.grace_minutes)
            assert base.feasible, base.violations

            for request in scenario.requests:
                if request in route:
                    continue
                for position in range(len(route) + 1):
                    positions_checked += 1
                    full = evaluate_route(
                        vehicle,
                        route[:position] + [request] + route[position:],
                        scenario.grace_minutes,
                    )
                    found = best_insertion(
                        vehicle,
                        route,
                        request,
                        scenario.grace_minutes,
                        base,
                        key=lambda ins, wanted=position: (
                            0 if ins.position == wanted else 1,
                            ins.position,
                        ),
                    )
                    accepted = found is not None and found.position == position
                    assert full.feasible == accepted, (
                        "position %d: simulation says %s, search says %s (%s)"
                        % (position, full.feasible, accepted, full.violations)
                    )
                    if not full.feasible:
                        continue
                    feasible_positions += 1
                    assert found.delta_km == pytest.approx(
                        full.distance_km - base.distance_km, abs=1e-6
                    )
                    assert found.finish_min == pytest.approx(full.finish_min, abs=1e-6)
                    assert found.lateness_delta_min == pytest.approx(
                        full.lateness_minutes - base.lateness_minutes, abs=1e-6
                    )
                    assert found.waiting_delta_min == pytest.approx(
                        full.waiting_minutes - base.waiting_minutes, abs=1e-6
                    )

    assert positions_checked > 600, positions_checked
    assert feasible_positions > 100, feasible_positions
