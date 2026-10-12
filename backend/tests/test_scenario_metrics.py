"""Scenario generator and metrics tests."""

from __future__ import annotations

import pytest

from app.domain import Point
from app.geo import haversine_km
from app.metrics import (
    DEFAULT_OBJECTIVE,
    ObjectiveConfig,
    compute_metrics,
    objective_value,
    plan_evaluations,
)
from app.scenario import (
    CITY_BOXES,
    PRESETS,
    generate_scenario,
    preset_info,
    preset_names,
    scenario_from_spec,
)
from tests.factories import make_request, make_scenario, make_vehicle


def test_generation_is_deterministic():
    first = generate_scenario("small", seed=42)
    second = generate_scenario("small", seed=42)
    assert first.to_dict() == second.to_dict()
    assert generate_scenario("small", seed=43).to_dict() != first.to_dict()


@pytest.mark.parametrize("preset", preset_names())
def test_presets_match_their_advertised_size(preset):
    scenario = generate_scenario(preset, seed=5)
    assert len(scenario.vehicles) == PRESETS[preset]["vehicles"]
    assert len(scenario.requests) == PRESETS[preset]["requests"]
    assert scenario.metadata["preset"] == preset
    assert scenario.name == "%s-seed5" % preset


def test_preset_info_is_ui_ready():
    info = preset_info()
    assert [entry["name"] for entry in info] == preset_names()
    assert all("description" in entry and "capacity_ratio" in entry for entry in info)


def test_capacity_ratio_is_a_real_knob():
    default = generate_scenario("medium", seed=9)
    assert default.capacity_ratio == pytest.approx(1.45, abs=0.02)
    tight = generate_scenario("medium", seed=9, capacity_ratio=1.05)
    assert tight.capacity_ratio == pytest.approx(1.05, abs=0.02)
    assert tight.total_capacity_kg < default.total_capacity_kg


def test_generated_books_are_self_consistent():
    scenario = generate_scenario("medium", seed=3)
    fleet_capabilities = frozenset().union(
        *(vehicle.capabilities for vehicle in scenario.vehicles)
    )
    south, west, north, east = CITY_BOXES[scenario.city]
    for vehicle in scenario.vehicles:
        assert vehicle.capacity_kg > 0 and vehicle.capacity_m3 > 0
        assert vehicle.shift_end_min > vehicle.shift_start_min
        assert 0 < vehicle.max_route_minutes <= vehicle.shift_minutes
        assert south <= vehicle.depot.lat <= north
    for request in scenario.requests:
        assert request.demand_kg > 0
        assert request.volume_m3 > 0
        assert request.window_close_min > request.window_open_min
        assert 1 <= request.priority <= 5
        assert south <= request.location.lat <= north
        assert west <= request.location.lon <= east
        # a request may only require capability that some vehicle actually has
        assert request.required_capabilities <= fleet_capabilities


def test_scenario_from_spec_accepts_knobs_and_full_payloads():
    generated = scenario_from_spec({"preset": "small", "seed": 11, "requests": 12, "vehicles": 3})
    assert len(generated.requests) == 12
    assert len(generated.vehicles) == 3

    explicit = make_scenario([make_vehicle()], [make_request()])
    rebuilt = scenario_from_spec(explicit.to_dict())
    assert rebuilt.to_dict() == explicit.to_dict()


@pytest.mark.parametrize(
    "payload",
    (
        {"preset": "nope"},
        {"preset": "small", "vehicles": -1},
        {"preset": "small", "capacity_ratio": 0.0},
        {"preset": "small", "window_scale": -2.0},
    ),
)
def test_scenario_from_spec_rejects_nonsense(payload):
    with pytest.raises(ValueError):
        scenario_from_spec(payload)


def test_unknown_city_falls_back_to_the_default():
    scenario = generate_scenario("small", seed=1, city="atlantis")
    assert scenario.city == "bengaluru"
    assert scenario.metadata["city"] == "bengaluru"


def test_metrics_on_an_empty_plan():
    scenario = generate_scenario("small", seed=42)
    metrics = compute_metrics(scenario, {vehicle.id: [] for vehicle in scenario.vehicles})
    assert metrics["requests_assigned"] == 0
    assert metrics["requests_unassigned"] == len(scenario.requests)
    assert metrics["coverage_pct"] == 0.0
    assert metrics["distance_km"] == 0.0
    assert metrics["valid"] and not metrics["hard_violations"]
    assert metrics["objective"] == pytest.approx(metrics["objective_components"]["unassigned_cost"], abs=0.01)
    assert metrics["on_time_pct"] == 100.0


def test_metrics_reject_double_assignment():
    scenario = make_scenario([make_vehicle("V1"), make_vehicle("V2")], [make_request("R1")])
    with pytest.raises(ValueError):
        plan_evaluations(scenario, {"V1": ["R1"], "V2": ["R1"]})


def test_objective_components_are_arithmetic():
    vehicle = make_vehicle("V1", speed_kmh=20.0)
    request = make_request(
        "LATE", location=Point(12.96, 77.61), window_open_min=480.0, window_close_min=480.0
    )
    scenario = make_scenario([vehicle], [request])
    routes = {"V1": ["LATE"]}

    unpriced = compute_metrics(scenario, routes, ObjectiveConfig(lateness_cost_per_min=0.0))
    priced = compute_metrics(scenario, routes, ObjectiveConfig(lateness_cost_per_min=3.0))
    assert priced["lateness_minutes"] > 0.0
    assert priced["objective"] - unpriced["objective"] == pytest.approx(
        3.0 * priced["lateness_minutes"], abs=0.01
    )

    refused = compute_metrics(
        scenario,
        {"V1": []},
        ObjectiveConfig(unassigned_base_cost=100.0, priority_step=0.5, makespan_cost_per_min=0.0),
    )
    assert refused["objective"] == pytest.approx(100.0 * (1.0 + (request.priority - 1) * 0.5))


def test_priority_weighting_shows_up_in_coverage_and_cost():
    low = make_request("LOW", priority=1)
    high = make_request("HIGH", priority=5)
    scenario = make_scenario([make_vehicle("V1")], [low, high])

    nothing = compute_metrics(scenario, {"V1": []})
    assert nothing["priority_coverage_pct"] == 0.0

    only_vip = compute_metrics(scenario, {"V1": ["HIGH"]})
    assert only_vip["priority_coverage_pct"] == pytest.approx(100.0 * 3.0 / 4.0, abs=0.01)
    assert only_vip["unassigned_ids"] == ["LOW"]

    only_low = compute_metrics(scenario, {"V1": ["LOW"]})
    assert only_vip["objective"] < only_low["objective"]


def test_distance_cost_uses_each_vehicle_own_tariff():
    cheap = make_vehicle("CHEAP", cost_per_km=0.5)
    dear = make_vehicle("DEAR", cost_per_km=2.0)
    request = make_request("R1")
    scenario = make_scenario([cheap, dear], [request])
    by_cheap = objective_value(scenario, {"CHEAP": ["R1"], "DEAR": []})
    by_dear = objective_value(scenario, {"CHEAP": [], "DEAR": ["R1"]})
    legs = haversine_km(cheap.depot, request.location)
    assert by_dear - by_cheap == pytest.approx(legs * (2.0 - 0.5), abs=0.05)


def test_reported_kpis_are_consistent(small_scenario):
    from app.algorithms.greedy import GreedyAllocator

    metrics = GreedyAllocator().allocate(small_scenario).metrics
    per_vehicle = metrics["per_vehicle"]
    assert metrics["distance_km"] == pytest.approx(
        sum(entry["distance_km"] for entry in per_vehicle), abs=0.01
    )
    assert metrics["vehicles_used"] == sum(1 for entry in per_vehicle if entry["stops"] > 0)
    assert metrics["vehicles_total"] == len(per_vehicle)
    assert metrics["on_time_requests"] + metrics["late_requests"] == metrics["requests_assigned"]
    assert 0.0 <= metrics["avg_utilization_pct"] <= 100.0
    assert metrics["objective"] == pytest.approx(
        sum(metrics["objective_components"].values()), abs=0.01
    )
    assert metrics["objective_config"] == DEFAULT_OBJECTIVE.to_dict()
    assert metrics["max_lateness_min"] >= 0.0
    assert metrics["workload_stdev_minutes"] >= 0.0
