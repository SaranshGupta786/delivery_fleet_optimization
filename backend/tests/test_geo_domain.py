"""Geometry and domain-model tests."""

from __future__ import annotations

import pytest

from app.domain import (
    DeliveryRequest,
    Point,
    Scenario,
    Vehicle,
    DEFAULT_GRACE_MINUTES,
    DEFAULT_HORIZON_START_MIN,
)
from app.geo import bbox, haversine_km, mercator, route_distance_km, travel_minutes
from tests.factories import make_request, make_vehicle


def test_haversine_matches_a_known_distance():
    # Bengaluru -> Chennai is about 290 km.
    distance = haversine_km(Point(12.9716, 77.5946), Point(13.0827, 80.2707))
    assert 280.0 < distance < 300.0


def test_haversine_is_symmetric_and_zero_for_the_same_point():
    first, second = Point(12.90, 77.50), Point(13.05, 77.70)
    assert haversine_km(first, second) == pytest.approx(haversine_km(second, first))
    assert haversine_km(first, first) == 0.0


def test_travel_minutes_scales_with_speed():
    first, second = Point(12.90, 77.50), Point(12.95, 77.55)
    distance = haversine_km(first, second)
    assert travel_minutes(first, second, 30.0) == pytest.approx(distance / 30.0 * 60.0)
    assert travel_minutes(first, second, 60.0) == pytest.approx(
        travel_minutes(first, second, 30.0) / 2.0
    )
    with pytest.raises(ValueError):
        travel_minutes(first, second, 0.0)


def test_bbox_and_route_distance():
    points = [Point(1.0, 2.0), Point(3.0, -1.0)]
    assert bbox(points) == (1.0, -1.0, 3.0, 2.0)
    assert bbox([]) == (0.0, 0.0, 0.0, 0.0)
    legs = [Point(0.0, 0.0), Point(0.0, 1.0), Point(0.0, 2.0)]
    assert route_distance_km(legs) == pytest.approx(2 * haversine_km(legs[0], legs[1]))


def test_mercator_projection_is_normalised():
    assert mercator(0.0, 0.0) == pytest.approx((0.5, 0.5))
    for lat, lon in ((12.9, 77.6), (-33.9, 151.2), (52.5, 13.4)):
        x, y = mercator(lat, lon)
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
    assert mercator(10.0, 0.0)[1] < mercator(0.0, 0.0)[1]


def test_vehicle_round_trip_and_capability_support():
    vehicle = make_vehicle()
    assert Vehicle.from_dict(vehicle.to_dict()) == vehicle
    assert vehicle.supports(make_request())
    assert vehicle.supports(make_request(required_capabilities=frozenset({"cold_chain"})))
    assert not vehicle.supports(
        make_request(required_capabilities=frozenset({"cold_chain", "lift_gate"}))
    )
    assert vehicle.shift_minutes == pytest.approx(540.0)


def test_request_round_trip_and_helpers():
    request = make_request(required_capabilities=frozenset({"fragile"}))
    assert DeliveryRequest.from_dict(request.to_dict()) == request
    assert request.window_minutes == pytest.approx(180.0)
    assert request.demand_score > request.demand_kg
    assert 0.0 < request.to_dict()["location"]["lat"] < 90.0


def test_scenario_round_trip():
    scenario = Scenario(
        name="toy",
        seed=1,
        city="bengaluru",
        vehicles=(make_vehicle(), make_vehicle("V2")),
        requests=(make_request(), make_request("R2")),
    )
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    assert scenario.horizon_start_min == DEFAULT_HORIZON_START_MIN
    assert scenario.grace_minutes == DEFAULT_GRACE_MINUTES
    assert scenario.request_ids() == ("R1", "R2")
    assert scenario.vehicle_ids() == ("V1", "V2")
    assert scenario.total_demand_kg == pytest.approx(50.0)
    assert scenario.total_capacity_kg == pytest.approx(1000.0)
    assert scenario.capacity_ratio == pytest.approx(20.0)
    assert scenario.summary()["requests"] == 2


def test_scenario_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        Scenario(
            name="dup",
            seed=1,
            city="bengaluru",
            vehicles=(make_vehicle(), make_vehicle()),
            requests=(make_request(),),
        )
    with pytest.raises(ValueError):
        Scenario(
            name="dup",
            seed=1,
            city="bengaluru",
            vehicles=(make_vehicle(),),
            requests=(make_request(), make_request()),
        )


@pytest.mark.parametrize(
    "overrides",
    ({"speed_kmh": 0.0}, {"capacity_kg": 0.0}, {"capacity_m3": -1.0}),
)
def test_scenario_rejects_invalid_vehicles(overrides):
    with pytest.raises(ValueError):
        Scenario(
            name="bad",
            seed=1,
            city="bengaluru",
            vehicles=(make_vehicle(**overrides),),
            requests=(make_request(),),
        )


def test_scenario_allows_empty_books():
    empty = Scenario(name="empty", seed=1, city="bengaluru", vehicles=(), requests=())
    assert empty.summary()["capacity_ratio"] == 0.0
    assert empty.all_points() == []
