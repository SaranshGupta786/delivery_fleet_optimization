"""Small builders for hand-made vehicles, requests and scenarios."""

from __future__ import annotations

from app.domain import DeliveryRequest, Point, Scenario, Vehicle

DEFAULT_DEPOT = Point(12.95, 77.60)


def make_vehicle(vehicle_id: str = "V1", depot: Point = None, **overrides) -> Vehicle:
    payload = dict(
        id=vehicle_id,
        depot=depot or DEFAULT_DEPOT,
        capacity_kg=500.0,
        capacity_m3=3.0,
        shift_start_min=480.0,
        shift_end_min=1020.0,
        max_route_minutes=420.0,
        speed_kmh=25.0,
        cost_per_km=1.0,
        capabilities=frozenset({"cold_chain"}),
    )
    payload.update(overrides)
    return Vehicle(**payload)


def make_request(request_id: str = "R1", location: Point = None, **overrides) -> DeliveryRequest:
    payload = dict(
        id=request_id,
        location=location or Point(12.96, 77.61),
        demand_kg=25.0,
        volume_m3=0.2,
        service_minutes=8.0,
        window_open_min=540.0,
        window_close_min=720.0,
        priority=3,
        required_capabilities=frozenset(),
    )
    payload.update(overrides)
    return DeliveryRequest(**payload)


def make_scenario(vehicles, requests, **overrides) -> Scenario:
    payload = dict(
        name="toy",
        seed=1,
        city="bengaluru",
        vehicles=tuple(vehicles),
        requests=tuple(requests),
    )
    payload.update(overrides)
    return Scenario(**payload)
