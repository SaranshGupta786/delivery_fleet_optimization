"""Deterministic synthetic scenario generator.

A scenario is produced purely from ``(preset, seed)``: the same seed always
yields the same fleet and the same requests, so every strategy is compared on
identical inputs and the numbers in ANALYSIS.md are reproducible. Randomness
comes from ``numpy.random.default_rng`` (PCG64), which is stable for a pinned
numpy version.

Demand is drawn around a handful of hotspots rather than uniformly, so the
resulting routes have realistic structure (clusters, long cross-town legs) and
the strategies actually differ.

Assumptions and their consequences are documented in the README; the short
version is: this is a synthetic last-mile delivery book, not an ingestion layer
for a real order feed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .domain import DEFAULT_GRACE_MINUTES, DeliveryRequest, Scenario, Vehicle
from .geo import Point

CITY_BOXES: dict = {
    "bengaluru": (12.88, 77.48, 13.07, 77.72),
    "berlin": (52.44, 13.28, 52.58, 13.55),
    "san_francisco": (37.70, -122.52, 37.81, -122.36),
}

VEHICLE_CAPACITY_KG = (180.0, 350.0, 600.0, 1000.0, 1600.0)
VEHICLE_CAPABILITY_PROBABILITY = {
    "cold_chain": 0.45,
    "fragile": 0.18,
    "heavy_lift": 0.22,
    "lift_gate": 0.30,
}
REQUEST_CAPABILITY_PROBABILITY = {
    "cold_chain": 0.22,
    "fragile": 0.10,
    "heavy_lift": 0.08,
    "lift_gate": 0.12,
}
SHIFT_STARTS = (480.0, 510.0, 540.0, 570.0)
SHIFT_LENGTHS = (300.0, 360.0, 420.0, 480.0, 540.0)
WINDOW_OPENS = (480.0, 540.0, 570.0, 600.0, 660.0)
WINDOW_WIDTHS = (60.0, 90.0, 120.0, 180.0, 240.0, 300.0)
PRIORITY_CHOICES = (1, 2, 3, 4, 5)
PRIORITY_PROBABILITIES = (0.10, 0.20, 0.35, 0.25, 0.10)

PRESETS: dict = {
    "small": {
        "description": "6 vehicles / 40 requests - fast enough for the UI default.",
        "vehicles": 6,
        "requests": 40,
        "capacity_ratio": 1.60,
        "window_scale": 1.0,
    },
    "medium": {
        "description": "14 vehicles / 140 requests - the analysis workhorse.",
        "vehicles": 14,
        "requests": 140,
        "capacity_ratio": 1.45,
        "window_scale": 1.0,
    },
    "large": {
        "description": "30 vehicles / 400 requests - latency stress for the API.",
        "vehicles": 30,
        "requests": 400,
        "capacity_ratio": 1.40,
        "window_scale": 1.0,
    },
    "tight": {
        "description": "Capacity and time windows tightened: coverage drops, regret wins.",
        "vehicles": 10,
        "requests": 120,
        "capacity_ratio": 1.12,
        "window_scale": 0.60,
    },
    "stress": {
        "description": "Deliberately over-subscribed: no strategy can cover everything.",
        "vehicles": 12,
        "requests": 260,
        "capacity_ratio": 1.00,
        "window_scale": 0.50,
    },
}


def preset_names() -> list:
    return list(PRESETS.keys())


def preset_info() -> list:
    info = []
    for name, params in PRESETS.items():
        entry = {"name": name}
        entry.update(params)
        info.append(entry)
    return info


def _make_vehicles(rng, box, count: int) -> list:
    """A small fleet spread over 1-3 depots, with mixed capabilities and costs."""
    depot_count = int(min(3, max(1, count)))
    depots = [
        Point(
            lat=float(rng.uniform(box[0] + 0.010, box[2] - 0.010)),
            lon=float(rng.uniform(box[1] + 0.010, box[3] - 0.010)),
        )
        for _ in range(depot_count)
    ]
    vehicles = []
    for index in range(count):
        capacity_kg = float(rng.choice(VEHICLE_CAPACITY_KG))
        capacity_m3 = round(max(0.6, capacity_kg / 180.0 + float(rng.uniform(-0.3, 0.6))), 3)
        capabilities = {
            capability
            for capability, probability in VEHICLE_CAPABILITY_PROBABILITY.items()
            if rng.random() < probability
        }
        if capacity_kg < 400.0:
            capabilities.discard("heavy_lift")
        shift_start = float(rng.choice(SHIFT_STARTS))
        shift_minutes = float(rng.choice(SHIFT_LENGTHS))
        vehicles.append(
            Vehicle(
                id="V%02d" % (index + 1),
                depot=depots[index % depot_count],
                capacity_kg=capacity_kg,
                capacity_m3=capacity_m3,
                shift_start_min=shift_start,
                shift_end_min=shift_start + shift_minutes,
                max_route_minutes=round(shift_minutes * float(rng.uniform(0.7, 1.0)), 1),
                speed_kmh=round(float(rng.uniform(20.0, 34.0)), 1),
                cost_per_km=round(float(rng.uniform(0.6, 1.3)), 2),
                capabilities=frozenset(capabilities),
            )
        )
    return vehicles


def _make_requests(rng, box, count: int, fleet_capabilities, clustered: bool, window_scale: float) -> list:
    """Demand clustered around 3-5 hotspots, with priority and capability mix."""
    hotspot_count = int(rng.integers(3, 6))
    hotspots = [
        (
            float(rng.uniform(box[0] + 0.015, box[2] - 0.015)),
            float(rng.uniform(box[1] + 0.015, box[3] - 0.015)),
        )
        for _ in range(hotspot_count)
    ]
    requests = []
    for index in range(count):
        if clustered and rng.random() < 0.8:
            centre = hotspots[int(rng.integers(0, hotspot_count))]
            lat = centre[0] + float(rng.normal(0.0, 0.012))
            lon = centre[1] + float(rng.normal(0.0, 0.015))
        else:
            lat = float(rng.uniform(box[0], box[2]))
            lon = float(rng.uniform(box[1], box[3]))
        lat = min(max(lat, box[0]), box[2])
        lon = min(max(lon, box[1]), box[3])
        demand_kg = round(float(np.clip(rng.lognormal(3.2, 0.7), 6.0, 160.0)), 1)
        volume_m3 = round(demand_kg / 220.0 * float(rng.uniform(0.7, 1.6)), 3)
        service_minutes = round(float(np.clip(4.0 + demand_kg / 12.0, 4.0, 20.0)), 1)
        window_open = float(rng.choice(WINDOW_OPENS))
        window_close = window_open + round(float(rng.choice(WINDOW_WIDTHS)) * window_scale, 1)
        priority = int(rng.choice(PRIORITY_CHOICES, p=PRIORITY_PROBABILITIES))
        required = {
            capability
            for capability in sorted(fleet_capabilities)
            if rng.random() < REQUEST_CAPABILITY_PROBABILITY.get(capability, 0.1)
        }
        if demand_kg < 70.0:
            required.discard("heavy_lift")
        requests.append(
            DeliveryRequest(
                id="R%03d" % (index + 1),
                location=Point(lat=lat, lon=lon),
                demand_kg=demand_kg,
                volume_m3=volume_m3,
                service_minutes=service_minutes,
                window_open_min=window_open,
                window_close_min=window_close,
                priority=priority,
                required_capabilities=frozenset(required),
            )
        )
    return requests


GENERATOR_PARAMETERS = (
    "preset",
    "seed",
    "city",
    "vehicles",
    "requests",
    "capacity_ratio",
    "window_scale",
    "grace_minutes",
    "clustered",
)


def generate_scenario(
    preset: str = "medium",
    seed: int = 42,
    city: str = "bengaluru",
    vehicles: int = None,
    requests: int = None,
    capacity_ratio: float = None,
    window_scale: float = None,
    grace_minutes: float = None,
    clustered: bool = True,
) -> Scenario:
    """Build a reproducible scenario; explicit arguments override the preset."""
    if preset not in PRESETS:
        raise ValueError("unknown preset %r (expected one of %s)" % (preset, ", ".join(PRESETS)))
    params = PRESETS[preset]
    fleet_size = int(vehicles if vehicles is not None else params["vehicles"])
    demand_size = int(requests if requests is not None else params["requests"])
    ratio = float(capacity_ratio if capacity_ratio is not None else params["capacity_ratio"])
    window_scale_value = float(window_scale if window_scale is not None else params["window_scale"])
    grace = float(grace_minutes if grace_minutes is not None else DEFAULT_GRACE_MINUTES)
    if fleet_size < 0 or demand_size < 0:
        raise ValueError("vehicles and requests must be non-negative")
    if ratio <= 0.0 or window_scale_value <= 0.0:
        raise ValueError("capacity_ratio and window_scale must be positive")

    rng = np.random.default_rng(int(seed))
    resolved_city = city if city in CITY_BOXES else "bengaluru"
    box = CITY_BOXES[resolved_city]
    fleet = _make_vehicles(rng, box, fleet_size)
    fleet_capabilities = (
        frozenset().union(*(vehicle.capabilities for vehicle in fleet)) if fleet else frozenset()
    )
    demand = _make_requests(
        rng, box, demand_size, fleet_capabilities, bool(clustered), window_scale_value
    )

    # Scale the fleet to the requested demand/capacity ratio: this is the knob
    # that turns "comfortable" into "over-subscribed" scenarios.
    total_demand = sum(request.demand_kg for request in demand)
    current_capacity = sum(vehicle.capacity_kg for vehicle in fleet)
    if total_demand > 0.0 and current_capacity > 0.0:
        factor = total_demand * ratio / current_capacity
        fleet = [
            replace(
                vehicle,
                capacity_kg=round(vehicle.capacity_kg * factor, 1),
                capacity_m3=round(vehicle.capacity_m3 * factor, 3),
            )
            for vehicle in fleet
        ]

    return Scenario(
        name="%s-seed%d" % (preset, int(seed)),
        seed=int(seed),
        city=resolved_city,
        vehicles=tuple(fleet),
        requests=tuple(demand),
        grace_minutes=grace,
        metadata={
            "preset": preset,
            "city": resolved_city,
            "vehicles": fleet_size,
            "requests": demand_size,
            "capacity_ratio": ratio,
            "window_scale": window_scale_value,
            "clustered": bool(clustered),
            "generator": "app.scenario.generate_scenario",
            "travel_model": "constant-speed haversine (no traffic, no paid routing)",
        },
    )


def scenario_from_spec(spec: Mapping[str, Any]) -> Scenario:
    """Build a scenario from an API payload: explicit fleet, or preset + knobs."""
    if not isinstance(spec, Mapping):
        raise ValueError("scenario spec must be a JSON object")
    explicit = isinstance(spec.get("vehicles"), (list, tuple)) and isinstance(
        spec.get("requests"), (list, tuple)
    )
    if explicit:
        return Scenario.from_dict(spec)
    kwargs = {
        key: spec[key] for key in GENERATOR_PARAMETERS if key in spec and spec[key] is not None
    }
    return generate_scenario(**kwargs)

