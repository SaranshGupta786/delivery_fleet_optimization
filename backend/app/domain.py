"""Domain model for the delivery-fleet resource allocation engine.

A :class:`Scenario` is an immutable snapshot of one allocation problem: a fleet
of vehicles (depot, capacities, shift window, cruising speed, cost per km,
capability set) and a set of delivery requests (location, demand, service time,
preferred delivery window, priority).

Times are minutes from midnight, so ``480.0`` is 08:00. Floats are used because
travel times are fractional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .geo import Point

CAPABILITIES: tuple = ("cold_chain", "fragile", "heavy_lift", "lift_gate")
PRIORITY_MIN = 1
PRIORITY_MAX = 5
DEFAULT_HORIZON_START_MIN = 8.0 * 60.0
DEFAULT_GRACE_MINUTES = 30.0


@dataclass(frozen=True)
class Vehicle:
    """A delivery resource: the thing we allocate work to."""

    id: str
    depot: Point
    capacity_kg: float
    capacity_m3: float
    shift_start_min: float
    shift_end_min: float
    max_route_minutes: float
    speed_kmh: float
    cost_per_km: float
    capabilities: frozenset = frozenset()

    @property
    def shift_minutes(self) -> float:
        return max(0.0, self.shift_end_min - self.shift_start_min)

    def supports(self, request: "DeliveryRequest") -> bool:
        """True when this vehicle has every capability the request needs."""
        return request.required_capabilities <= self.capabilities

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "depot": self.depot.to_dict(),
            "capacity_kg": self.capacity_kg,
            "capacity_m3": self.capacity_m3,
            "shift_start_min": self.shift_start_min,
            "shift_end_min": self.shift_end_min,
            "max_route_minutes": self.max_route_minutes,
            "speed_kmh": self.speed_kmh,
            "cost_per_km": self.cost_per_km,
            "capabilities": sorted(self.capabilities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Vehicle":
        return cls(
            id=str(data["id"]),
            depot=Point.from_dict(data["depot"]),
            capacity_kg=float(data["capacity_kg"]),
            capacity_m3=float(data["capacity_m3"]),
            shift_start_min=float(data["shift_start_min"]),
            shift_end_min=float(data["shift_end_min"]),
            max_route_minutes=float(data["max_route_minutes"]),
            speed_kmh=float(data["speed_kmh"]),
            cost_per_km=float(data["cost_per_km"]),
            capabilities=frozenset(data.get("capabilities") or ()),
        )


@dataclass(frozen=True)
class DeliveryRequest:
    """A unit of demand that must be routed to exactly one vehicle."""

    id: str
    location: Point
    demand_kg: float
    volume_m3: float
    service_minutes: float
    window_open_min: float
    window_close_min: float
    priority: int
    required_capabilities: frozenset = frozenset()

    @property
    def window_minutes(self) -> float:
        return max(0.0, self.window_close_min - self.window_open_min)

    @property
    def demand_score(self) -> float:
        """Rough "size" of the request, used for tie-breaking."""
        return self.demand_kg + 5.0 * self.volume_m3

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "location": self.location.to_dict(),
            "demand_kg": self.demand_kg,
            "volume_m3": self.volume_m3,
            "service_minutes": self.service_minutes,
            "window_open_min": self.window_open_min,
            "window_close_min": self.window_close_min,
            "priority": self.priority,
            "required_capabilities": sorted(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeliveryRequest":
        return cls(
            id=str(data["id"]),
            location=Point.from_dict(data["location"]),
            demand_kg=float(data["demand_kg"]),
            volume_m3=float(data["volume_m3"]),
            service_minutes=float(data["service_minutes"]),
            window_open_min=float(data["window_open_min"]),
            window_close_min=float(data["window_close_min"]),
            priority=int(data["priority"]),
            required_capabilities=frozenset(data.get("required_capabilities") or ()),
        )


@dataclass(frozen=True)
class Scenario:
    """One allocation problem instance (fleet + requests)."""

    name: str
    seed: int
    city: str
    vehicles: tuple
    requests: tuple
    horizon_start_min: float = DEFAULT_HORIZON_START_MIN
    grace_minutes: float = DEFAULT_GRACE_MINUTES
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_vehicle_index", {v.id: v for v in self.vehicles})
        object.__setattr__(self, "_request_index", {r.id: r for r in self.requests})
        if len(self._vehicle_index) != len(self.vehicles):
            raise ValueError("vehicle ids must be unique")
        if len(self._request_index) != len(self.requests):
            raise ValueError("request ids must be unique")
        for vehicle in self.vehicles:
            if vehicle.speed_kmh <= 0:
                raise ValueError("vehicle %s has non-positive speed" % vehicle.id)
            if vehicle.capacity_kg <= 0 or vehicle.capacity_m3 <= 0:
                raise ValueError("vehicle %s has non-positive capacity" % vehicle.id)

    @property
    def vehicle_map(self) -> Mapping[str, Vehicle]:
        return self._vehicle_index

    @property
    def request_map(self) -> Mapping[str, DeliveryRequest]:
        return self._request_index

    def vehicle(self, vehicle_id: str) -> Vehicle:
        return self._vehicle_index[vehicle_id]

    def request(self, request_id: str) -> DeliveryRequest:
        return self._request_index[request_id]

    def request_ids(self) -> tuple:
        return tuple(r.id for r in self.requests)

    def vehicle_ids(self) -> tuple:
        return tuple(v.id for v in self.vehicles)

    @property
    def total_demand_kg(self) -> float:
        return sum(r.demand_kg for r in self.requests)

    @property
    def total_capacity_kg(self) -> float:
        return sum(v.capacity_kg for v in self.vehicles)

    @property
    def capacity_ratio(self) -> float:
        if not self.total_demand_kg:
            return 0.0
        return self.total_capacity_kg / self.total_demand_kg

    def all_points(self) -> list:
        points = [v.depot for v in self.vehicles]
        points.extend(r.location for r in self.requests)
        return points

    def summary(self) -> dict:
        return {
            "name": self.name,
            "city": self.city,
            "seed": self.seed,
            "vehicles": len(self.vehicles),
            "requests": len(self.requests),
            "total_demand_kg": round(self.total_demand_kg, 1),
            "total_capacity_kg": round(self.total_capacity_kg, 1),
            "capacity_ratio": round(self.capacity_ratio, 3),
            "grace_minutes": self.grace_minutes,
            "horizon_start_min": self.horizon_start_min,
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "city": self.city,
            "horizon_start_min": self.horizon_start_min,
            "grace_minutes": self.grace_minutes,
            "metadata": dict(self.metadata),
            "vehicles": [v.to_dict() for v in self.vehicles],
            "requests": [r.to_dict() for r in self.requests],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        return cls(
            name=str(data.get("name", "scenario")),
            seed=int(data.get("seed", 0)),
            city=str(data.get("city", "unknown")),
            horizon_start_min=float(data.get("horizon_start_min", DEFAULT_HORIZON_START_MIN)),
            grace_minutes=float(data.get("grace_minutes", DEFAULT_GRACE_MINUTES)),
            metadata=dict(data.get("metadata") or {}),
            vehicles=tuple(Vehicle.from_dict(v) for v in data.get("vehicles") or ()),
            requests=tuple(DeliveryRequest.from_dict(r) for r in data.get("requests") or ()),
        )
