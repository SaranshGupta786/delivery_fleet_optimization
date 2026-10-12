"""Shared plan container, explanation strings and the strategy protocol.

Every strategy emits the same :class:`AllocationPlan`, so the comparison table,
the map and the explainability panel work identically for all of them. Plans are
assembled here rather than inside each strategy so that metrics, per-stop
explanations and unassignment reasons can never drift between strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Sequence

from ..domain import DeliveryRequest, Scenario, Vehicle
from ..feasibility import Insertion, RouteEvaluation, Stop, best_insertion, evaluate_route
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig, compute_metrics, plan_evaluations


def format_clock(minutes: float) -> str:
    """Format minutes-from-midnight as ``HH:MM``."""
    total = int(round(minutes))
    return "%02d:%02d" % (total // 60, total % 60)


@dataclass(frozen=True)
class Assignment:
    """One request served by one vehicle, with the reasoning behind it."""

    request_id: str
    vehicle_id: str
    sequence: int
    arrival_min: float
    lateness_min: float
    leg_km: float
    load_kg: float
    load_pct: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "vehicle_id": self.vehicle_id,
            "sequence": self.sequence,
            "arrival_min": round(self.arrival_min, 2),
            "arrival_clock": format_clock(self.arrival_min),
            "lateness_min": round(self.lateness_min, 2),
            "leg_km": round(self.leg_km, 3),
            "load_kg": round(self.load_kg, 2),
            "load_pct": round(self.load_pct, 2),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class UnassignedRequest:
    """A request no vehicle could serve, plus the reason."""

    request_id: str
    priority: int
    reason: str

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "priority": self.priority, "reason": self.reason}


@dataclass(frozen=True)
class AllocationPlan:
    """The uniform result every strategy returns."""

    algorithm: str
    label: str
    scenario_name: str
    routes: Mapping[str, tuple]
    assignments: tuple
    unassigned: tuple
    metrics: dict
    runtime_ms: float
    notes: tuple

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "label": self.label,
            "scenario": self.scenario_name,
            "runtime_ms": round(self.runtime_ms, 3),
            "routes": {key: list(value) for key, value in self.routes.items()},
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "unassigned": [unassigned.to_dict() for unassigned in self.unassigned],
            "metrics": self.metrics,
            "notes": list(self.notes),
        }


ReasonBuilder = Callable[[Scenario, Vehicle, DeliveryRequest, Stop, RouteEvaluation, int], str]


def default_reason(scenario, vehicle, request, stop, evaluation, sequence) -> str:
    """One sentence of justification, shared by every strategy."""
    previous = (
        "the depot"
        if sequence == 0
        else scenario.request(evaluation.stops[sequence - 1].request_id).id
    )
    parts = [
        "stop %d/%d, %.1f km from %s" % (sequence + 1, evaluation.stop_count, stop.leg_km, previous)
    ]
    if stop.lateness_min > 0.05:
        parts.append(
            "arrives %s, %.0f min past the %s deadline (inside the %.0f min grace)"
            % (
                format_clock(stop.arrival_min),
                stop.lateness_min,
                format_clock(request.window_close_min),
                scenario.grace_minutes,
            )
        )
    else:
        parts.append(
            "arrives %s, %.0f min before the %s deadline"
            % (
                format_clock(stop.arrival_min),
                request.window_close_min - stop.arrival_min,
                format_clock(request.window_close_min),
            )
        )
    if stop.waiting_min > 0.5:
        parts.append(
            "waits %.0f min for the %s window"
            % (stop.waiting_min, format_clock(request.window_open_min))
        )
    parts.append(
        "load %d%% of %.0f kg" % (round(stop.load_kg / vehicle.capacity_kg * 100.0), vehicle.capacity_kg)
    )
    if request.required_capabilities:
        parts.append("uses %s" % ", ".join(sorted(request.required_capabilities)))
    return "%s serves %s: %s." % (vehicle.id, request.id, "; ".join(parts))


def build_plan(
    scenario: Scenario,
    algorithm: str,
    label: str,
    routes: Mapping[str, Sequence[str]],
    unassigned_reasons: Mapping[str, str],
    runtime_ms: float,
    notes: Sequence[str] = (),
    config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    reason_builder: Optional[ReasonBuilder] = None,
) -> AllocationPlan:
    """Turn ``vehicle_id -> [request_id, ...]`` into a fully explained plan."""
    evaluations = plan_evaluations(scenario, routes)
    build_reason = reason_builder or default_reason
    assignments: list = []
    assigned_ids: set = set()

    for vehicle_id, request_ids in routes.items():
        vehicle = scenario.vehicle(vehicle_id)
        evaluation = evaluations[vehicle_id]
        for sequence, stop in enumerate(evaluation.stops):
            request = scenario.request(stop.request_id)
            assigned_ids.add(request.id)
            assignments.append(
                Assignment(
                    request_id=request.id,
                    vehicle_id=vehicle_id,
                    sequence=sequence,
                    arrival_min=stop.arrival_min,
                    lateness_min=stop.lateness_min,
                    leg_km=stop.leg_km,
                    load_kg=stop.load_kg,
                    load_pct=stop.load_kg / vehicle.capacity_kg * 100.0,
                    reason=build_reason(scenario, vehicle, request, stop, evaluation, sequence),
                )
            )

    unassigned = tuple(
        UnassignedRequest(
            request_id=request.id,
            priority=request.priority,
            reason=unassigned_reasons.get(request.id)
            or "no feasible insertion into any vehicle's route",
        )
        for request in scenario.requests
        if request.id not in assigned_ids
    )

    return AllocationPlan(
        algorithm=algorithm,
        label=label,
        scenario_name=scenario.name,
        routes={vehicle_id: tuple(request_ids) for vehicle_id, request_ids in routes.items()},
        assignments=tuple(assignments),
        unassigned=unassigned,
        metrics=compute_metrics(scenario, routes, config),
        runtime_ms=runtime_ms,
        notes=tuple(notes),
    )


class RouteState:
    """One vehicle's route while a strategy is building it.

    Caches the current :class:`RouteEvaluation` so that scoring a candidate
    request costs O(route length) instead of re-simulating the whole route for
    every candidate - this is what keeps the regret heuristic usable on the
    large preset.
    """

    __slots__ = ("scenario", "vehicle", "config", "requests", "evaluation")

    def __init__(
        self, scenario: Scenario, vehicle: Vehicle, config: ObjectiveConfig = DEFAULT_OBJECTIVE
    ) -> None:
        self.scenario = scenario
        self.vehicle = vehicle
        self.config = config
        self.requests: list = []
        self.evaluation = evaluate_route(vehicle, (), scenario.grace_minutes)

    @property
    def vehicle_id(self) -> str:
        return self.vehicle.id

    @property
    def stop_count(self) -> int:
        return len(self.requests)

    @property
    def distance_cost(self) -> float:
        return self.evaluation.distance_km * self.vehicle.cost_per_km

    @property
    def feasible(self) -> bool:
        return self.evaluation.feasible

    def best_insertion(
        self, request: DeliveryRequest, key: Optional[Callable[[Insertion], tuple]] = None
    ) -> Optional[Insertion]:
        return best_insertion(
            self.vehicle,
            self.requests,
            request,
            self.scenario.grace_minutes,
            self.evaluation,
            key,
        )

    def insert(self, request: DeliveryRequest, position: int) -> None:
        self.requests.insert(position, request)
        self.evaluation = evaluate_route(self.vehicle, self.requests, self.scenario.grace_minutes)

    def remove(self, request_id: str) -> DeliveryRequest:
        for index, request in enumerate(self.requests):
            if request.id == request_id:
                self.requests.pop(index)
                self.evaluation = evaluate_route(
                    self.vehicle, self.requests, self.scenario.grace_minutes
                )
                return request
        raise KeyError(request_id)

    def request_ids(self) -> tuple:
        return tuple(request.id for request in self.requests)


class Allocator(Protocol):
    """The contract every strategy implements; the API only sees this."""

    name: str
    label: str
    description: str
    kind: str

    def allocate(
        self, scenario: Scenario, config: ObjectiveConfig = DEFAULT_OBJECTIVE, **options
    ) -> AllocationPlan:
        ...
def explain_unassigned(scenario: Scenario, request: DeliveryRequest, states: Mapping[str, RouteState]) -> str:
    """Why a request could not be placed - shared by every strategy."""
    capable = [state for state in states.values() if state.vehicle.supports(request)]
    if not capable:
        return "requires %s, which no vehicle in the fleet provides" % (
            ", ".join(sorted(request.required_capabilities))
        )
    oversized = [
        state
        for state in capable
        if state.vehicle.capacity_kg < request.demand_kg
        or state.vehicle.capacity_m3 < request.volume_m3
    ]
    if len(oversized) == len(capable):
        return "%.0f kg / %.2f m3 exceeds the capacity of every capable vehicle" % (
            request.demand_kg,
            request.volume_m3,
        )
    return (
        "no feasible slot left: every capable vehicle would miss the %s deadline "
        "(+%.0f min grace), run past its shift end or exceed its route limit"
        % (format_clock(request.window_close_min), scenario.grace_minutes)
    )

