"""Plan scoring: the composite objective and the metrics surfaced by the API.

Every strategy minimises the same composite objective (lower is better)::

    objective = distance cost
              + lateness cost
              + unassigned cost (priority weighted)
              + makespan cost

All weights live in :class:`ObjectiveConfig`, so the trade-offs are explicit,
documented and overridable per request through the API. Distance cost uses each
vehicle's own ``cost_per_km``, so sending a cheaper vehicle further away can be
the right call - which is exactly the kind of trade-off the strategies differ on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .domain import DeliveryRequest, Scenario
from .feasibility import EPS, RouteEvaluation, evaluate_route


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights of the composite objective (cost units, not currency)."""

    lateness_cost_per_min: float = 2.0
    unassigned_base_cost: float = 60.0
    priority_step: float = 0.5
    makespan_cost_per_min: float = 0.05

    def unassigned_cost(self, request: DeliveryRequest) -> float:
        """A late-by-grace request is refused; refusing a VIP costs more."""
        weight = 1.0 + (request.priority - 1) * self.priority_step
        return self.unassigned_base_cost * weight

    def to_dict(self) -> dict:
        return {
            "lateness_cost_per_min": self.lateness_cost_per_min,
            "unassigned_base_cost": self.unassigned_base_cost,
            "priority_step": self.priority_step,
            "makespan_cost_per_min": self.makespan_cost_per_min,
        }


DEFAULT_OBJECTIVE = ObjectiveConfig()


def plan_evaluations(
    scenario: Scenario, routes: Mapping[str, Sequence[str]]
) -> dict:
    """Evaluate every route in ``routes`` (``vehicle_id -> [request_id, ...]``).

    Raises ``ValueError`` when a request is assigned twice, which would silently
    double-count demand and distance if it slipped through.
    """
    evaluations: dict = {}
    seen: set = set()
    for vehicle_id, request_ids in routes.items():
        vehicle = scenario.vehicle(vehicle_id)
        requests = []
        for request_id in request_ids:
            if request_id in seen:
                raise ValueError("request %s is assigned to more than one route" % request_id)
            seen.add(request_id)
            requests.append(scenario.request(request_id))
        evaluations[vehicle_id] = evaluate_route(vehicle, requests, scenario.grace_minutes)
    return evaluations


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _priority_weight(request: DeliveryRequest, step: float) -> float:
    return 1.0 + (request.priority - 1) * step


def compute_metrics(
    scenario: Scenario,
    routes: Mapping[str, Sequence[str]],
    config: ObjectiveConfig = DEFAULT_OBJECTIVE,
) -> dict:
    """Score a plan and return every KPI reported by the API, UI and ANALYSIS.md."""
    evaluations = plan_evaluations(scenario, routes)
    assigned_ids: set = set()
    for request_ids in routes.values():
        assigned_ids.update(request_ids)
    unassigned = [request for request in scenario.requests if request.id not in assigned_ids]

    distance_km = distance_cost = lateness_minutes = 0.0
    travel_minutes = waiting_minutes = service_minutes = 0.0
    stops_total = 0
    hard_violations: list = []
    late_stops = 0
    finishes: list = []
    per_vehicle: list = []

    for vehicle_id, evaluation in evaluations.items():
        vehicle = scenario.vehicle(vehicle_id)
        distance_km += evaluation.distance_km
        distance_cost += evaluation.distance_km * vehicle.cost_per_km
        lateness_minutes += evaluation.lateness_minutes
        travel_minutes += evaluation.travel_minutes
        waiting_minutes += evaluation.waiting_minutes
        service_minutes += evaluation.service_minutes
        stops_total += evaluation.stop_count
        if not evaluation.feasible:
            hard_violations.extend(
                "%s:%s" % (vehicle_id, violation) for violation in evaluation.violations
            )
        late_stops += sum(1 for stop in evaluation.stops if stop.lateness_min > EPS)
        if evaluation.stop_count:
            finishes.append(evaluation.finish_min)
        shift = vehicle.shift_minutes
        per_vehicle.append(
            {
                "vehicle_id": vehicle_id,
                "route": list(routes[vehicle_id]),
                "stops": evaluation.stop_count,
                "distance_km": round(evaluation.distance_km, 3),
                "busy_minutes": round(evaluation.busy_minutes, 2),
                "utilization_pct": round(evaluation.busy_minutes / shift * 100.0, 2) if shift else 0.0,
                "load_kg": round(evaluation.peak_load_kg, 2),
                "load_pct": round(evaluation.peak_load_kg / vehicle.capacity_kg * 100.0, 2),
                "finish_min": round(evaluation.finish_min, 2),
                "lateness_minutes": round(evaluation.lateness_minutes, 2),
                "feasible": evaluation.feasible,
            }
        )

    total_requests = len(scenario.requests)
    assigned_weight = sum(
        _priority_weight(scenario.request(rid), config.priority_step) for rid in assigned_ids
    )
    total_weight = sum(_priority_weight(r, config.priority_step) for r in scenario.requests)
    unassigned_cost = sum(config.unassigned_cost(request) for request in unassigned)
    makespan = (max(finishes) - scenario.horizon_start_min) if finishes else 0.0
    lateness_cost = lateness_minutes * config.lateness_cost_per_min
    makespan_cost = makespan * config.makespan_cost_per_min
    objective = distance_cost + lateness_cost + unassigned_cost + makespan_cost
    used = [entry for entry in per_vehicle if entry["stops"] > 0]
    workload = [entry["busy_minutes"] for entry in per_vehicle]

    return {
        "requests_total": total_requests,
        "requests_assigned": stops_total,
        "requests_unassigned": len(unassigned),
        "unassigned_ids": [request.id for request in unassigned],
        "coverage_pct": round(stops_total / total_requests * 100.0, 2) if total_requests else 0.0,
        "priority_coverage_pct": round(assigned_weight / total_weight * 100.0, 2) if total_weight else 0.0,
        "distance_km": round(distance_km, 3),
        "travel_minutes": round(travel_minutes, 2),
        "waiting_minutes": round(waiting_minutes, 2),
        "service_minutes": round(service_minutes, 2),
        "makespan_min": round(makespan, 2),
        "latest_finish_min": round(max(finishes), 2) if finishes else 0.0,
        "on_time_requests": stops_total - late_stops,
        "late_requests": late_stops,
        "lateness_minutes": round(lateness_minutes, 2),
        "max_lateness_min": round(
            max((stop.lateness_min for evaluation in evaluations.values() for stop in evaluation.stops), default=0.0),
            2,
        ),
        "on_time_pct": round((stops_total - late_stops) / stops_total * 100.0, 2) if stops_total else 100.0,
        "vehicles_total": len(scenario.vehicles),
        "vehicles_used": len(used),
        "avg_utilization_pct": round(
            sum(entry["utilization_pct"] for entry in per_vehicle) / len(per_vehicle), 2
        )
        if per_vehicle
        else 0.0,
        "avg_load_pct": round(sum(entry["load_pct"] for entry in used) / len(used), 2) if used else 0.0,
        "workload_stdev_minutes": round(_stdev(workload), 2),
        "workload_max_minutes": round(max(workload), 2) if workload else 0.0,
        "hard_violations": hard_violations,
        "valid": not hard_violations,
        "objective": round(objective, 3),
        "objective_components": {
            "distance_cost": round(distance_cost, 3),
            "lateness_cost": round(lateness_cost, 3),
            "unassigned_cost": round(unassigned_cost, 3),
            "makespan_cost": round(makespan_cost, 3),
        },
        "objective_config": config.to_dict(),
        "per_vehicle": per_vehicle,
    }


def objective_value(
    scenario: Scenario,
    routes: Mapping[str, Sequence[str]],
    config: ObjectiveConfig = DEFAULT_OBJECTIVE,
) -> float:
    """Just the scalar objective - handy for brute-force reference searches."""
    return compute_metrics(scenario, routes, config)["objective"]

