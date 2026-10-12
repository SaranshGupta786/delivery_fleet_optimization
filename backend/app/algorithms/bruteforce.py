"""Exhaustive reference search for tiny instances - the ground truth.

Enumerates every partition of the requests across vehicles (including leaving a
request unassigned) and every visiting order inside each vehicle, then scores
each complete plan with exactly the same composite objective the real strategies
use. With <= 6 requests and <= 3 vehicles that is a few hundred thousand
candidate plans - cheap enough for the test suite, and it gives a provably
optimal baseline for the optimality claims in the README and ANALYSIS.md.

This is deliberately *not* a production strategy: the search is exponential, so
it refuses anything beyond its guard rails.
"""

from __future__ import annotations

import time
from itertools import permutations, product

from ..domain import Scenario
from ..feasibility import evaluate_route
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .base import AllocationPlan, RouteState, build_plan, explain_unassigned

MAX_REQUESTS = 6
MAX_VEHICLES = 3
MAX_CANDIDATES = 3_000_000


class BruteForceTooLarge(ValueError):
    """Raised when the instance is beyond what exhaustive search can handle."""


class ExactSolution:
    """Provably optimal split/order for a tiny instance."""

    __slots__ = ("objective", "routes", "unassigned_ids", "evaluated")

    def __init__(self, objective: float, routes: dict, unassigned_ids: tuple, evaluated: int) -> None:
        self.objective = objective
        self.routes = routes
        self.unassigned_ids = unassigned_ids
        self.evaluated = evaluated

    def to_dict(self) -> dict:
        return {
            "objective": round(self.objective, 3),
            "routes": {key: list(value) for key, value in self.routes.items()},
            "unassigned_ids": list(self.unassigned_ids),
            "evaluated_plans": self.evaluated,
        }


def solve(
    scenario: Scenario,
    config: ObjectiveConfig = DEFAULT_OBJECTIVE,
    max_requests: int = MAX_REQUESTS,
    max_vehicles: int = MAX_VEHICLES,
    max_candidates: int = MAX_CANDIDATES,
) -> ExactSolution:
    """Exhaustively minimise the composite objective; ties keep the first plan."""
    requests = list(scenario.requests)
    vehicles = list(scenario.vehicles)
    if len(requests) > max_requests or len(vehicles) > max_vehicles:
        raise BruteForceTooLarge(
            "exhaustive search is limited to %d requests and %d vehicles (got %d/%d)"
            % (max_requests, max_vehicles, len(requests), len(vehicles))
        )

    park = len(vehicles)
    best_objective = None
    best_routes = None
    best_unassigned: tuple = ()
    evaluated = 0

    for assignment in product(range(len(vehicles) + 1), repeat=len(requests)):
        buckets: list = [[] for _ in range(len(vehicles))]
        unassigned = []
        for request, target in zip(requests, assignment):
            if target == park:
                unassigned.append(request)
            else:
                buckets[target].append(request)
        unassigned_cost = sum(config.unassigned_cost(request) for request in unassigned)
        orderings = [list(permutations(bucket)) if bucket else [()] for bucket in buckets]

        for ordering in product(*orderings):
            evaluated += 1
            if evaluated > max_candidates:
                raise BruteForceTooLarge(
                    "exhaustive search exceeded %d candidate plans" % max_candidates
                )
            distance_cost = lateness_minutes = 0.0
            finishes: list = []
            feasible = True
            for vehicle, ordered in zip(vehicles, ordering):
                if not ordered:
                    continue
                evaluation = evaluate_route(vehicle, ordered, scenario.grace_minutes)
                if not evaluation.feasible:
                    feasible = False
                    break
                distance_cost += evaluation.distance_km * vehicle.cost_per_km
                lateness_minutes += evaluation.lateness_minutes
                finishes.append(evaluation.finish_min)
            if not feasible:
                continue
            makespan = (max(finishes) - scenario.horizon_start_min) if finishes else 0.0
            objective = (
                distance_cost
                + lateness_minutes * config.lateness_cost_per_min
                + unassigned_cost
                + makespan * config.makespan_cost_per_min
            )
            if best_objective is None or objective < best_objective - 1e-9:
                best_objective = objective
                best_routes = {
                    vehicle.id: tuple(request.id for request in ordered)
                    for vehicle, ordered in zip(vehicles, ordering)
                }
                best_unassigned = tuple(request.id for request in unassigned)

    if best_routes is None:  # no feasible plan at all: everything stays unassigned
        best_objective = sum(config.unassigned_cost(request) for request in requests)
        best_routes = {vehicle.id: () for vehicle in vehicles}
        best_unassigned = tuple(request.id for request in requests)

    return ExactSolution(best_objective, best_routes, best_unassigned, evaluated)


class BruteForceAllocator:
    """Exposes the exhaustive solver as a strategy (guarded, tiny instances)."""

    name = "exact"
    label = "Exhaustive reference (tiny instances)"
    description = (
        "Enumerates every request-to-vehicle split and every visiting order, so its "
        "plan is provably optimal for the composite objective. Exponential: limited "
        "to <= %d requests and %d vehicles." % (MAX_REQUESTS, MAX_VEHICLES)
    )
    kind = "exact_reference"

    def __init__(self, max_requests: int = MAX_REQUESTS, max_vehicles: int = MAX_VEHICLES) -> None:
        self.max_requests = max_requests
        self.max_vehicles = max_vehicles

    def allocate(
        self, scenario: Scenario, config: ObjectiveConfig = DEFAULT_OBJECTIVE, **options
    ) -> AllocationPlan:
        started = time.perf_counter()
        solution = solve(scenario, config, self.max_requests, self.max_vehicles)
        runtime_ms = (time.perf_counter() - started) * 1000.0
        states = {
            vehicle.id: RouteState(scenario, vehicle, config) for vehicle in scenario.vehicles
        }
        reasons = {
            request_id: explain_unassigned(scenario, scenario.request(request_id), states)
            for request_id in solution.unassigned_ids
        }
        return build_plan(
            scenario,
            self.name,
            self.label,
            routes=solution.routes,
            unassigned_reasons=reasons,
            runtime_ms=runtime_ms,
            config=config,
            notes=[
                "%d candidate plans evaluated (no pruning, no heuristics)"
                % solution.evaluated,
                "provably optimal for the composite objective on this instance",
            ],
        )
