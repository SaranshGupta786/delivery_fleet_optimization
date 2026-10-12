"""Hungarian (Jonker-Volgenant) batch matching - the optimal-per-round strategy.

Each round builds a cost matrix over every (unassigned request x vehicle) pair
using the incremental insertion cost, then solves it exactly with
``scipy.optimize.linear_sum_assignment``. That yields the *globally* optimal
one-request-per-vehicle batch for the current routes; rounds repeat until no
feasible pair remains. The overall plan is therefore optimal per round, not
globally (global vehicle routing is NP-hard), which is what makes the comparison
against the greedy baseline meaningful.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..domain import Scenario
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .base import AllocationPlan, RouteState, build_plan, explain_unassigned

INFEASIBLE = 1e9


class HungarianAllocator:
    """Repeated exact batch matching on incremental insertion cost."""

    name = "hungarian"
    label = "Hungarian batch matching"
    description = (
        "Each round solves the optimal request-to-vehicle matching with "
        "scipy.optimize.linear_sum_assignment on incremental insertion cost; rounds "
        "repeat until no feasible pair is left. Optimal per round, not globally."
    )
    kind = "optimal_batch"

    def allocate(
        self,
        scenario: Scenario,
        config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        max_rounds: int = 500,
        lateness_weight: float = 0.0,
        **options
    ) -> AllocationPlan:
        started = time.perf_counter()
        vehicles = list(scenario.vehicles)
        states = {
            vehicle.id: RouteState(scenario, vehicle, config) for vehicle in vehicles
        }
        pending: list = list(scenario.requests)
        reasons: dict = {}
        rounds = 0
        matched_total = 0

        while pending and rounds < int(max_rounds):
            rounds += 1
            costs = np.full((len(pending), len(vehicles)), INFEASIBLE, dtype=float)
            insertions: dict = {}
            for row, request in enumerate(pending):
                for column, vehicle in enumerate(vehicles):
                    insertion = states[vehicle.id].best_insertion(request)
                    if insertion is None:
                        continue
                    costs[row, column] = (
                        insertion.cost + lateness_weight * insertion.lateness_delta_min
                    )
                    insertions[(row, column)] = insertion

            if not insertions:
                for request in pending:
                    reasons[request.id] = explain_unassigned(scenario, request, states)
                break

            rows, columns = linear_sum_assignment(costs)
            matched: list = []
            for row, column in zip(rows, columns):
                if costs[row, column] >= INFEASIBLE:
                    continue
                request = pending[row]
                vehicle = vehicles[column]
                insertion = insertions[(row, column)]
                states[vehicle.id].insert(request, insertion.position)
                matched.append(request.id)
            matched_total += len(matched)

            if not matched:
                for request in pending:
                    reasons.setdefault(request.id, explain_unassigned(scenario, request, states))
                break

            matched_set = set(matched)
            pending = [request for request in pending if request.id not in matched_set]

        for request in pending:
            reasons.setdefault(request.id, explain_unassigned(scenario, request, states))

        runtime_ms = (time.perf_counter() - started) * 1000.0
        return build_plan(
            scenario,
            self.name,
            self.label,
            routes={vehicle_id: state.request_ids() for vehicle_id, state in states.items()},
            unassigned_reasons=reasons,
            runtime_ms=runtime_ms,
            config=config,
            notes=[
                "%d batch rounds, %d requests matched" % (rounds, matched_total),
                "solver: scipy.optimize.linear_sum_assignment (Jonker-Volgenant)",
                "batch cost = incremental distance cost (+ %.2f x lateness minutes)" % lateness_weight,
            ],
        )
