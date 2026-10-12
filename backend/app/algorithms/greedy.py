"""Greedy cheapest-insertion strategy - the fast, obvious baseline.

Requests are ordered by deadline (or priority / demand) and each one is pushed
into whichever vehicle makes it cheapest to serve, subject to every hard
constraint. It is O(n * m * route length), never revisits a decision, and is the
yardstick the other strategies have to beat.
"""

from __future__ import annotations

import time

from ..domain import DeliveryRequest, Scenario
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .base import AllocationPlan, RouteState, build_plan, explain_unassigned

ORDERINGS = ("deadline", "priority", "demand")


def _ordering_key(order: str):
    if order == "priority":
        return lambda request: (-request.priority, request.window_close_min, request.id)
    if order == "demand":
        return lambda request: (-request.demand_kg, request.window_close_min, request.id)
    return lambda request: (
        request.window_close_min,
        -request.priority,
        -request.demand_kg,
        request.id,
    )


class GreedyAllocator:
    """Cheapest insertion, one request at a time, in a fixed priority order."""

    name = "greedy"
    label = "Greedy cheapest insertion"
    description = (
        "Sort requests by deadline/priority/demand and insert each into the vehicle "
        "where it is cheapest (distance + priced lateness), honouring all hard "
        "constraints. Fast, never revisits a decision."
    )
    kind = "heuristic"

    def allocate(
        self,
        scenario: Scenario,
        config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        ordering: str = "deadline",
        lateness_weight: float = None,
        **options
    ) -> AllocationPlan:
        if ordering not in ORDERINGS:
            raise ValueError(
                "unknown ordering %r (expected one of %s)" % (ordering, ", ".join(ORDERINGS))
            )
        weight = (
            config.lateness_cost_per_min if lateness_weight is None else float(lateness_weight)
        )
        started = time.perf_counter()
        states = {
            vehicle.id: RouteState(scenario, vehicle, config) for vehicle in scenario.vehicles
        }
        reasons: dict = {}

        requests: list = sorted(scenario.requests, key=_ordering_key(ordering))
        for request in requests:
            best = None
            for vehicle_id, state in states.items():
                insertion = state.best_insertion(
                    request,
                    key=lambda ins: (
                        round(ins.cost + weight * ins.lateness_delta_min, 6),
                        ins.position,
                    ),
                )
                if insertion is None:
                    continue
                score = (round(insertion.cost + weight * insertion.lateness_delta_min, 6), vehicle_id)
                if best is None or score < best[0]:
                    best = (score, vehicle_id, insertion)
            if best is None:
                reasons[request.id] = explain_unassigned(scenario, request, states)
                continue
            _, vehicle_id, insertion = best
            states[vehicle_id].insert(request, insertion.position)

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
                "request order: %s" % ordering,
                "candidate cost = incremental distance cost + %.2f x lateness minutes" % weight,
            ],
        )
