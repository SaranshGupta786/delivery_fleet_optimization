"""Deadline-regret insertion with local search - the custom heuristic.

Two ideas combined, both of them things a dispatcher actually does:

1. *Regret insertion*: repeatedly commit the request that loses the most if it is
   not placed now - the gap between its best and second-best vehicle slot - and
   break ties towards VIPs and the earliest deadline. Unlike greedy it never
   spends a scarce slot on a request that had plenty of alternatives.
2. *Local search*: then relocate single requests (inside a route or across
   routes) while that strictly lowers the priced route cost (distance + priced
   lateness), accepting only moves verified by full route simulation.

Both passes are deterministic: ties break on vehicle id and request id.
"""

from __future__ import annotations

import time

from ..domain import DeliveryRequest, Scenario
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .base import (
    AllocationPlan,
    RouteState,
    build_plan,
    default_reason,
    explain_unassigned,
)


def _reason_builder(margins):
    """Explain a regret assignment by quoting the margin that selected it."""

    def builder(scenario, vehicle, request, stop, evaluation, sequence):
        text = default_reason(scenario, vehicle, request, stop, evaluation, sequence)
        margin = margins.get(request.id)
        if margin is not None:
            text += (
                " Regret rule: %.1f cost units cheaper here than in the next-best "
                "vehicle slot." % margin
            )
        return text

    return builder


def relocate_pass(states, weight: float) -> int:
    """One full relocate pass over the routes; returns the number of accepted moves.

    Every candidate move is verified with real route simulations (the states are
    mutated and then re-evaluated), and a move is kept only when the priced route
    cost - distance plus ``weight`` x lateness minutes - strictly drops. Ties and
    scan order are deterministic, so two runs produce identical plans. Shared by
    the regret heuristic and by the optional polish step of the learned policy.
    """
    improved = 0
    for origin in list(states.values()):
        for request in list(origin.requests):
            before = sum(
                state.distance_cost + weight * state.evaluation.lateness_minutes
                for state in states.values()
            )
            position = origin.requests.index(request)
            origin.remove(request.id)

            choice = None
            for candidate in states.values():
                insertion = candidate.best_insertion(request)
                if insertion is None:
                    continue
                value = insertion.cost + weight * insertion.lateness_delta_min
                key = (round(value, 6), candidate.vehicle_id, insertion.position)
                if choice is None or key < choice[0]:
                    choice = (key, candidate, insertion)

            if choice is None:
                origin.insert(request, min(position, len(origin.requests)))
                continue

            _, candidate, insertion = choice
            candidate.insert(request, insertion.position)
            after = sum(
                state.distance_cost + weight * state.evaluation.lateness_minutes
                for state in states.values()
            )
            if after < before - 1e-6:
                improved += 1
            else:
                candidate.remove(request.id)
                origin.insert(request, min(position, len(origin.requests)))
    return improved


class RegretAllocator:
    """Regret-ordered insertion followed by exact-cost local search."""

    name = "regret"
    label = "Deadline-regret insertion + local search"
    description = (
        "Commit the request that loses most if deferred (best vs second-best slot), "
        "prioritising VIPs and early deadlines, then improve the plan by relocating "
        "single requests while the priced route cost strictly drops."
    )
    kind = "heuristic"

    @staticmethod
    def _priced_cost(states, weight: float) -> float:
        """Distance cost plus priced lateness across all routes (the moving objective)."""
        return sum(
            state.distance_cost + weight * state.evaluation.lateness_minutes
            for state in states.values()
        )

    def _relocate_pass(self, states, weight: float) -> int:
        """Delegate to the shared relocate pass (kept as a method for clarity)."""
        return relocate_pass(states, weight)


    def allocate(
        self,
        scenario: Scenario,
        config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        regret_cap: float = 40.0,
        urgency_weight: float = 2.0,
        local_search: bool = True,
        local_search_passes: int = 2,
        **options
    ) -> AllocationPlan:
        started = time.perf_counter()
        states = {
            vehicle.id: RouteState(scenario, vehicle, config) for vehicle in scenario.vehicles
        }
        pending = {request.id: request for request in scenario.requests}
        reasons: dict = {}
        margins: dict = {}

        while pending:
            best_choice = None
            for request in pending.values():
                options_found: list = []
                for vehicle_id, state in states.items():
                    insertion = state.best_insertion(request)
                    if insertion is not None:
                        options_found.append((insertion.cost, vehicle_id, insertion))
                if not options_found:
                    continue
                options_found.sort(key=lambda item: (item[0], item[1]))
                best_cost, vehicle_id, insertion = options_found[0]
                second_cost = (
                    options_found[1][0]
                    if len(options_found) > 1
                    else best_cost + float(regret_cap)
                )
                regret = min(second_cost - best_cost, float(regret_cap))
                urgency = 1.0 / max(
                    0.25, (request.window_close_min - scenario.horizon_start_min) / 60.0
                )
                priority_weight = 1.0 + (request.priority - 1) * config.priority_step
                score = regret + float(urgency_weight) * urgency * priority_weight
                key = (-round(score, 6), request.window_close_min, -request.priority, request.id)
                if best_choice is None or key < best_choice[0]:
                    best_choice = (key, request, vehicle_id, insertion, regret)

            if best_choice is None:
                for request in pending.values():
                    reasons[request.id] = explain_unassigned(scenario, request, states)
                break

            _, request, vehicle_id, insertion, regret = best_choice
            states[vehicle_id].insert(request, insertion.position)
            margins[request.id] = regret
            del pending[request.id]

        moves = 0
        if local_search:
            for _ in range(max(0, int(local_search_passes))):
                accepted = self._relocate_pass(states, config.lateness_cost_per_min)
                moves += accepted
                if not accepted:
                    break

        runtime_ms = (time.perf_counter() - started) * 1000.0
        return build_plan(
            scenario,
            self.name,
            self.label,
            routes={vehicle_id: state.request_ids() for vehicle_id, state in states.items()},
            unassigned_reasons=reasons,
            runtime_ms=runtime_ms,
            config=config,
            reason_builder=_reason_builder(margins),
            notes=[
                "regret cap %.0f cost units, urgency weight %.2f" % (regret_cap, urgency_weight),
                "local search: %s (%d accepted relocate moves)"
                % ("on" if local_search else "off", moves),
            ],
        )
