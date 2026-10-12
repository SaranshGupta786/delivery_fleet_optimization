"""The allocation MDP used to train - and to run - the learned policy.

An episode builds a plan action by action:

* state  - the partial plan (one route per vehicle) plus the unplaced requests;
* action - one (request, vehicle) pair from the feasible set; the request is
           inserted at that vehicle's cheapest feasible position;
* reward - the negative *incremental* priced cost of the insertion, and at the
           end the negative cost of whatever stayed unassigned plus the makespan
           term.

Because the step reward is exactly the true incremental cost and the terminal
reward is exactly the leftover unassigned + makespan cost, the undiscounted
episode return equals ``-objective`` of the finished plan. Training therefore
optimises the same number the benchmark reports: there is no reward/metric gap.

Efficiency: a (request, vehicle) pair's insertion only changes when that vehicle's
route changes, so candidates are cached per vehicle and only the affected column
is repaired after a step. Static features are cached with each candidate, and the
situational block is assembled vectorised (numpy) at decision time, so an episode
costs O(n^2 * route length) instead of re-scoring every pair from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..algorithms.base import RouteState
from ..domain import DeliveryRequest, Scenario
from ..feasibility import Insertion
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .features import (
    FEATURE_DIM,
    REGRET_CAP,
    build_static_features,
    situational_columns,
    urgency_value,
)


@dataclass
class Action:
    """One candidate decision: place ``request`` on ``vehicle``."""

    request_id: str
    request_index: int
    vehicle_id: str
    insertion: Insertion
    static: list
    reward: float

    @property
    def detour_cost(self) -> float:
        return self.insertion.cost


class AllocationEnv:
    """Sequential construction MDP over a single scenario."""

    def __init__(self, scenario: Scenario, config: ObjectiveConfig = DEFAULT_OBJECTIVE) -> None:
        self.scenario = scenario
        self.config = config
        self.states: Dict[str, RouteState] = {
            vehicle.id: RouteState(scenario, vehicle, config) for vehicle in scenario.vehicles
        }
        self.pending: Dict[str, DeliveryRequest] = {
            request.id: request for request in scenario.requests
        }
        total = len(scenario.requests)
        self._request_pos: Dict[str, int] = {
            request.id: index for index, request in enumerate(scenario.requests)
        }
        self._counts = np.zeros(total, dtype=float)
        self._regret = np.zeros(total, dtype=float)
        self._best_cost = np.zeros(total, dtype=float)
        self._urgency = np.asarray(
            [
                urgency_value(request, scenario.horizon_start_min)
                for request in scenario.requests
            ],
            dtype=float,
        )
        self._feasible: Dict[Tuple[str, str], bool] = {}
        self._by_vehicle: Dict[str, List[Action]] = {}
        for vehicle_id in self.states:
            self._by_vehicle[vehicle_id] = self._enumerate_vehicle(vehicle_id)
            for candidate in self._by_vehicle[vehicle_id]:
                self._feasible[(candidate.request_id, vehicle_id)] = True
                self._counts[candidate.request_index] += 1.0
        self._refresh_request_stats()
        self.steps = 0
        self.total_reward = 0.0

    def _enumerate_vehicle(self, vehicle_id: str) -> List[Action]:
        """Best feasible insertion for every still-pending request on one vehicle."""
        state = self.states[vehicle_id]
        vehicle = state.vehicle
        actions: List[Action] = []
        for request in self.pending.values():
            insertion = state.best_insertion(request)
            if insertion is None:
                continue
            actions.append(
                Action(
                    request_id=request.id,
                    request_index=self._request_pos[request.id],
                    vehicle_id=vehicle_id,
                    insertion=insertion,
                    static=build_static_features(vehicle, request, insertion, state.evaluation),
                    reward=-(
                        insertion.cost
                        + self.config.lateness_cost_per_min * insertion.lateness_delta_min
                    ),
                )
            )
        return actions


    # -- candidate bookkeeping ---------------------------------------------
    def feasible_actions(self) -> List[Action]:
        """All currently feasible (request, vehicle) pairs, in deterministic order."""
        actions: List[Action] = []
        for candidates in self._by_vehicle.values():
            actions.extend(candidates)
        return actions

    def snapshot(self) -> Tuple[List[Action], np.ndarray]:
        """Current feasible actions plus their feature matrix (same order)."""
        actions = self.feasible_actions()
        if not actions:
            return actions, np.zeros((0, FEATURE_DIM), dtype=float)
        static = np.asarray([action.static for action in actions], dtype=float)
        request_index = np.fromiter(
            (action.request_index for action in actions), dtype=np.int64, count=len(actions)
        )
        costs = np.fromiter(
            (action.insertion.cost for action in actions), dtype=float, count=len(actions)
        )
        situational = situational_columns(
            request_index,
            costs,
            self._counts,
            self._regret,
            self._best_cost,
            self._urgency,
            len(self.states),
        )
        return actions, np.hstack([static, situational])

    def _refresh_request_stats(self) -> None:
        """Best and runner-up insertion cost per pending request -> regret feature.

        This mirrors the regret heuristic's definition exactly (same cap, ties
        ignored) so the learned policy *can* represent that rule if it pays off.
        """
        total = self._urgency.size
        actions = self.feasible_actions()
        best = np.full(total, np.inf, dtype=float)
        second = np.full(total, np.inf, dtype=float)
        if actions:
            request_index = np.fromiter(
                (action.request_index for action in actions), dtype=np.int64, count=len(actions)
            )
            costs = np.fromiter(
                (action.insertion.cost for action in actions), dtype=float, count=len(actions)
            )
            np.minimum.at(best, request_index, costs)
            strictly_worse = np.where(costs <= best[request_index] + 1e-9, np.inf, costs)
            np.minimum.at(second, request_index, strictly_worse)
        with np.errstate(invalid="ignore"):
            regret = np.clip(
                np.nan_to_num(
                    np.where(
                        np.isfinite(second) & np.isfinite(best),
                        second - best,
                        REGRET_CAP,
                    ),
                    nan=0.0,
                    posinf=REGRET_CAP,
                ),
                0.0,
                REGRET_CAP,
            )
        self._best_cost = np.where(np.isfinite(best), best, 0.0)

    def step(self, action: Action) -> float:
        """Apply an action and return its reward (negative incremental priced cost)."""
        request = self.pending[action.request_id]
        vehicle_id = action.vehicle_id
        previous = {
            request_id: self._feasible.get((request_id, vehicle_id), False)
            for request_id in self.pending
        }

        self.states[vehicle_id].insert(request, action.insertion.position)
        del self.pending[request.id]
        self.steps += 1
        self.total_reward += action.reward
        self._feasible.pop((request.id, vehicle_id), None)
        self._counts[action.request_index] = 0.0

        for other_id, candidates in list(self._by_vehicle.items()):
            self._by_vehicle[other_id] = [
                candidate for candidate in candidates if candidate.request_id != request.id
            ]

        self._by_vehicle[vehicle_id] = self._enumerate_vehicle(vehicle_id)
        refreshed = {candidate.request_id for candidate in self._by_vehicle[vehicle_id]}
        for request_id in self.pending:
            was_feasible = previous.get(request_id, False)
            is_feasible = request_id in refreshed
            if was_feasible != is_feasible:
                index = self._request_pos[request_id]
                self._counts[index] += 1.0 if is_feasible else -1.0
            self._feasible[(request_id, vehicle_id)] = is_feasible

        self._refresh_request_stats()
        return action.reward


    # -- scoring ------------------------------------------------------------
    def unassigned_cost(self) -> float:
        return sum(self.config.unassigned_cost(request) for request in self.pending.values())

    def makespan_cost(self) -> float:
        finishes = [
            state.evaluation.finish_min for state in self.states.values() if state.stop_count
        ]
        makespan = (max(finishes) - self.scenario.horizon_start_min) if finishes else 0.0
        return makespan * self.config.makespan_cost_per_min

    def terminal_reward(self) -> float:
        """End-of-episode reward: refused requests plus the makespan term."""
        return -(self.unassigned_cost() + self.makespan_cost())

    def objective(self) -> float:
        """Composite objective of the plan built so far (final once stopped)."""
        distance_cost = sum(state.distance_cost for state in self.states.values())
        lateness_cost = (
            sum(state.evaluation.lateness_minutes for state in self.states.values())
            * self.config.lateness_cost_per_min
        )
        return distance_cost + lateness_cost + self.unassigned_cost() + self.makespan_cost()

    def episode_return(self) -> float:
        """Equals ``-objective()``: the property the whole training loop relies on."""
        return self.total_reward + self.terminal_reward()

    def plan_routes(self) -> Dict[str, tuple]:
        return {vehicle_id: state.request_ids() for vehicle_id, state in self.states.items()}

    def stopped(self) -> bool:
        return not any(self._by_vehicle.values())


def rollout(env: AllocationEnv, policy=None, sample: bool = False, rng=None):
    """Run one episode to completion; returns ``(actions, return)``.

    With a policy, actions are ``argmax`` (deterministic) unless ``sample=True``.
    Called on the *trained* policy this is exactly what the API serves.
    """
    chosen: List[Action] = []
    while True:
        actions, matrix = env.snapshot()
        if not actions:
            break
        if policy is None:
            index = 0
        elif sample:
            index = int(rng.choice(len(actions), p=policy.probabilities(matrix)))
        else:
            index = policy.best_index(matrix)
        action = actions[index]
        env.step(action)
        chosen.append(action)
    return chosen, env.episode_return()
