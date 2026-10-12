"""The learned strategy: run the committed policy inside the live engine.

Inference is a deterministic argmax rollout over the feasible actions - no
sampling, no training, no randomness - so the same scenario always yields the
same plan. The artifact is loaded once and cached; a missing or corrupt file
disables this strategy with a clear message instead of breaking the others.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..algorithms.base import (
    AllocationPlan,
    RouteState,
    build_plan,
    default_reason,
    explain_unassigned,
)
from ..domain import Scenario
from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .env import AllocationEnv
from .features import FEATURE_DIM
from .policy import LinearSoftmaxPolicy, default_policy_path


def _policy_reason_builder(scores):
    """Append the policy's own score and margin to the standard explanation."""

    def builder(scenario, vehicle, request, stop, evaluation, sequence):
        text = default_reason(scenario, vehicle, request, stop, evaluation, sequence)
        entry = scores.get(request.id)
        if entry is not None:
            score, margin = entry
            if margin is None:
                text += " Policy score %.2f (the only feasible action)." % score
            else:
                text += (
                    " Policy score %.2f, %.2f ahead of the runner-up action." % (score, margin)
                )
        return text

    return builder


class RLAllocator:
    """Greedy rollout of the trained linear softmax policy."""

    name = "rl"
    label = "Learned policy (REINFORCE)"
    description = (
        "Deterministic argmax rollout of a linear softmax policy trained with "
        "REINFORCE across sampled scenarios. Weights ship as a small JSON artifact; "
        "nothing is trained or downloaded at runtime."
    )
    kind = "learned"

    def __init__(self, policy_path: str = None) -> None:
        self.policy_path = policy_path or default_policy_path()
        self.policy = LinearSoftmaxPolicy.load(self.policy_path)

    def allocate(
        self,
        scenario: Scenario,
        config: ObjectiveConfig = DEFAULT_OBJECTIVE,
        local_search: bool = False,
        local_search_passes: int = 1,
        **options
    ) -> AllocationPlan:
        started = time.perf_counter()
        env = AllocationEnv(scenario, config)
        scores: dict = {}

        while True:
            actions, matrix = env.snapshot()
            if not actions:
                break
            vector = self.policy.scores(matrix)
            index = int(np.argmax(vector))
            margin = (
                float(vector[index] - np.max(np.delete(vector, index)))
                if len(vector) > 1
                else None
            )
            action = actions[index]
            scores[action.request_id] = (float(vector[index]), margin)
            env.step(action)

        moves = 0
        if local_search:
            # Optional polish: the same generic relocate pass the regret heuristic
            # uses. Kept opt-in and reported in the notes, so "raw policy" and
            # "policy + polish" can be compared without ambiguity.
            from ..algorithms.heuristic import relocate_pass

            for _ in range(max(0, int(local_search_passes))):
                accepted = relocate_pass(env.states, config.lateness_cost_per_min)
                moves += accepted
                if not accepted:
                    break

        runtime_ms = (time.perf_counter() - started) * 1000.0
        states = env.states
        reasons = {
            request_id: explain_unassigned(scenario, scenario.request(request_id), states)
            for request_id in env.pending
        }
        metadata = self.policy.metadata
        return build_plan(
            scenario,
            self.name,
            self.label,
            routes=env.plan_routes(),
            unassigned_reasons=reasons,
            runtime_ms=runtime_ms,
            config=config,
            reason_builder=None if local_search else _policy_reason_builder(scores),
            notes=[
                "policy artifact: %s" % os.path.basename(self.policy_path),
                "trained for %s episodes (seed %s)"
                % (
                    metadata.get("train_episodes", "?"),
                    metadata.get("train_config", {}).get("seed", "?"),
                ),
                "deterministic argmax over %d pair features; no training at runtime" % FEATURE_DIM,
                "polish: %d accepted relocate moves" % moves
                if local_search
                else "polish: off (raw policy output)",
            ],
        )

    def policy_summary(self) -> dict:
        """Weight table plus training provenance, for the UI's RL panel."""
        metadata = dict(self.policy.metadata)
        curve = metadata.pop("curve", [])
        smoothed = metadata.pop("curve_smoothed", [])
        return {
            "name": self.name,
            "label": self.label,
            "path": self.policy_path,
            "feature_names": list(self.policy.feature_names),
            "weights": self.policy.weight_table(),
            "curve": curve,
            "curve_smoothed": smoothed,
            "metadata": metadata,
        }
