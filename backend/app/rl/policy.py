"""A linear softmax policy over (request, vehicle) pair features.

Deliberately small and dependency-light: scores are ``w . phi`` and the action
distribution is the softmax over the *feasible* actions only. The learned policy
is 13 floats, so the artifact is reviewable in a diff, inference is one dot
product per candidate action, and there is no model server, no GPU and no
network access at any point.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

from .features import FEATURE_DIM, FEATURE_NAMES

ARTIFACT_VERSION = 1
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_POLICY_PATH = os.path.join(BACKEND_ROOT, "artifacts", "rl_policy.json")
POLICY_PATH_ENV = "RL_POLICY_PATH"


class PolicyArtifactError(RuntimeError):
    """Raised when the committed policy artifact is missing or malformed."""


def default_policy_path() -> str:
    return os.environ.get(POLICY_PATH_ENV) or DEFAULT_POLICY_PATH


class LinearSoftmaxPolicy:
    """``pi(a | s) = softmax_a(w . phi(s, a))`` over feasible actions."""

    def __init__(
        self,
        weights: Sequence[float],
        feature_names: Sequence[str] = FEATURE_NAMES,
        metadata: Optional[dict] = None,
    ) -> None:
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (FEATURE_DIM,):
            raise PolicyArtifactError(
                "policy has %d weights but %d features are expected"
                % (self.weights.size, FEATURE_DIM)
            )
        self.feature_names = tuple(feature_names)
        if self.feature_names != tuple(FEATURE_NAMES):
            raise PolicyArtifactError("policy feature names do not match this build")
        self.metadata = dict(metadata or {})

    def scores(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=float) @ self.weights

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Softmax over the supplied (already feasible) action features."""
        scores = self.scores(features)
        shifted = scores - float(np.max(scores))
        exponentials = np.exp(shifted)
        return exponentials / float(np.sum(exponentials))

    def best_index(self, features: np.ndarray) -> int:
        """Deterministic greedy action; ties break to the lowest index."""
        return int(np.argmax(self.scores(features)))

    def weight_table(self) -> list:
        """Weights sorted by absolute magnitude - rendered in the UI."""
        rows = [
            {
                "feature": name,
                "weight": round(float(weight), 6),
                "abs_weight": round(abs(float(weight)), 6),
            }
            for name, weight in zip(self.feature_names, self.weights)
        ]
        rows.sort(key=lambda row: -row["abs_weight"])
        return rows

    def to_dict(self) -> dict:
        return {
            "version": ARTIFACT_VERSION,
            "algorithm": "reinforce-linear-softmax",
            "feature_names": list(self.feature_names),
            "weights": [round(float(weight), 8) for weight in self.weights],
            "metadata": self.metadata,
        }

    def save(self, path: Optional[str] = None) -> str:
        target = path or default_policy_path()
        directory = os.path.dirname(os.path.abspath(target))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1, sort_keys=True)
            handle.write("\n")
        return target

    @classmethod
    def load(cls, path: Optional[str] = None) -> "LinearSoftmaxPolicy":
        target = path or default_policy_path()
        if not os.path.exists(target):
            raise PolicyArtifactError(
                "no policy artifact at %s - run `python -m app.rl.train` to create it "
                "(the image ships one, so a missing file usually means a bad mount)" % target
            )
        try:
            with open(target, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError) as error:
            raise PolicyArtifactError("cannot read policy artifact %s: %s" % (target, error))
        version = payload.get("version")
        if version != ARTIFACT_VERSION:
            raise PolicyArtifactError(
                "policy artifact version %r is not supported (expected %d)" % (version, ARTIFACT_VERSION)
            )
        return cls(
            weights=payload["weights"],
            feature_names=payload.get("feature_names", FEATURE_NAMES),
            metadata=payload.get("metadata") or {},
        )
