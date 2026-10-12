"""Strategy registry - the single lookup point for the API, CLI and tests.

Classical strategies are always available. The learned strategy is loaded lazily
and reported as *unavailable* (with the reason) when its artifact is missing or
corrupt, so a bad policy file can never take the rest of the engine down.
"""

from __future__ import annotations

from typing import Dict, List

from .base import AllocationPlan, Allocator
from .bruteforce import BruteForceAllocator
from .greedy import GreedyAllocator
from .heuristic import RegretAllocator
from .hungarian import HungarianAllocator

RL_NAME = "rl"
RL_FALLBACK_LABEL = "Learned policy (REINFORCE)"
FALLBACK_ALGORITHM = "regret"

ALGORITHMS: Dict[str, Allocator] = {}
UNAVAILABLE: Dict[str, str] = {}
_LOADED = False


def _register(allocator: Allocator) -> None:
    ALGORITHMS[allocator.name] = allocator


def _ensure_loaded() -> None:
    """Populate the registry once, on first use (keeps import order irrelevant)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for allocator in (
        GreedyAllocator(),
        HungarianAllocator(),
        RegretAllocator(),
        BruteForceAllocator(),
    ):
        _register(allocator)
    try:
        from ..rl.algorithm import RLAllocator

        _register(RLAllocator())
    except Exception as error:  # pragma: no cover - depends on the artifact
        UNAVAILABLE[RL_NAME] = "%s: %s" % (type(error).__name__, error)


def algorithm_names() -> List[str]:
    _ensure_loaded()
    return list(ALGORITHMS.keys())


def default_algorithm() -> str:
    _ensure_loaded()
    return RL_NAME if RL_NAME in ALGORITHMS else FALLBACK_ALGORITHM


def get_allocator(name: str) -> Allocator:
    _ensure_loaded()
    if name in ALGORITHMS:
        return ALGORITHMS[name]
    if name in UNAVAILABLE:
        raise KeyError("strategy %r is unavailable: %s" % (name, UNAVAILABLE[name]))
    raise KeyError("unknown strategy %r (available: %s)" % (name, ", ".join(ALGORITHMS)))


def algorithm_info() -> List[dict]:
    """Catalogue for ``GET /api/algorithms`` and the UI's strategy list."""
    _ensure_loaded()
    info = [
        {
            "name": name,
            "label": allocator.label,
            "description": allocator.description,
            "kind": allocator.kind,
            "available": True,
            "is_default": name == default_algorithm(),
        }
        for name, allocator in ALGORITHMS.items()
    ]
    for name, reason in UNAVAILABLE.items():
        info.append(
            {
                "name": name,
                "label": RL_FALLBACK_LABEL,
                "description": "Unavailable in this deployment: %s" % reason,
                "kind": "learned",
                "available": False,
                "is_default": False,
            }
        )
    return info
