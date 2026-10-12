"""Pair features for the learned (RL) policy.

The vector is split in two blocks:

* *static* features (first 12) depend only on the vehicle's current route and the
  request, so they survive between decisions and are computed once per candidate;
* *situational* features (last 5) describe the request's standing in the whole
  book right now - how many vehicles could still take it, how much worse the
  runner-up slot is (regret), whether this vehicle is the cheapest option and how
  urgent the deadline is. They are assembled vectorised at decision time.

The situational block is what lets a linear policy express "commit the request
that would lose the most if deferred", the rule the regret heuristic uses; it is
computed from data the other strategies also see, so no oracle information leaks
in. FEATURE_NAMES is the contract shared with the artifact: training and
inference refuse to run against a mismatched weight vector.
"""

from __future__ import annotations

import numpy as np

from ..domain import DeliveryRequest, Vehicle
from ..feasibility import Insertion, RouteEvaluation
from ..geo import haversine_km

FEATURE_NAMES = (
    "bias",
    "detour_km",
    "leg_in_km",
    "deadline_slack",
    "lateness",
    "capacity_headroom",
    "volume_headroom",
    "shift_headroom",
    "priority",
    "utilization",
    "position_ratio",
    "depot_distance",
    "scarcity",
    "request_regret",
    "request_best_detour",
    "is_best_vehicle",
    "urgency",
)
STATIC_FEATURE_NAMES = FEATURE_NAMES[:12]
STATIC_DIM = len(STATIC_FEATURE_NAMES)
FEATURE_DIM = len(FEATURE_NAMES)

DETOUR_SCALE_KM = 5.0
LEG_SCALE_KM = 10.0
SLACK_SCALE_MIN = 120.0
LATENESS_SCALE_MIN = 30.0
SHIFT_SCALE_MIN = 120.0
DEPOT_SCALE_KM = 15.0
REGRET_CAP = 40.0
REGRET_SCALE = 10.0
BEST_COST_SCALE = 5.0


def _clip(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def build_static_features(
    vehicle: Vehicle, request: DeliveryRequest, insertion: Insertion, base: RouteEvaluation
) -> list:
    """The per-candidate block: geometry, capacity, time and priority."""
    peak_kg = base.peak_load_kg + request.demand_kg
    peak_m3 = base.peak_load_m3 + request.volume_m3
    slack = request.window_close_min - insertion.arrival_min
    busy_minutes = insertion.finish_min - vehicle.shift_start_min
    shift = vehicle.shift_minutes
    return [
        1.0,
        insertion.delta_km / DETOUR_SCALE_KM,
        insertion.leg_in_km / LEG_SCALE_KM,
        _clip(slack / SLACK_SCALE_MIN, -1.0, 2.0),
        insertion.lateness_delta_min / LATENESS_SCALE_MIN,
        1.0 - peak_kg / vehicle.capacity_kg,
        1.0 - peak_m3 / vehicle.capacity_m3,
        _clip((vehicle.shift_end_min - insertion.finish_min) / SHIFT_SCALE_MIN, -0.5, 2.0) / 2.0,
        (1.0 + (request.priority - 1) * 0.5) / 3.0,
        _clip(busy_minutes / shift, 0.0, 1.5) if shift else 0.0,
        insertion.position / max(1, len(base.stops) + 1),
        haversine_km(vehicle.depot, request.location) / DEPOT_SCALE_KM,
    ]


def urgency_value(request: DeliveryRequest, horizon_start_min: float) -> float:
    """How soon this deadline falls relative to the planning horizon (0.5..4)."""
    return 1.0 / max(0.25, (request.window_close_min - horizon_start_min) / 60.0) / 4.0


def situational_columns(
    request_index: np.ndarray,
    costs: np.ndarray,
    counts: np.ndarray,
    regret: np.ndarray,
    best_cost: np.ndarray,
    urgency: np.ndarray,
    vehicle_count: int,
) -> np.ndarray:
    """Assemble the situational block for a batch of candidate actions.

    ``counts``/``regret``/``best_cost``/``urgency`` are indexed by scenario request
    index, while ``request_index`` and ``costs`` are per action.
    """
    best = best_cost[request_index]
    is_best = (np.abs(costs - best) <= 1e-9).astype(float)
    return np.column_stack(
        [
            counts[request_index] / max(1, vehicle_count),
            regret[request_index] / REGRET_SCALE,
            best / BEST_COST_SCALE,
            is_best,
            urgency[request_index],
        ]
    )
