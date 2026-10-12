"""FastAPI surface for the allocation engine.

Endpoints (all under ``/api``):

* ``GET  /api/health``           - liveness plus what this image can do;
* ``GET  /api/algorithms``       - strategy catalogue with availability;
* ``GET  /api/scenario/presets`` - preset catalogue for the UI;
* ``POST /api/scenario``         - build and inspect a scenario (no allocation);
* ``POST /api/allocate``         - run one strategy, get an explained plan;
* ``POST /api/compare``          - run several strategies on one scenario and rank them;
* ``GET  /api/rl/policy``        - committed policy weights, training curve, holdout audit.

The built single-page app is mounted at ``/`` when ``frontend/dist`` exists, so a
single container serves API and UI on one port. A missing RL artifact disables
only the learned strategy - every other endpoint keeps working, and the reason is
reported rather than hidden.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .algorithms import UNAVAILABLE, algorithm_info, algorithm_names, default_algorithm, get_allocator
from .domain import Scenario
from .metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from .scenario import (
    CITY_BOXES,
    GENERATOR_PARAMETERS,
    PRESETS,
    generate_scenario,
    preset_info,
    preset_names,
    scenario_from_spec,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIST = os.environ.get("FRONTEND_DIST") or os.path.join(REPO_ROOT, "frontend", "dist")

MAX_REQUESTS = 1000
MAX_VEHICLES = 60

ALLOWED_OPTIONS: Dict[str, set] = {
    "greedy": {"ordering", "lateness_weight"},
    "hungarian": {"max_rounds", "lateness_weight"},
    "regret": {"regret_cap", "urgency_weight", "local_search", "local_search_passes"},
    "exact": set(),
    "rl": {"local_search", "local_search_passes"},
}


class ObjectiveConfigModel(BaseModel):
    """Optional overrides for the composite objective's weights."""

    lateness_cost_per_min: Optional[float] = Field(default=None, ge=0.0)
    unassigned_base_cost: Optional[float] = Field(default=None, ge=0.0)
    priority_step: Optional[float] = Field(default=None, ge=0.0)
    makespan_cost_per_min: Optional[float] = Field(default=None, ge=0.0)

    def to_config(self) -> ObjectiveConfig:
        base = DEFAULT_OBJECTIVE
        return ObjectiveConfig(
            lateness_cost_per_min=(
                base.lateness_cost_per_min
                if self.lateness_cost_per_min is None
                else self.lateness_cost_per_min
            ),
            unassigned_base_cost=(
                base.unassigned_base_cost if self.unassigned_base_cost is None else self.unassigned_base_cost
            ),
            priority_step=base.priority_step if self.priority_step is None else self.priority_step,
            makespan_cost_per_min=(
                base.makespan_cost_per_min
                if self.makespan_cost_per_min is None
                else self.makespan_cost_per_min
            ),
        )


class ScenarioRequest(BaseModel):
    """Either generate a scenario from knobs, or send a full scenario payload."""

    preset: Optional[str] = Field(default=None, description="small | medium | large | tight | stress")
    seed: Optional[int] = Field(default=None, ge=0)
    city: Optional[str] = None
    vehicles: Optional[int] = Field(default=None, ge=0, le=MAX_VEHICLES)
    requests: Optional[int] = Field(default=None, ge=0, le=MAX_REQUESTS)
    capacity_ratio: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    window_scale: Optional[float] = Field(default=None, gt=0.0, le=3.0)
    grace_minutes: Optional[float] = Field(default=None, ge=0.0, le=240.0)
    clustered: Optional[bool] = None
    scenario: Optional[dict] = Field(default=None, description="full scenario (see POST /api/scenario)")

    def build(self) -> Scenario:
        try:
            if self.scenario is not None:
                return scenario_from_spec(self.scenario)
            payload = {
                "preset": self.preset or "medium",
                "seed": 42 if self.seed is None else self.seed,
                "city": self.city or "bengaluru",
            }
            for key in ("vehicles", "requests", "capacity_ratio", "window_scale", "grace_minutes", "clustered"):
                value = getattr(self, key)
                if value is not None:
                    payload[key] = value
            return scenario_from_spec(payload)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (KeyError, TypeError) as error:
            raise HTTPException(
                status_code=400, detail="malformed scenario payload: %s" % error
            ) from error


def _check_options(algorithm: str, options: Optional[dict]) -> dict:
    options = dict(options or {})
    allowed = ALLOWED_OPTIONS.get(algorithm, set())
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail="unsupported option(s) for %s: %s (allowed: %s)"
            % (algorithm, ", ".join(unknown), ", ".join(sorted(allowed)) or "none"),
        )
    return options


def _run(algorithm: str, scenario: Scenario, config: ObjectiveConfig, options: Optional[dict]) -> dict:
    try:
        allocator = get_allocator(algorithm)
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    options = _check_options(algorithm, options)
    try:
        plan = allocator.allocate(scenario, config, **options)
    except ValueError as error:  # e.g. the exhaustive solver's instance guard
        raise HTTPException(status_code=400, detail=str(error)) from error
    return plan.to_dict()


class AllocateRequest(BaseModel):
    """One strategy, one scenario, one explained plan."""

    scenario: ScenarioRequest = Field(default_factory=ScenarioRequest)
    algorithm: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    objective: ObjectiveConfigModel = Field(default_factory=ObjectiveConfigModel)
    include_scenario: bool = True


class CompareRequest(BaseModel):
    """Several strategies on the same scenario, ranked on the same objective."""

    scenario: ScenarioRequest = Field(default_factory=ScenarioRequest)
    algorithms: Optional[List[str]] = None
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    objective: ObjectiveConfigModel = Field(default_factory=ObjectiveConfigModel)
    include_scenario: bool = True


COMPARED_METRICS = (
    "objective",
    "coverage_pct",
    "priority_coverage_pct",
    "requests_assigned",
    "requests_unassigned",
    "distance_km",
    "late_requests",
    "lateness_minutes",
    "max_lateness_min",
    "on_time_pct",
    "vehicles_used",
    "avg_utilization_pct",
    "avg_load_pct",
    "makespan_min",
    "workload_stdev_minutes",
    "runtime_ms",
)


def _ranking_row(plan: dict, rank: int) -> dict:
    metrics = plan["metrics"]
    row = {"rank": rank, "algorithm": plan["algorithm"], "label": plan["label"]}
    for name in COMPARED_METRICS:
        row[name] = round(plan["runtime_ms"], 2) if name == "runtime_ms" else metrics[name]
    row["valid"] = metrics["valid"]
    return row


app = FastAPI(
    title="Delivery-fleet allocation engine",
    version=__version__,
    description=(
        "Allocate a delivery book to a fleet with four strategies (greedy cheapest "
        "insertion, Hungarian batch matching, deadline-regret insertion with local "
        "search, and a REINFORCE-trained linear policy), plus an exhaustive reference "
        "for tiny instances. Every plan is explained stop by stop and scored on one "
        "documented composite objective."
    ),
)


@app.get("/api/health")
def health() -> dict:
    """Liveness plus everything this image can actually do."""
    return {
        "status": "ok",
        "version": __version__,
        "algorithms": algorithm_names(),
        "unavailable": dict(UNAVAILABLE),
        "default_algorithm": default_algorithm(),
        "presets": preset_names(),
        "cities": sorted(CITY_BOXES),
        "objective_defaults": DEFAULT_OBJECTIVE.to_dict(),
        "frontend_built": os.path.isdir(FRONTEND_DIST),
        "data_source": "seeded synthetic generator (offline, no external calls)",
    }


@app.get("/api/algorithms")
def algorithms() -> dict:
    """Strategy catalogue, including why a strategy is unavailable."""
    return {"algorithms": algorithm_info(), "default": default_algorithm()}


@app.get("/api/scenario/presets")
def scenario_presets() -> dict:
    return {
        "presets": preset_info(),
        "cities": {name: list(box) for name, box in CITY_BOXES.items()},
        "generator_parameters": list(GENERATOR_PARAMETERS),
        "objective_defaults": DEFAULT_OBJECTIVE.to_dict(),
    }


@app.post("/api/scenario")
def build_scenario(request: ScenarioRequest) -> dict:
    """Build a scenario and return it in full (nothing is allocated yet)."""
    scenario = request.build()
    return {"summary": scenario.summary(), "scenario": scenario.to_dict()}


@app.post("/api/allocate")
def allocate(request: AllocateRequest) -> dict:
    """Run one strategy and return the plan, its metrics and its explanations."""
    scenario = request.scenario.build()
    algorithm = request.algorithm or default_algorithm()
    plan = _run(algorithm, scenario, request.objective.to_config(), request.options)
    payload = {"summary": scenario.summary(), "algorithm": algorithm, "plan": plan}
    if request.include_scenario:
        payload["scenario"] = scenario.to_dict()
    return payload


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    """Run several strategies on one scenario and rank them on the same objective.

    The exhaustive reference is only included automatically when the instance is
    actually small enough to enumerate; asking for it explicitly on a bigger
    scenario returns a clear 400 instead of stalling.
    """
    scenario = request.scenario.build()
    if request.algorithms:
        names = list(request.algorithms)
    else:
        names = [name for name in algorithm_names() if name != "exact"]
        if len(scenario.requests) <= 6 and len(scenario.vehicles) <= 3:
            names.append("exact")
    if not names:
        raise HTTPException(status_code=400, detail="no strategies are available in this image")
    config = request.objective.to_config()
    results = [_run(name, scenario, config, request.options.get(name)) for name in names]
    ranked = sorted(results, key=lambda plan: (plan["metrics"]["objective"], plan["algorithm"]))
    payload = {
        "summary": scenario.summary(),
        "results": results,
        "rankings": [_ranking_row(plan, index + 1) for index, plan in enumerate(ranked)],
        "winner": ranked[0]["algorithm"],
        "metrics_compared": list(COMPARED_METRICS),
        "objective": config.to_dict(),
    }
    if request.include_scenario:
        payload["scenario"] = scenario.to_dict()
    return payload


@app.get("/api/rl/policy")
def rl_policy(include_curve: bool = True) -> dict:
    """Committed policy weights plus its training and holdout audit."""
    try:
        allocator = get_allocator("rl")
    except KeyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    summary = allocator.policy_summary()
    if not include_curve:
        summary.pop("curve", None)
        summary.pop("curve_smoothed", None)
    return summary


def _api_index_html() -> str:
    endpoints = (
        ("GET /api/health", "service status and capabilities"),
        ("GET /api/algorithms", "strategy catalogue and availability"),
        ("GET /api/scenario/presets", "scenario presets, cities and objective defaults"),
        ("POST /api/scenario", "build a scenario"),
        ("POST /api/allocate", "allocate with one strategy"),
        ("POST /api/compare", "compare every strategy on one scenario"),
        ("GET /api/rl/policy", "committed RL policy, weights and training curve"),
        ("GET /docs", "interactive OpenAPI documentation"),
    )
    rows = "".join(
        "<li><code>%s</code> &mdash; %s</li>" % (path, description) for path, description in endpoints
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Delivery-fleet allocation engine</title></head>"
        "<body style='font-family:system-ui;margin:2.5rem;max-width:44rem;line-height:1.5'>"
        "<h1>Delivery-fleet allocation engine</h1>"
        "<p>This image has no built frontend (<code>%s</code> is missing), so here is the "
        "API instead.</p><ul>%s</ul></body></html>" % (FRONTEND_DIST, rows)
    )


if os.path.isdir(FRONTEND_DIST):
    # One port for everything: the SPA is served from the same origin as the API.
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
else:

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _api_index_html()


def main() -> None:
    """``python -m app.main`` boots the API (used by the container entrypoint)."""
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()

