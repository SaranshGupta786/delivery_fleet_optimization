"""API contract tests: happy paths, validation errors and the SPA mount."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import FRONTEND_DIST, app
from app.rl.policy import default_policy_path

client = TestClient(app)


def test_health_reports_capabilities():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert {"greedy", "hungarian", "regret", "exact"} <= set(body["algorithms"])
    assert body["default_algorithm"] in body["algorithms"]
    assert "small" in body["presets"]
    assert body["objective_defaults"]["lateness_cost_per_min"] > 0
    assert body["frontend_built"] is os.path.isdir(FRONTEND_DIST)


def test_algorithms_catalogue_explains_unavailability():
    body = client.get("/api/algorithms").json()
    names = [entry["name"] for entry in body["algorithms"]]
    assert names[:4] == ["greedy", "hungarian", "regret", "exact"]
    for entry in body["algorithms"]:
        assert entry["label"] and entry["description"] and entry["kind"]
        if not entry["available"]:
            assert "navailable" in entry["description"]


def test_presets_endpoint():
    body = client.get("/api/scenario/presets").json()
    assert [preset["name"] for preset in body["presets"]] == [
        "small",
        "medium",
        "large",
        "tight",
        "stress",
    ]
    assert "bengaluru" in body["cities"]
    assert "seed" in body["generator_parameters"]


def test_scenario_endpoint_returns_a_full_book():
    response = client.post("/api/scenario", json={"preset": "small", "seed": 7})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["requests"] == 40
    assert len(body["scenario"]["vehicles"]) == 6
    assert len(body["scenario"]["requests"]) == 40
    assert body["scenario"]["metadata"]["preset"] == "small"


def test_scenario_endpoint_accepts_an_explicit_payload():
    scenario = client.post("/api/scenario", json={"preset": "small", "seed": 3}).json()["scenario"]
    scenario["seed"] = 99
    response = client.post("/api/scenario", json={"scenario": scenario})
    assert response.status_code == 200
    assert response.json()["scenario"]["seed"] == 99


def test_allocate_returns_an_explained_plan():
    response = client.post(
        "/api/allocate",
        json={"scenario": {"preset": "small", "seed": 7}, "algorithm": "regret"},
    )
    assert response.status_code == 200
    body = response.json()
    metrics = body["plan"]["metrics"]
    assert metrics["valid"]
    assert metrics["requests_assigned"] + metrics["requests_unassigned"] == 40
    assert body["plan"]["assignments"]
    assert all(assignment["reason"] for assignment in body["plan"]["assignments"])
    assert body["scenario"]["name"] == "small-seed7"


@pytest.mark.parametrize("algorithm", ["greedy", "hungarian", "regret"])
def test_every_classical_strategy_allocates(algorithm):
    response = client.post(
        "/api/allocate",
        json={
            "scenario": {"preset": "small", "seed": 11},
            "algorithm": algorithm,
            "include_scenario": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["plan"]["metrics"]["valid"]


def test_compare_ranks_every_strategy_and_can_disable_local_search():
    response = client.post(
        "/api/compare",
        json={
            "scenario": {"preset": "small", "seed": 7},
            "options": {"regret": {"local_search": False}},
            "include_scenario": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    ranked = body["rankings"]
    assert len(ranked) >= 3
    assert body["winner"] == ranked[0]["algorithm"]
    objectives = [row["objective"] for row in ranked]
    assert objectives == sorted(objectives)
    for key in ("objective", "coverage_pct", "distance_km", "runtime_ms", "valid"):
        assert key in ranked[0]
    regret_plan = next(plan for plan in body["results"] if plan["algorithm"] == "regret")
    assert any("local search: off" in note for note in regret_plan["notes"])


def test_compare_only_includes_the_exhaustive_solver_for_tiny_instances():
    tiny = client.post(
        "/api/compare",
        json={
            "scenario": {"preset": "small", "vehicles": 2, "requests": 5},
            "include_scenario": False,
        },
    ).json()
    assert "exact" in [row["algorithm"] for row in tiny["rankings"]]

    big = client.post(
        "/api/compare", json={"scenario": {"preset": "small"}, "include_scenario": False}
    ).json()
    assert "exact" not in [row["algorithm"] for row in big["rankings"]]


def test_objective_overrides_are_honoured():
    spec = {"preset": "stress", "vehicles": 2, "requests": 30, "seed": 5}
    base = client.post(
        "/api/allocate",
        json={"scenario": spec, "algorithm": "regret", "include_scenario": False},
    ).json()["plan"]["metrics"]
    assert base["requests_unassigned"] > 0, "this scenario is meant to force refusals"

    priced = client.post(
        "/api/allocate",
        json={
            "scenario": spec,
            "algorithm": "regret",
            "objective": {"unassigned_base_cost": 300.0},
            "include_scenario": False,
        },
    ).json()["plan"]["metrics"]

    assert priced["objective_config"]["unassigned_base_cost"] == 300.0
    assert priced["objective"] > base["objective"]
    # the refusal term scales exactly with the new weight, and the plan is the same
    ratio = 300.0 / base["objective_config"]["unassigned_base_cost"]
    assert priced["objective_components"]["unassigned_cost"] == pytest.approx(
        base["objective_components"]["unassigned_cost"] * ratio, rel=1e-3
    )
    assert priced["requests_unassigned"] == base["requests_unassigned"]


@pytest.mark.parametrize(
    "payload,expected",
    (
        ({"scenario": {}, "algorithm": "nope"}, 400),
        ({"scenario": {"preset": "small"}, "algorithm": "greedy", "options": {"bogus": 1}}, 400),
        ({"scenario": {"preset": "medium", "requests": 200}, "algorithm": "exact"}, 400),
        ({"scenario": {"preset": "nope"}}, 400),
        ({"scenario": {"preset": "small", "requests": 5000}}, 422),
        ({"scenario": {"preset": "small", "seed": -1}}, 422),
    ),
)
def test_invalid_requests_fail_loudly(payload, expected):
    response = client.post("/api/allocate", json=payload)
    assert response.status_code == expected
    assert response.json()["detail"]


def test_rl_policy_endpoint_is_available_or_explains_itself():
    response = client.get("/api/rl/policy")
    if os.path.exists(default_policy_path()):
        assert response.status_code == 200
        body = response.json()
        assert body["weights"] and body["feature_names"]
        assert "train_episodes" in body["metadata"]
        without_curve = client.get("/api/rl/policy?include_curve=false").json()
        assert "curve" not in without_curve
    else:
        assert response.status_code == 503
        assert "policy artifact" in response.json()["detail"]


def test_openapi_document_is_exposed():
    schema = client.get("/openapi.json").json()
    assert "/api/allocate" in schema["paths"]
    assert "/api/compare" in schema["paths"]


def test_root_serves_the_spa_or_an_api_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    if os.path.isdir(FRONTEND_DIST):
        assert 'id="root"' in response.text
        asset = response.text.split('src="')[1].split('"')[0]
        asset_response = client.get(asset)
        assert asset_response.status_code == 200
        assert len(asset_response.content) > 1000
    else:
        assert "/api/allocate" in response.text
