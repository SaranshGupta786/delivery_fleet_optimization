"""RL package tests: MDP identity, artifact contract, determinism, registration."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.algorithms import get_allocator
from app.rl.env import AllocationEnv, rollout
from app.rl.features import FEATURE_DIM, FEATURE_NAMES, STATIC_DIM
from app.rl.policy import LinearSoftmaxPolicy, PolicyArtifactError, default_policy_path
from app.rl.train import TrainConfig, evaluate_policy_on, imitation_phase, subsample, train_policy
from app.scenario import generate_scenario
from tests.conftest import assert_plan_is_consistent


def tiny_train_config(**overrides):
    payload = dict(
        episodes=4,
        eval_every=4,
        holdout_scenarios=1,
        imitation_passes=1,
        imitation_scenarios=1,
        train_presets=("small",),
    )
    payload.update(overrides)
    return TrainConfig(**payload)


def test_episode_return_equals_negative_objective(small_scenario):
    """The property the whole training loop rests on."""
    env = AllocationEnv(small_scenario)
    actions, episode_return = rollout(env)
    assert actions
    assert episode_return == pytest.approx(-env.objective(), abs=1e-6)
    assert env.stopped()
    assert len(env.pending) + env.steps == len(small_scenario.requests)


def test_snapshot_features_are_finite_and_well_shaped(small_scenario):
    env = AllocationEnv(small_scenario)
    actions, matrix = env.snapshot()
    assert 0 < len(actions) <= len(small_scenario.requests) * len(small_scenario.vehicles)
    assert matrix.shape == (len(actions), FEATURE_DIM)
    assert np.all(np.isfinite(matrix))
    scarcity = matrix[:, FEATURE_NAMES.index("scarcity")]
    assert scarcity.min() >= 0.0 and scarcity.max() <= 1.0 + 1e-9
    assert all(action.request_id in env.pending for action in actions)


def test_feature_contract_is_pinned():
    assert FEATURE_DIM == 17
    assert STATIC_DIM == 12
    assert FEATURE_NAMES[:2] == ("bias", "detour_km")
    assert FEATURE_NAMES[-1] == "urgency"


def test_policy_artifact_round_trip(tmp_path):
    policy = LinearSoftmaxPolicy(weights=np.arange(FEATURE_DIM, dtype=float))
    target = str(tmp_path / "policy.json")
    policy.save(target)
    reloaded = LinearSoftmaxPolicy.load(target)
    assert np.allclose(reloaded.weights, policy.weights)
    assert reloaded.feature_names == FEATURE_NAMES


def test_policy_rejects_broken_artifacts(tmp_path):
    with pytest.raises(PolicyArtifactError):
        LinearSoftmaxPolicy(weights=[1.0, 2.0])

    target = tmp_path / "wrong-version.json"
    payload = LinearSoftmaxPolicy(weights=np.zeros(FEATURE_DIM)).to_dict()
    payload["version"] = 99
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PolicyArtifactError):
        LinearSoftmaxPolicy.load(str(target))

    with pytest.raises(PolicyArtifactError):
        LinearSoftmaxPolicy.load(str(tmp_path / "does-not-exist.json"))


def test_training_is_deterministic():
    first = train_policy(tiny_train_config())
    second = train_policy(tiny_train_config())
    assert np.allclose(first["weights"], second["weights"])
    assert len(first["curve"]) == 4
    assert first["validation"][-1]["episode"] == 4


def test_training_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        train_policy(tiny_train_config(advantage_scale="nonsense"))
    with pytest.raises(ValueError):
        train_policy(tiny_train_config(temperature=0.0))
    with pytest.raises(ValueError):
        train_policy(tiny_train_config(baseline_mode="nonsense"))


def test_imitation_improves_on_an_untrained_policy():
    """Behaviour cloning must at least beat an untrained policy on average."""
    config = tiny_train_config(imitation_passes=3, imitation_scenarios=2, episodes=0)
    weights = imitation_phase(config)
    assert float(np.linalg.norm(weights)) > 0.1

    imitated = LinearSoftmaxPolicy(weights)
    untrained = LinearSoftmaxPolicy(np.zeros(FEATURE_DIM))
    books = [generate_scenario("small", seed=900_000 + index) for index in range(3)]
    imitated_mean = float(np.mean([evaluate_policy_on(imitated, s) for s in books]))
    untrained_mean = float(np.mean([evaluate_policy_on(untrained, s) for s in books]))
    assert imitated_mean < untrained_mean, "%.1f should beat %.1f" % (imitated_mean, untrained_mean)


def test_rl_policy_uses_greedy_argmax_deterministically(small_scenario):
    config = tiny_train_config()
    weights = imitation_phase(config)
    policy = LinearSoftmaxPolicy(weights)
    features = np.array([[0.0] * FEATURE_DIM, [1.0] * FEATURE_DIM])
    assert policy.best_index(features) == policy.best_index(features)
    first = evaluate_policy_on(policy, small_scenario)
    second = evaluate_policy_on(policy, small_scenario)
    assert first == second


def test_subsample_caps_curve_length():
    assert len(subsample(list(range(1000)), limit=100)) == 100
    assert subsample([1.0, 2.0], limit=100) == [1.0, 2.0]


@pytest.mark.skipif(
    not os.path.exists(default_policy_path()), reason="no policy artifact committed yet"
)
def test_committed_artifact_matches_this_build(small_scenario):
    policy = LinearSoftmaxPolicy.load(default_policy_path())
    assert policy.feature_names == FEATURE_NAMES
    assert policy.metadata.get("train_episodes", 0) > 0
    assert "rl" in policy.metadata.get("holdout_comparison", {})

    allocator = get_allocator("rl")
    first = allocator.allocate(small_scenario)
    second = allocator.allocate(small_scenario)
    assert_plan_is_consistent(first, small_scenario)
    assert first.routes == second.routes, "inference must be deterministic"
    assert any("policy artifact" in note for note in first.notes)

    polished = allocator.allocate(small_scenario, local_search=True)
    assert_plan_is_consistent(polished, small_scenario)
    assert any("polish:" in note for note in polished.notes)
