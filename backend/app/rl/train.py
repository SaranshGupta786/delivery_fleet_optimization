"""REINFORCE training for the allocation policy.

Run it with::

    python -m app.rl.train --episodes 900 --seed 7 --out artifacts/rl_policy.json

What happens:

* fresh scenarios are sampled from the same generator the benchmark uses (a new
  seed every episode), so the weights have to generalise across books instead of
  memorising one;
* each episode is played by *sampling* from the softmax at a temperature below 1
  (sharper exploration while the weights are still small), the exact return is
  computed, and REINFORCE is applied with a *self-critical baseline*: the same
  scenario is also played greedily with the current weights, and the advantage is
  the difference between the two reward-to-go profiles, z-scored per decision
  index and clipped. That baseline depends only on the weights, never on the
  sampled actions, so the gradient stays unbiased while losing most of the
  episode variance. An entropy bonus keeps the softmax from collapsing early;
* every ``--eval-every`` episodes the current policy plays a fixed holdout set
  with the deterministic argmax rule, producing a validation curve that is
  written into the artifact alongside the weights;
* the artifact is JSON: 13 weights, feature names, hyper-parameters, curves and
  the holdout comparison against the classical strategies. Inference never
  trains - the API only loads this file.

Training is deterministic for a given seed (numpy Generator throughout).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from ..metrics import DEFAULT_OBJECTIVE, ObjectiveConfig
from ..scenario import generate_scenario
from .env import AllocationEnv
from .features import FEATURE_DIM
from .policy import DEFAULT_POLICY_PATH, LinearSoftmaxPolicy

TRAIN_PRESETS = ("small", "small", "small", "small", "small", "small", "small", "medium")
TRAIN_SEED_BASE = 10_000
IMITATION_SEED_BASE = 900_000
HOLDOUT_SEED_BASE = 500_000
MAX_CURVE_POINTS = 400


@dataclass
class TrainConfig:
    """Everything needed to reproduce an artifact."""

    episodes: int = 2500
    lr: float = 0.05
    entropy_coef: float = 0.03
    temperature: float = 0.3
    advantage_scale: str = "z_score"
    baseline_mode: str = "rollout"
    imitation_passes: int = 6
    imitation_scenarios: int = 6
    imitation_lr: float = 0.6
    baseline_decay: float = 0.9
    advantage_clip: float = 3.0
    gradient_clip: float = 1.0
    seed: int = 7
    eval_every: int = 250
    holdout_scenarios: int = 3
    validation_preset: str = "small"
    train_presets: tuple = TRAIN_PRESETS
    city: str = "bengaluru"
    capacity_ratio_range: tuple = (1.05, 1.90)
    window_scale_range: tuple = (0.50, 1.20)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["train_presets"] = list(self.train_presets)
        payload["capacity_ratio_range"] = list(self.capacity_ratio_range)
        payload["window_scale_range"] = list(self.window_scale_range)
        return payload


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(np.max(scores))
    exponentials = np.exp(shifted)
    return exponentials / float(np.sum(exponentials))


def _sample_episode(env: AllocationEnv, weights: np.ndarray, rng, temperature: float = 1.0):
    """Play one episode by sampling from the softmax; returns (matrices, indices, rewards)."""
    matrices = []
    indices = []
    rewards = []
    while True:
        actions, matrix = env.snapshot()
        if not actions:
            break
        probabilities = _softmax(matrix @ weights / temperature)
        index = int(rng.choice(len(actions), p=probabilities))
        matrices.append(matrix)
        indices.append(index)
        rewards.append(env.step(actions[index]))
    return matrices, indices, rewards


def _returns_to_go(rewards, terminal: float) -> list:
    """Undiscounted reward-to-go for every decision of an episode."""
    returns: list = []
    running = terminal
    for reward in reversed(list(rewards)):
        running += reward
        returns.append(running)
    returns.reverse()
    return returns


def _greedy_returns_to_go(scenario, weights: np.ndarray) -> list:
    """Reward-to-go of the current policy's greedy rollout.

    This is the self-critical baseline: it depends only on the *weights*, never on
    the actions that were sampled, so subtracting it leaves the policy gradient
    unbiased while removing most of the episode-to-episode variance that made the
    plain REINFORCE update so slow to converge.
    """
    env = AllocationEnv(scenario, DEFAULT_OBJECTIVE)
    rewards: list = []
    while True:
        actions, matrix = env.snapshot()
        if not actions:
            break
        rewards.append(env.step(actions[int(np.argmax(matrix @ weights))]))
    return _returns_to_go(rewards, env.terminal_reward())


def imitation_phase(config: TrainConfig) -> np.ndarray:
    """Warm-start the policy by imitating the regret heuristic (behaviour cloning).

    For every decision the expert makes, the cross-entropy gradient toward the
    expert's (request, vehicle) action is applied - which is precisely the
    REINFORCE update with a constant +1 advantage on the expert action. The policy
    therefore starts at roughly the hand-tuned heuristic's quality, and REINFORCE
    only has to improve it instead of rediscovering the regret rule from noise.

    Approximation worth stating: the expert's plan is replayed vehicle by vehicle
    (each vehicle's route in the expert's order) rather than in the expert's exact
    global decision order, and each target is inserted at the position the engine
    finds cheapest. Both keep the trajectory close to the expert's without needing
    the heuristic to expose its internal step order.
    """
    from ..algorithms.heuristic import RegretAllocator

    weights = np.zeros(FEATURE_DIM, dtype=float)
    scenario_count = max(0, int(config.imitation_scenarios))
    if scenario_count == 0 or config.imitation_passes <= 0:
        return weights

    scenarios = [
        generate_scenario(
            config.train_presets[index % len(config.train_presets)],
            seed=IMITATION_SEED_BASE + index,
            city=config.city,
        )
        for index in range(scenario_count)
    ]
    temperature = max(float(config.temperature), 1e-6)
    expert = RegretAllocator()

    for _ in range(int(config.imitation_passes)):
        for scenario in scenarios:
            plan = expert.allocate(scenario, DEFAULT_OBJECTIVE)
            targets = [
                (request_id, vehicle_id)
                for vehicle_id, request_ids in plan.routes.items()
                for request_id in request_ids
            ]
            env = AllocationEnv(scenario, DEFAULT_OBJECTIVE)
            gradient = np.zeros(FEATURE_DIM, dtype=float)
            matched = 0
            for request_id, vehicle_id in targets:
                actions, matrix = env.snapshot()
                index = None
                for position, action in enumerate(actions):
                    if action.request_id == request_id and action.vehicle_id == vehicle_id:
                        index = position
                        break
                if index is None:
                    continue
                probabilities = _softmax(matrix @ weights / temperature)
                gradient += matrix[index] - probabilities @ matrix
                matched += 1
                env.step(actions[index])
            if matched:
                weights = weights + config.imitation_lr * (gradient / matched)
    return weights


def train_policy(config: TrainConfig, progress=None) -> dict:
    """Run imitation (optional) plus REINFORCE; returns weights and a report."""
    rng = np.random.default_rng(config.seed)
    if config.advantage_scale not in ("z_score", "baseline_magnitude"):
        raise ValueError(
            "advantage_scale must be 'z_score' or 'baseline_magnitude' (got %r)"
            % config.advantage_scale
        )
    if config.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if config.baseline_mode not in ("rollout", "ema"):
        raise ValueError("baseline_mode must be 'rollout' or 'ema' (got %r)" % config.baseline_mode)

    imitation_weights = (
        imitation_phase(config) if config.imitation_passes > 0 else np.zeros(FEATURE_DIM)
    )
    weights = np.array(imitation_weights, dtype=float)
    baseline = 0.0
    baselines: list = []
    scales: list = []
    curve: list = []
    validation: list = []
    holdout = [
        generate_scenario(
            config.validation_preset, seed=HOLDOUT_SEED_BASE + index, city=config.city
        )
        for index in range(config.holdout_scenarios)
    ]
    started = time.perf_counter()

    for episode in range(1, int(config.episodes) + 1):
        preset = config.train_presets[(episode - 1) % len(config.train_presets)]
        # Regime randomisation: vary capacity pressure and window width so the
        # policy transfers to the tight/stress presets instead of overfitting to
        # a single difficulty level.
        scenario = generate_scenario(
            preset,
            seed=TRAIN_SEED_BASE + episode,
            city=config.city,
            capacity_ratio=float(rng.uniform(*config.capacity_ratio_range)),
            window_scale=float(rng.uniform(*config.window_scale_range)),
        )
        env = AllocationEnv(scenario, DEFAULT_OBJECTIVE)
        matrices, indices, rewards = _sample_episode(env, weights, rng, config.temperature)
        terminal = env.terminal_reward()
        episode_return = float(np.sum(rewards) + terminal) if rewards else terminal
        curve.append(episode_return)

        if matrices:
            returns_to_go = _returns_to_go(rewards, terminal)
            while len(baselines) < len(matrices):
                baselines.append(0.0)
                scales.append(1.0)

            reference_returns = (
                _greedy_returns_to_go(scenario, weights)
                if config.baseline_mode == "rollout"
                else None
            )

            if config.baseline_mode == "ema" and episode == 1:
                # The first EMA episode only seeds the per-decision baseline.
                for step, value in enumerate(returns_to_go):
                    baselines[step] = value
                    scales[step] = value * value
            else:
                gradient = np.zeros(FEATURE_DIM, dtype=float)
                for step, (matrix, index) in enumerate(zip(matrices, indices)):
                    if reference_returns is None:
                        reference = baselines[step]
                    else:
                        reference = (
                            reference_returns[step]
                            if step < len(reference_returns)
                            else reference_returns[-1]
                        )
                    deviation = returns_to_go[step] - reference
                    if config.advantage_scale == "baseline_magnitude":
                        scale = max(1.0, abs(reference))
                    else:
                        # z-score style normalisation per decision index, so every
                        # decision gets a comparable learning signal.
                        scale = max(1.0, float(np.sqrt(scales[step])))
                    advantage = float(
                        np.clip(deviation / scale, -config.advantage_clip, config.advantage_clip)
                    )
                    probabilities = _softmax(matrix @ weights / config.temperature)
                    features = matrix[index]
                    gradient += advantage * (features - probabilities @ matrix)
                    log_probabilities = np.log(probabilities + 1e-12)
                    entropy = float(-np.sum(probabilities * log_probabilities))
                    gradient += config.entropy_coef * (
                        (-probabilities * (log_probabilities + entropy)) @ matrix
                    )
                    if reference_returns is None:
                        baselines[step] = (
                            config.baseline_decay * reference
                            + (1.0 - config.baseline_decay) * returns_to_go[step]
                        )
                    scales[step] = (
                        config.baseline_decay * scales[step]
                        + (1.0 - config.baseline_decay) * deviation * deviation
                    )
                gradient = gradient / len(matrices)
                norm = float(np.linalg.norm(gradient))
                if norm > config.gradient_clip:
                    gradient = gradient * (config.gradient_clip / norm)
                weights = weights + config.lr * gradient
        baseline = config.baseline_decay * baseline + (1.0 - config.baseline_decay) * episode_return

        if progress is not None and (episode == 1 or episode % config.eval_every == 0):
            progress(episode, episode_return, baseline)

        if episode % config.eval_every == 0 or episode == int(config.episodes):
            policy = LinearSoftmaxPolicy(weights)
            objectives = [evaluate_policy_on(policy, scenario) for scenario in holdout]
            validation.append(
                {
                    "episode": episode,
                    "mean_objective": round(float(np.mean(objectives)), 3),
                    "objectives": [round(value, 3) for value in objectives],
                }
            )

    return {
        "weights": weights,
        "curve": curve,
        "baselines": baselines,
        "scales": scales,
        "validation": validation,
        "train_seconds": time.perf_counter() - started,
    }


def evaluate_policy_on(policy: LinearSoftmaxPolicy, scenario, config=DEFAULT_OBJECTIVE) -> float:
    """Deterministic (argmax) rollout - exactly what the API serves."""
    env = AllocationEnv(scenario, config)
    while True:
        actions, matrix = env.snapshot()
        if not actions:
            break
        env.step(actions[policy.best_index(matrix)])
    return env.objective()


def subsample(values, limit: int = MAX_CURVE_POINTS) -> list:
    """Thin a curve down to at most ``limit`` points for a compact artifact."""
    if len(values) <= limit:
        return [round(float(value), 3) for value in values]
    stride = len(values) / float(limit)
    return [round(float(values[min(len(values) - 1, int(index * stride))]), 3) for index in range(limit)]


def moving_average(values, window: int = 25) -> list:
    if not values:
        return []
    averaged = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        averaged.append(running / min(window, index + 1))
    return averaged


def holdout_comparison(holdout, policy: LinearSoftmaxPolicy, config=DEFAULT_OBJECTIVE) -> dict:
    """Objective of every strategy on the holdout set - the artifact's own audit."""
    # Imported inside the function: app.algorithms registers the RL strategy, so
    # importing it at module import time would create a cycle.
    from ..algorithms.greedy import GreedyAllocator
    from ..algorithms.heuristic import RegretAllocator
    from ..algorithms.hungarian import HungarianAllocator

    comparison = {}
    for allocator in (GreedyAllocator(), HungarianAllocator(), RegretAllocator()):
        values = [allocator.allocate(scenario, config).metrics["objective"] for scenario in holdout]
        comparison[allocator.name] = {
            "mean": round(float(np.mean(values)), 3),
            "values": [round(float(value), 3) for value in values],
        }
    policy_values = [evaluate_policy_on(policy, scenario, config) for scenario in holdout]
    comparison["rl"] = {
        "mean": round(float(np.mean(policy_values)), 3),
        "values": [round(float(value), 3) for value in policy_values],
    }
    comparison["best"] = min(comparison.items(), key=lambda item: item[1]["mean"])[0]
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the linear softmax allocation policy with REINFORCE."
    )
    parser.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--entropy-coef", type=float, default=TrainConfig.entropy_coef)
    parser.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    parser.add_argument("--baseline-mode", default=TrainConfig.baseline_mode)
    parser.add_argument("--advantage-scale", default=TrainConfig.advantage_scale)
    parser.add_argument("--imitation-passes", type=int, default=TrainConfig.imitation_passes)
    parser.add_argument("--imitation-scenarios", type=int, default=TrainConfig.imitation_scenarios)
    parser.add_argument("--imitation-lr", type=float, default=TrainConfig.imitation_lr)
    parser.add_argument("--baseline-decay", type=float, default=TrainConfig.baseline_decay)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--holdout", type=int, default=TrainConfig.holdout_scenarios)
    parser.add_argument("--validation-preset", default=TrainConfig.validation_preset)
    parser.add_argument("--out", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        episodes=args.episodes,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        temperature=args.temperature,
        baseline_mode=args.baseline_mode,
        advantage_scale=args.advantage_scale,
        imitation_passes=args.imitation_passes,
        imitation_scenarios=args.imitation_scenarios,
        imitation_lr=args.imitation_lr,
        baseline_decay=args.baseline_decay,
        seed=args.seed,
        eval_every=args.eval_every,
        holdout_scenarios=args.holdout,
        validation_preset=args.validation_preset,
    )

    def progress(episode, episode_return, baseline):
        print(
            "episode %4d  return %10.2f  baseline %10.2f" % (episode, episode_return, baseline),
            flush=True,
        )

    report = train_policy(config, progress=None if args.quiet else progress)
    policy = LinearSoftmaxPolicy(report["weights"])
    holdout = [
        generate_scenario(
            config.validation_preset, seed=HOLDOUT_SEED_BASE + index, city=config.city
        )
        for index in range(config.holdout_scenarios)
    ]
    comparison = holdout_comparison(holdout, policy)
    curve = report["curve"]
    policy.metadata = {
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_config": config.to_dict(),
        "train_seconds": round(report["train_seconds"], 1),
        "train_episodes": int(config.episodes),
        "train_return_first_50": round(float(np.mean(curve[:50])), 3) if curve else None,
        "train_return_last_50": round(float(np.mean(curve[-50:])), 3) if curve else None,
        "validation": report["validation"],
        "holdout_preset": config.validation_preset,
        "holdout_comparison": comparison,
        "curve": subsample(curve),
        "curve_smoothed": subsample(moving_average(curve)),
        "policy": "linear softmax over feasible (request, vehicle) pair features",
        "reward_identity": "undiscounted episode return equals -objective of the finished plan",
        "numpy_version": np.__version__,
    }
    target = policy.save(args.out)

    if not args.quiet:
        print("artifact written to %s" % target)
        print("weights: %s" % json.dumps(policy.to_dict()["weights"]))
        print(
            "holdout mean objective - %s"
            % ", ".join(
                "%s %.1f" % (name, stats["mean"])
                for name, stats in comparison.items()
                if isinstance(stats, dict)
            )
        )
        print("best on holdout: %s" % comparison["best"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

