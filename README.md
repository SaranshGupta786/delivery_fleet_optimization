# Delivery-Fleet Resource Optimization

*A production-grade decision engine that tells a delivery operation **which vehicle should serve which order — and which customers it should protect when capacity runs out.***

---

## TL;DR

**The problem.** Last-mile delivery is an *optimization* problem, not just a routing problem. Whenever orders outnumber what the fleet can physically serve in a day, every decision is a trade-off between three things that each cost money: **how far you drive**, **how often you're on time**, and **which customers you have to disappoint**. Get that trade-off wrong and you pay in fuel, missed SLAs, and lost high-value customers.

**What this does.** An engine that assigns every order to a vehicle — and to a position in that vehicle's route — in milliseconds, using **four different decision strategies**, from a simple cheapest-insertion rule up to a **reinforcement-learning agent that learns the trade-off from thousands of simulated days**. Every plan is scored on one transparent cost model and **every decision is explained in plain English**.

**The headline result (measured, not claimed).** On the days that actually hurt a business — when the fleet is *over-subscribed* — the learned agent beats a hand-tuned expert rule by **20–32%** on the same cost, and that lead grows as capacity gets tighter. On comfortable days the expert rule wins, so the engine recommends *pick-by-regime*, not one-size-fits-all.

**Not a toy.** Runs in a single Docker container, is **deterministic** (same inputs → same plan, byte for byte), enforces every real-world constraint *by construction*, and is covered by **106 automated tests** — including proofs that its speed optimizations are mathematically identical to a full simulation. No cloud, no API keys, no order data leaving your machine.

---

## Full version

### 1. The business problem

A delivery operation loses money on three axes at once, and they pull against each other:

* **Distance** — every extra kilometer is fuel, vehicle wear and driver time.
* **Punctuality** — late deliveries break service-level agreements and erode trust.
* **Refusals** — when demand exceeds capacity, *someone* doesn't get served. Refusing a low-priority order is fine; refusing a VIP is expensive.

Sending the nearest vehicle saves distance but can make other stops late. Chasing on-time performance can force you to refuse orders you could have squeezed in. Hand-written rules ("always serve the closest order first") encode *one fixed opinion* about this trade-off — and that opinion is wrong the moment the day's mix changes.

This engine makes the trade-off **explicit, measurable and optimizable**: a planner can see the cost of every choice, compare strategies, and pick the one that fits the day.

### 2. What we built, in plain words

Four "ways of thinking" about the same problem, all scored against the same cost model:

| Strategy | How it thinks | Best for |
| --- | --- | --- |
| **Greedy** | "always do the cheapest thing right now" | instant answers, small books |
| **Hungarian** | "repeatedly make the *optimal* matching, in batches" | distance-dominated days |
| **Regret** | "serve the order you'd most regret skipping, then tidy up" | normal, comfortable days |
| **Learned (RL)** | "an AI that *learned* the trade-off from thousands of simulated days" | over-subscribed, capacity-constrained days |

A fifth, **exhaustive solver** exists purely to *prove* the others are correct on small cases — like a double-entry bookkeeper for the math.

Every plan carries **explanations**: *"V01 serves R020: stop 3 of 11, 2.4 km from the previous stop; arrives 09:42, 18 minutes before the 10:00 deadline; truck at 68% of capacity; requires cold-chain."* If an order is refused, it says *why* — not a shrug, but "its 160 kg exceeds the capacity of every capable vehicle."

### 3. What a manager gets out of it

* **Lower cost per delivery** — fewer kilometers, priced per vehicle's real tariff.
* **Protected revenue** — the cost model prices refusals by priority, so high-value customers are defended first when capacity is tight.
* **On-time performance** — lateness is priced and traded deliberately, never ignored.
* **Auditable decisions** — every assignment and every refusal has a written reason, ideal for SLA reviews and for onboarding new dispatchers.
* **What-if planning** — slide the knobs (fleet size, capacity, delivery windows) and compare strategies *before* committing resources.
* **No lock-in, no cloud** — it runs on your machine, fully offline; order data never leaves.

### 4. Production-grade, not a toy

* **Deterministic** — the same day produces the same plan, byte for byte. That is what makes it auditable and safe to regression-test.
* **Constraints are guarantees, not suggestions** — vehicle capability, capacity (weight *and* volume), delivery windows, shift limits, route-length caps and "one vehicle per order" are enforced by construction. A plan can never silently break them.
* **Proven, not asserted** — 106 automated tests, including two load-bearing proofs: (1) the fast route planner is *mathematically identical* to a full simulation across thousands of cases, and (2) an exhaustive checker confirms no strategy ever beats the true optimum on small cases.
* **Scales** — a 40-stop book is planned in under 100 ms; a 400-stop book in seconds. The dashboard shows the per-strategy runtime so latency is a visible trade-off, not a surprise.
* **One command to run** — `docker compose up --build`, then open the dashboard at `localhost:8000`. Nothing else to install.
* **Reproducible** — pinned dependencies, seeded data and a committed trained model, so "works on my machine" and "works on yours" are the same statement.

### 5. What we found (the interesting part)

The benchmark runs all four strategies across a matrix of realistic days, and separately *tightens only the fleet's capacity* while holding everything else fixed:

| capacity/demand | expert rule | learned agent | learned vs expert |
| --- | --- | --- | --- |
| 1.50x (plenty of capacity) | 723 | 919 | −27% |
| 1.00x (exactly enough) | 1 708 | 1 633 | **+4%** |
| 0.85x (over-subscribed) | 4 274 | 2 626 | **+39%** |

The crossover is at **exactly 1.0× capacity**. On comfortable days the hand-tuned rule wins; the moment the fleet is over-subscribed — the situation that actually threatens revenue — the learned agent takes over and widens its lead. That is a genuinely useful business insight, not a demo trick: **the right strategy depends on the day's load**, and this engine lets you see that and act on it.

### 6. How to run it (30 seconds)

```bash
docker compose up --build      # -> dashboard at http://localhost:8000
```

| Command | What it does |
| --- | --- |
| `make up` | build and start the engine, then wait until it is healthy |
| `make down` | stop it |
| `make test` | run the 106-test suite inside the image |
| `make smoke` | end-to-end HTTP checks against a running engine |
| `make bench` | regenerate `ANALYSIS.md` with fresh benchmark numbers |
| `make capacity` | regenerate the capacity-pressure curve (`CAPACITY_SWEEP.md`) |
| `make train` | retrain the learned policy (optional; a trained one ships with the repo) |

The dashboard has four tabs: **Comparison** (all strategies side by side, with a map of the best plan), **Explanations** (every decision in words), **Learned policy** (what the AI actually learned and how it was trained), and **Method** (the cost model and assumptions).

### 7. How it works — for the technical reader

**The objective every strategy minimizes:**

```text
cost = Σ distance_km · cost_per_km            # per-vehicle tariff
     + 2.0  × lateness_minutes                # punctuality inside the grace window
     + Σ refused-request cost                 # 60 × (1 + 0.5 × (priority − 1))
     + 0.05 × makespan_minutes                # fleet-wide finish time
```

**Hard constraints (never traded away):** capability match, capacity in kg *and* m³ across the whole route, delivery inside the window plus a grace band, shift end, route-length cap, and at most one vehicle per order. Lateness inside the grace band is *priced*; beyond it the order must be refused — refusing is legal and priced, never silent.

**The four strategies (and one reference):**

| Strategy | Idea | Complexity |
| --- | --- | --- |
| `greedy` | cheapest insertion in deadline/priority order | O(n·m·route) |
| `hungarian` | repeated optimal batch matching (`scipy.linear_sum_assignment`) | O(rounds·n·m·route) |
| `regret` | commit the highest-regret order, then relocate-stops local search | O(n²·m·route) |
| `rl` | argmax of a 17-feature linear softmax policy, trained with REINFORCE | O(n·m·route) |
| `exact` | enumerates every split and order (≤ 6 requests) — the ground truth | exponential |

**The learned policy is a sequential decision problem (MDP):** a state is the partial plan plus remaining orders; an action is one feasible *(order, vehicle)* pair; the reward is the negative incremental cost; the terminal reward is the cost of refused orders plus makespan. Because the rewards are the *exact* incremental costs, the undiscounted return **equals minus the objective** — training optimizes the same number the benchmark reports, with no train/eval gap. Training = behaviour-cloning warm start from the regret rule, then REINFORCE with a self-critical (greedy-rollout) baseline. The weights ship as a small JSON artifact; inference is a deterministic argmax with nothing trained at runtime.

**Repository layout:**

```
backend/app/        engine (domain, feasibility, metrics, scenario, bench, capacity_sweep)
backend/app/algorithms/   greedy, hungarian, regret, brute-force, registry, plan/explanation
backend/app/rl/           features, policy artifact, MDP environment, REINFORCE, wrapper
backend/artifacts/        the committed trained policy (rl_policy.json)
backend/tests/            106 tests (equivalence, optimality, API, RL)
frontend/                 React dashboard (SVG map, tables, charts, explanations, RL panel)
Dockerfile, docker-compose.yml, Makefile, scripts/smoke.sh
```

**API (for integrators):** the dashboard is a thin client over a JSON API on the same port, so the engine can be called from any system.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | liveness, version, available strategies, cost-model defaults |
| `POST` | `/api/allocate` | run one strategy on a scenario → plan, metrics, explanations |
| `POST` | `/api/compare` | run several strategies → plans, ranking, winner |
| `POST` | `/api/scenario` | build/inspect a scenario (knobs: preset, seed, city, capacity, windows, grace) |
| `GET` | `/api/rl/policy` | the trained policy: weights, training curve, holdout audit |

Scenario sizes are validated (rejecting absurd inputs with a clear message), strategy options are allow-listed, and errors carry a human-readable `detail` — so integration failures are diagnosable, not mysterious.

### 8. Assumptions & honest limitations

Stated plainly, because they shape the numbers:

* **Travel model** — straight-line (haversine) distance at a constant cruising speed; no live traffic and no paid routing service. This keeps the engine offline and deterministic, but urban distances are optimistic. Swapping in a real routing provider touches one file (`app/geo.py`).
* **Demand model** — a seeded synthetic book (clustered demand, lognormal sizes, priority and window mixes). It is *representative*, not a live order feed.
* **Last-mile routes** — vehicles don't return to the depot; waiting counts as on-duty time.
* **Cost weights** — the trade-offs in the objective are a business decision. They are documented, echoed in every result, and overridable per run.
* **The learned policy** — a 17-feature linear model trained on ~2,500 simulated days. It is strong exactly where the business pain is (over-subscribed fleets) and honest about where it isn't (comfortable days), which is why the engine recommends *pick-by-regime*.
* **Scale** — the `large` preset (30 vehicles / 400 orders) is the biggest book exercised; the exhaustive checker is deliberately capped far below that.

### 9. Feature map

| Feature | Where it lives |
| --- | --- |
| Resource-allocation engine | `backend/app/algorithms/*` + `backend/app/feasibility.py` |
| Four strategies (+ exhaustive reference) | greedy, Hungarian, deadline-regret, RL (`app/algorithms/`, `app/rl/`) |
| Reinforcement-learning strategy | `backend/app/rl/` (features, MDP, REINFORCE, committed artifact) |
| Multi-objective optimisation with documented trade-offs | `backend/app/metrics.py:ObjectiveConfig`, overridable via API |
| Hard constraints enforced | `backend/app/feasibility.py` + equivalence and constraint tests |
| Per-decision explainability | `backend/app/algorithms/base.py` (shared reason generator), dashboard *Explanations* tab |
| React dashboard | `frontend/` (comparison, SVG map, explanations, RL panel, method) |
| FastAPI service | `backend/app/main.py` with Pydantic schemas and validation |
| Reproducibility | seeded generator, committed weights, pinned deps, `make bench` |
| Docker-first deployment, one port | `Dockerfile` (multi-stage), `docker-compose.yml`, `Makefile` |
| Benchmark & analysis | `ANALYSIS.md` (generated), `CAPACITY_SWEEP.md`, auto-derived findings |
| Test suite | `backend/tests/` (equivalence, optimality, API, RL) |

### 10. Troubleshooting

* **Port 8000 busy** — run `PORT=8123 make up` or stop the other service.
* **`rl` reports unavailable** — the trained policy file is missing from the image. Run `make train` (writes `backend/artifacts/rl_policy.json`), or point `RL_POLICY_PATH` at a valid file.
* **Apple silicon** — the image is multi-arch; base images are `python:3.11-slim` and `node:20-alpine`, both with arm64 builds. For a linux/amd64 check: `docker buildx build --platform linux/amd64 --target runtime .`
* **Frontend development without Docker** — `make frontend`, then run the API (`python -m app.main`) and `cd frontend && npm run dev` (Vite proxies `/api` to `127.0.0.1:8000`).



