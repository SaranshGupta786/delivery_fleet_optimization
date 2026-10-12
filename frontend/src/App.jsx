import { useCallback, useEffect, useMemo, useState } from "react";

import ControlBar from "./components/ControlBar.jsx";
import MapView from "./components/MapView.jsx";
import MetricTable from "./components/MetricTable.jsx";
import ObjectiveChart from "./components/ObjectiveChart.jsx";
import PlanPanel from "./components/PlanPanel.jsx";
import PolicyPanel from "./components/PolicyPanel.jsx";
import {
  compareStrategies,
  fetchAlgorithms,
  fetchHealth,
  fetchPolicy,
  fetchPresets,
} from "./lib/api.js";
import { formatNumber } from "./lib/format.js";

const DEFAULT_CONTROLS = {
  preset: "small",
  seed: 42,
  city: "bengaluru",
  capacityRatio: "",
  windowScale: "",
  graceMinutes: 30,
  selectedAlgorithms: ["greedy", "hungarian", "regret", "rl"],
  polishRL: false,
};

const TABS = [
  { id: "compare", label: "Comparison" },
  { id: "plan", label: "Explanations" },
  { id: "policy", label: "Learned policy" },
  { id: "method", label: "Method & objective" },
];

function scenarioPayload(controls) {
  const scenario = {
    preset: controls.preset,
    seed: controls.seed,
    city: controls.city,
    grace_minutes: controls.graceMinutes,
  };
  if (controls.capacityRatio !== "") scenario.capacity_ratio = Number(controls.capacityRatio);
  if (controls.windowScale !== "") scenario.window_scale = Number(controls.windowScale);
  return scenario;
}

function MethodPanel({ comparison }) {
  const objective = comparison.objective || {};
  const strategies = [
    ["greedy", "Cheapest insertion in deadline/priority order, one request at a time.", "O(n·m·route length)"],
    ["hungarian", "Repeated optimal batch matching (scipy linear_sum_assignment) on insertion cost.", "O(rounds·n·m·route length)"],
    ["regret", "Commit the request that loses most if deferred, then relocate-stop local search.", "O(n²·m·route length)"],
    ["rl", "Linear softmax policy trained with REINFORCE on sampled books; ship weights only.", "O(n·m·route length)"],
    ["exact", "Exhaustive enumeration of splits and orders — the optimality reference (tiny instances).", "exponential (guarded)"],
  ];

  return (
    <div className="grid">
      <div className="card">
        <h2>The objective every strategy minimises</h2>
        <pre className="formula">{`objective = Σ distance_km · cost_per_km          (per vehicle)
          + ${objective.lateness_cost_per_min ?? 2} × lateness_minutes
          + Σ refused request cost  (base ${objective.unassigned_base_cost ?? 60} × priority weight)
          + ${objective.makespan_cost_per_min ?? 0.05} × makespan_minutes`}</pre>
        <ul className="bullets">
          <li>
            Distance is priced per vehicle, so sending a cheaper vehicle further away can beat
            sending an expensive one nearby.
          </li>
          <li>
            Lateness is tolerated inside the grace window (and priced), beyond it the request must be
            refused - refusing is legal and priced, never silently dropped.
          </li>
          <li>
            Refusing a priority-5 request costs {1 + 4 * (objective.priority_step ?? 0.5)}× a
            priority-1 one, which is what makes priority-aware coverage measurable.
          </li>
        </ul>

        <h3>Hard constraints (never traded away)</h3>
        <ul className="bullets">
          <li>capability match (cold chain, fragile, heavy lift, lift gate)</li>
          <li>capacity in kg and m³, summed over the whole route</li>
          <li>service inside the window plus grace; shift end; route-length cap</li>
          <li>each request served at most once - enforced by an explicit check</li>
        </ul>
      </div>

      <div className="card">
        <h2>The strategies</h2>
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Strategy</th>
              <th scope="col">Idea</th>
              <th scope="col">Cost class</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map(([name, idea, complexity]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td className="notes">{idea}</td>
                <td className="muted small">{complexity}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h3>Assumptions</h3>
        <ul className="bullets">
          <li>
            Travel is straight-line haversine at a constant cruising speed: no traffic feed and no
            paid routing service, so the engine stays offline and reproducible.
          </li>
          <li>
            Demand is a seeded synthetic book clustered around hotspots; the generator is
            deterministic for a given (preset, seed).
          </li>
          <li>
            Vehicle utilisation counts waiting time as on-duty, and routes do not return to the
            depot (last-mile model).
          </li>
          <li>The learned policy ships as a committed JSON artifact; nothing trains at runtime.</li>
        </ul>

        <h3>Reproduce</h3>
        <pre className="formula">{`docker compose up --build      # API + UI on http://localhost:8000
make test                      # in-image pytest suite
make bench                     # writes ANALYSIS.md
make train                     # retrains the policy artifact (optional)`}</pre>
      </div>
    </div>
  );
}


export default function App() {
  const [health, setHealth] = useState(null);
  const [algorithms, setAlgorithms] = useState([]);
  const [presets, setPresets] = useState([]);
  const [controls, setControls] = useState(DEFAULT_CONTROLS);
  const [comparison, setComparison] = useState(null);
  const [selected, setSelected] = useState(null);
  const [tab, setTab] = useState("compare");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [policy, setPolicy] = useState(null);
  const [policyError, setPolicyError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [healthPayload, algorithmPayload, presetPayload] = await Promise.all([
          fetchHealth(),
          fetchAlgorithms(),
          fetchPresets(),
        ]);
        if (cancelled) return;
        setHealth(healthPayload);
        setAlgorithms(algorithmPayload.algorithms);
        setPresets(presetPayload.presets);
        const available = algorithmPayload.algorithms
          .filter((entry) => entry.available)
          .map((entry) => entry.name);
        setControls((previous) => ({
          ...previous,
          selectedAlgorithms: previous.selectedAlgorithms.filter((name) =>
            available.includes(name)
          ),
        }));
      } catch (loadError) {
        if (!cancelled) setError(loadError.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchPolicy(true)
      .then((payload) => {
        if (!cancelled) setPolicy(payload);
      })
      .catch((policyLoadError) => {
        if (!cancelled) setPolicyError(policyLoadError.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const run = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const available = algorithms.filter((entry) => entry.available).map((entry) => entry.name);
      let names = controls.selectedAlgorithms.filter((name) => available.includes(name));
      if (names.length === 0) names = available;
      const response = await compareStrategies({
        scenario: scenarioPayload(controls),
        algorithms: names,
        options: controls.polishRL && names.includes("rl") ? { rl: { local_search: true } } : {},
        include_scenario: true,
      });
      setComparison(response);
      setSelected(response.winner);
      setTab("compare");
    } catch (runError) {
      setError(runError.message);
    } finally {
      setBusy(false);
    }
  }, [algorithms, controls]);

  useEffect(() => {
    if (algorithms.length > 0 && comparison === null && !busy) {
      run();
    }
    // Intentionally runs once, after the strategy catalogue arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algorithms.length]);

  const plans = comparison ? comparison.results : [];
  const selectedPlan = useMemo(() => {
    if (plans.length === 0) return null;
    return plans.find((plan) => plan.algorithm === selected) || plans[0];
  }, [plans, selected]);

  const change = (patch) => setControls((previous) => ({ ...previous, ...patch }));

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Delivery-fleet allocation engine</h1>
          <p className="muted">
            Four strategies on one fleet, one composite objective, every decision explained. Offline
            and deterministic: seeded synthetic books, committed RL weights, no external services.
          </p>
        </div>
        <div className="topbar-meta">
          {health ? (
            <>
              <span className="chip">api v{health.version}</span>
              <span className="chip">
                {health.algorithms.length} strategies ready
                {Object.keys(health.unavailable || {}).length
                  ? ` · ${Object.keys(health.unavailable).length} unavailable`
                  : ""}
              </span>
            </>
          ) : null}
        </div>
      </header>

      <ControlBar
        presets={presets}
        cities={health ? health.cities : ["bengaluru"]}
        algorithms={algorithms}
        controls={controls}
        onChange={change}
        onRun={run}
        busy={busy}
        error={error}
        summary={comparison ? comparison.summary : null}
      />

      <nav className="tabs">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className={tab === entry.id ? "tab active" : "tab"}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      {comparison ? (
        <main>
          {tab === "compare" ? (
            <div className="grid">
              <div className="card">
                <h2>
                  Winner: {comparison.winner}
                  <span className="muted small"> ranked on the composite objective</span>
                </h2>
                <ObjectiveChart results={plans} winner={comparison.winner} />
                <MetricTable
                  results={plans}
                  winner={comparison.winner}
                  selected={selectedPlan ? selectedPlan.algorithm : null}
                  onSelect={setSelected}
                />
              </div>
              <div className="card">
                <h2>
                  Routes — {selectedPlan ? selectedPlan.label : "no plan"}
                  {selectedPlan ? (
                    <span className="muted small">
                      objective {formatNumber(selectedPlan.metrics.objective, 1)} ·{" "}
                      {formatNumber(selectedPlan.metrics.coverage_pct, 1)}% covered
                    </span>
                  ) : null}
                </h2>
                <MapView
                  scenario={comparison.scenario}
                  plan={selectedPlan}
                  selectedAlgorithm={selectedPlan ? selectedPlan.algorithm : null}
                />
              </div>
            </div>
          ) : null}

          {tab === "plan" ? (
            <div className="card">
              <PlanPanel scenario={comparison.scenario} plan={selectedPlan} />
            </div>
          ) : null}

          {tab === "policy" ? (
            <div className="card">
              <PolicyPanel policy={policy} error={policyError} />
            </div>
          ) : null}

          {tab === "method" ? <MethodPanel comparison={comparison} /> : null}
        </main>
      ) : (
        <div className="placeholder">{busy ? "Allocating…" : "Loading the strategy catalogue…"}</div>
      )}
    </div>
  );
}
