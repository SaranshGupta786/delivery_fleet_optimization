import { formatNumber } from "../lib/format.js";

function Curve({ series, width = 660, height = 140, padding = 10 }) {
  const values = series.flatMap((entry) => entry.values).filter((value) => Number.isFinite(value));
  if (values.length < 2) return <div className="placeholder small">not enough data to plot</div>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const toPoints = (points) =>
    points
      .map((value, index) => {
        const x = padding + (index / Math.max(1, points.length - 1)) * (width - 2 * padding);
        const y = height - padding - ((value - min) / span) * (height - 2 * padding);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="curve" role="img">
      <rect x="0" y="0" width={width} height={height} className="curve-bg" />
      {series.map((entry) => (
        <polyline
          key={entry.label}
          points={toPoints(entry.values)}
          fill="none"
          stroke={entry.color}
          strokeWidth={entry.width || 1.6}
          strokeOpacity={entry.opacity || 1}
        />
      ))}
      <text x={padding + 2} y={13} className="curve-label">
        {formatNumber(max, 0)}
      </text>
      <text x={padding + 2} y={height - 3} className="curve-label">
        {formatNumber(min, 0)}
      </text>
    </svg>
  );
}

function BarRow({ label, value, max, color, highlight }) {
  return (
    <div className="chart-row">
      <div className="chart-label">
        <span className={highlight ? "chip chip-winner" : "chip"}>{label}</span>
        <span className="muted">{formatNumber(value, 1)}</span>
      </div>
      <div className="chart-track" style={{ width: `${Math.max(2, (value / max) * 100)}%` }}>
        <span className="chart-seg" style={{ width: "100%", background: color }} />
      </div>
    </div>
  );
}

const COMPARISON_COLORS = {
  rl: "#5eead4",
  regret: "#93c5fd",
  greedy: "#fcd34d",
  hungarian: "#fca5a5",
  exact: "#c4b5fd",
};

/** The RL transparency panel: weights, training curves and the holdout audit. */
export default function PolicyPanel({ policy, error }) {
  if (error) {
    return (
      <div className="placeholder">
        <strong>The learned strategy is unavailable in this deployment.</strong>
        <p className="muted small">{error}</p>
      </div>
    );
  }
  if (!policy) return <div className="placeholder">Loading the policy artifact…</div>;

  const metadata = policy.metadata || {};
  const comparison = metadata.holdout_comparison || {};
  const comparisonRows = Object.entries(comparison).filter(([, value]) => typeof value === "object");
  const maxComparison = Math.max(...comparisonRows.map(([, value]) => value.mean), 1);
  const validation = metadata.validation || [];
  const maxWeight = Math.max(...policy.weights.map((row) => row.abs_weight), 1e-6);

  return (
    <div className="policy">
      <div className="chips">
        <span className="chip">{metadata.train_episodes || "?"} training episodes</span>
        <span className="chip">seed {metadata.train_config ? metadata.train_config.seed : "?"}</span>
        <span className="chip">{formatNumber(metadata.train_seconds, 0)} s to train</span>
        <span className="chip">{policy.weights.length} weights</span>
        <span className="chip">numpy {metadata.numpy_version || "?"}</span>
      </div>

      <p className="muted small">
        Policy: {metadata.policy || "linear softmax over feasible (request, vehicle) actions"}. Reward
        identity: {metadata.reward_identity || "episode return equals -objective"}. Trained{" "}
        {metadata.trained_at_utc || "?"}. Inference is a deterministic argmax - nothing is trained at
        runtime, and the weights ship as a JSON artifact.
      </p>
      <p className="muted small">
        artifact: <code>{policy.path}</code> (override with <code>RL_POLICY_PATH</code>)
      </p>

      <h4>What the policy learned</h4>
      <table className="table weights">
        <thead>
          <tr>
            <th scope="col">Feature</th>
            <th scope="col">Weight</th>
            <th scope="col">Magnitude</th>
          </tr>
        </thead>
        <tbody>
          {policy.weights.map((row) => (
            <tr key={row.feature}>
              <th scope="row">{row.feature}</th>
              <td className={row.weight >= 0 ? "positive" : "negative"}>{row.weight.toFixed(3)}</td>
              <td>
                <span
                  className="weight-bar"
                  style={{
                    width: `${(row.abs_weight / maxWeight) * 100}%`,
                    background: row.weight >= 0 ? "#5eead4" : "#fca5a5",
                  }}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Training return per episode (higher is better)</h4>
      <Curve
        series={[
          { label: "raw", values: metadata.curve || [], color: "#64748b", width: 1 },
          { label: "smoothed", values: metadata.curve_smoothed || [], color: "#5eead4", width: 2.2 },
        ]}
      />

      <h4>Holdout objective at each checkpoint (lower is better)</h4>
      <Curve
        series={[
          {
            label: "mean objective",
            values: validation.map((entry) => entry.mean_objective),
            color: "#93c5fd",
            width: 2.2,
          },
        ]}
      />

      <h4>Holdout audit - every strategy on the same unseen books</h4>
      {comparisonRows.map(([name, value]) => (
        <BarRow
          key={name}
          label={name}
          value={value.mean}
          max={maxComparison}
          color={COMPARISON_COLORS[name] || "#c4b5fd"}
          highlight={comparison.best === name}
        />
      ))}
      <p className="muted small">
        Preset <code>{metadata.holdout_preset || "small"}</code> over{" "}
        {comparisonRows.length ? comparisonRows[0][1].values.length : 0} unseen books. Best on
        holdout: <strong>{comparison.best || "?"}</strong>.
      </p>
    </div>
  );
}
