import { formatNumber, OBJECTIVE_PARTS } from "../lib/format.js";

/** Stacked bars: what makes up each strategy's objective. */
export default function ObjectiveChart({ results, winner }) {
  if (!results || results.length === 0) return null;
  const max = Math.max(...results.map((plan) => plan.metrics.objective), 1);

  return (
    <div className="chart">
      <div className="chart-title">Objective decomposition (lower is better)</div>
      {results.map((plan) => {
        const total = plan.metrics.objective;
        const parts = plan.metrics.objective_components;
        return (
          <div className="chart-row" key={plan.algorithm}>
            <div className="chart-label">
              <span className={plan.algorithm === winner ? "chip chip-winner" : "chip"}>
                {plan.algorithm}
              </span>
              <span className="muted">{formatNumber(total, 1)}</span>
            </div>
            <div className="chart-track" style={{ width: `${Math.max(2, (total / max) * 100)}%` }}>
              {OBJECTIVE_PARTS.map((part) => {
                const value = parts[part.key] || 0;
                const share = total > 0 ? (value / total) * 100 : 0;
                return (
                  <span
                    key={part.key}
                    className="chart-seg"
                    style={{ width: `${share}%`, background: part.color }}
                    title={`${part.label}: ${formatNumber(value, 1)} (${formatNumber(share, 0)}%)`}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="chart-legend">
        {OBJECTIVE_PARTS.map((part) => (
          <span key={part.key} className="chart-legend-item">
            <span className="swatch" style={{ background: part.color }} />
            {part.label}
          </span>
        ))}
      </div>
    </div>
  );
}
