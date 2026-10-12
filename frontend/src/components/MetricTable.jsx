import { formatNumber, METRIC_ROWS, metricValue } from "../lib/format.js";

/** One column per strategy, one row per KPI, best value per row highlighted. */
export default function MetricTable({ results, winner, selected, onSelect }) {
  if (!results || results.length === 0) return null;

  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th scope="col">Metric</th>
            {results.map((plan) => (
              <th
                key={plan.algorithm}
                scope="col"
                className={plan.algorithm === selected ? "col-selected" : ""}
              >
                <button
                  type="button"
                  className="link"
                  onClick={() => onSelect(plan.algorithm)}
                  title={plan.label}
                >
                  {plan.algorithm}
                </button>
                {plan.algorithm === winner ? <span className="badge">best</span> : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map((row) => {
            const values = results.map((plan) => metricValue(row, plan));
            const best = row.lowerIsBetter ? Math.min(...values) : Math.max(...values);
            return (
              <tr key={row.key}>
                <th scope="row">{row.label}</th>
                {results.map((plan, index) => (
                  <td
                    key={plan.algorithm}
                    className={[
                      values[index] === best ? "best" : "",
                      plan.algorithm === selected ? "col-selected" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                  >
                    {formatNumber(values[index], row.digits)}
                    {row.suffix || ""}
                  </td>
                ))}
              </tr>
            );
          })}
          <tr>
            <th scope="row">Hard constraints</th>
            {results.map((plan) => (
              <td key={plan.algorithm} className={plan.metrics.valid ? "best" : "invalid"}>
                {plan.metrics.valid ? "all satisfied" : plan.metrics.hard_violations.join(", ")}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
