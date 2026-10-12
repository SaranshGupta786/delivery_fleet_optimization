import { formatNumber, vehicleColor } from "../lib/format.js";

/** The explainability surface: every stop's reasoning, plus every refusal. */
export default function PlanPanel({ scenario, plan }) {
  if (!scenario || !plan) {
    return <div className="placeholder">Pick a strategy to inspect its plan.</div>;
  }
  const metrics = plan.metrics;
  const perVehicle = metrics.per_vehicle || [];

  return (
    <div className="plan">
      <div className="plan-head">
        <h3>{plan.label}</h3>
        <div className="chips">
          <span className="chip">objective {formatNumber(metrics.objective, 1)}</span>
          <span className="chip">coverage {formatNumber(metrics.coverage_pct, 1)}%</span>
          <span className="chip">late {metrics.late_requests}</span>
          <span className="chip">distance {formatNumber(metrics.distance_km, 1)} km</span>
          <span className="chip">runtime {formatNumber(plan.runtime_ms, 1)} ms</span>
        </div>
      </div>

      <ul className="notes">
        {plan.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>

      <div className="plan-vehicles">
        {scenario.vehicles.map((vehicle) => {
          const color = vehicleColor(scenario.vehicles, vehicle.id);
          const stops = plan.assignments
            .filter((assignment) => assignment.vehicle_id === vehicle.id)
            .sort((left, right) => left.sequence - right.sequence);
          const summary = perVehicle.find((entry) => entry.vehicle_id === vehicle.id);
          return (
            <section key={vehicle.id} className="vehicle-card">
              <header>
                <span className="swatch" style={{ background: color }} />
                <strong>{vehicle.id}</strong>
                <span className="muted">
                  {stops.length} stops
                  {summary
                    ? ` · ${formatNumber(summary.distance_km, 1)} km · ${formatNumber(
                        summary.utilization_pct,
                        0
                      )}% of shift · ${formatNumber(summary.load_pct, 0)}% of capacity`
                    : ""}
                </span>
              </header>
              {stops.length === 0 ? (
                <p className="muted small">idle this plan</p>
              ) : (
                <ol className="stops">
                  {stops.map((assignment) => (
                    <li key={assignment.request_id}>
                      <div className="stop-head">
                        <span className="stop-id">
                          {assignment.sequence + 1}. {assignment.request_id}
                        </span>
                        <span className="muted">
                          {assignment.arrival_clock} · load {formatNumber(assignment.load_pct, 0)}%
                          {assignment.lateness_min > 0
                            ? ` · ${formatNumber(assignment.lateness_min, 0)} min late`
                            : " · on time"}
                        </span>
                      </div>
                      <p className="reason">{assignment.reason}</p>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          );
        })}

        {plan.unassigned.length > 0 ? (
          <section className="vehicle-card refused">
            <header>
              <strong>Refused ({plan.unassigned.length})</strong>
              <span className="muted">no feasible vehicle within the hard constraints</span>
            </header>
            <ul className="stops">
              {plan.unassigned.map((entry) => (
                <li key={entry.request_id}>
                  <div className="stop-head">
                    <span className="stop-id">{entry.request_id}</span>
                    <span className="muted">priority {entry.priority}</span>
                  </div>
                  <p className="reason">{entry.reason}</p>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </div>
  );
}
