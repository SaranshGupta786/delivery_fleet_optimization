/** Formatting helpers and the shared colour palette. */

export const VEHICLE_COLORS = [
  "#5eead4",
  "#93c5fd",
  "#fca5a5",
  "#fcd34d",
  "#c4b5fd",
  "#86efac",
  "#f9a8d4",
  "#7dd3fc",
  "#fdba74",
  "#a3e635",
  "#d8b4fe",
  "#67e8f9",
];

export function vehicleColor(vehicles, vehicleId) {
  const index = vehicles.findIndex((vehicle) => vehicle.id === vehicleId);
  return VEHICLE_COLORS[(index < 0 ? 0 : index) % VEHICLE_COLORS.length];
}

export function formatClock(minutes) {
  if (minutes === null || minutes === undefined || Number.isNaN(minutes)) return "--:--";
  const total = Math.round(minutes);
  const hours = Math.floor(total / 60) % 24;
  const mins = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`;
}

export function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPercent(value, digits = 1) {
  return `${formatNumber(value, digits)}%`;
}

/** Human labels + direction of preference for the comparison table. */
export const METRIC_ROWS = [
  { key: "objective", label: "Objective (lower wins)", digits: 1, lowerIsBetter: true, primary: true },
  { key: "coverage_pct", label: "Requests covered", digits: 1, suffix: "%", lowerIsBetter: false },
  { key: "priority_coverage_pct", label: "Priority-weighted coverage", digits: 1, suffix: "%", lowerIsBetter: false },
  { key: "distance_km", label: "Distance driven", digits: 1, suffix: " km", lowerIsBetter: true },
  { key: "late_requests", label: "Late requests", digits: 0, lowerIsBetter: true },
  { key: "lateness_minutes", label: "Total lateness", digits: 0, suffix: " min", lowerIsBetter: true },
  { key: "max_lateness_min", label: "Worst lateness", digits: 0, suffix: " min", lowerIsBetter: true },
  { key: "unassigned", label: "Refused requests", digits: 0, lowerIsBetter: true },
  { key: "vehicles_used", label: "Vehicles used", digits: 0, lowerIsBetter: true },
  { key: "avg_utilization_pct", label: "Average utilisation", digits: 1, suffix: "%", lowerIsBetter: false },
  { key: "avg_load_pct", label: "Average load factor", digits: 1, suffix: "%", lowerIsBetter: false },
  { key: "makespan_min", label: "Makespan", digits: 0, suffix: " min", lowerIsBetter: true },
  { key: "runtime_ms", label: "Runtime", digits: 1, suffix: " ms", lowerIsBetter: true },
];

export const OBJECTIVE_PARTS = [
  { key: "distance_cost", label: "Distance cost", color: "#5eead4" },
  { key: "lateness_cost", label: "Lateness cost", color: "#fcd34d" },
  { key: "unassigned_cost", label: "Refused demand", color: "#fca5a5" },
  { key: "makespan_cost", label: "Makespan cost", color: "#93c5fd" },
];

export function metricValue(row, plan) {
  if (row.key === "unassigned") return plan.metrics.requests_unassigned;
  if (row.key === "runtime_ms") return plan.runtime_ms;
  return plan.metrics[row.key];
}
