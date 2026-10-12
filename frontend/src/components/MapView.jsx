import { useMemo } from "react";

import { formatClock, formatNumber, vehicleColor } from "../lib/format.js";
import { makeProjector } from "../lib/projection.js";

const WIDTH = 820;
const HEIGHT = 540;

/**
 * Offline SVG map: depots, requests and the selected strategy's routes.
 *
 * Hovering a stop shows the arrival time, load and the engine's explanation for
 * that exact decision, so the map doubles as an explainability surface.
 */
export default function MapView({ scenario, plan, selectedAlgorithm }) {
  const geometry = useMemo(() => {
    if (!scenario) return null;
    const points = [
      ...scenario.vehicles.map((vehicle) => vehicle.depot),
      ...scenario.requests.map((request) => request.location),
    ];
    if (points.length === 0) return null;
    const project = makeProjector(points, WIDTH, HEIGHT);

    const assignmentByRequest = new Map();
    const unassignedByRequest = new Map();
    if (plan) {
      plan.assignments.forEach((assignment) => {
        assignmentByRequest.set(assignment.request_id, assignment);
      });
      plan.unassigned.forEach((entry) => unassignedByRequest.set(entry.request_id, entry));
    }

    const routes = scenario.vehicles
      .map((vehicle) => {
        const requestIds = (plan && plan.routes[vehicle.id]) || [];
        const stops = requestIds
          .map((requestId) => scenario.requests.find((request) => request.id === requestId))
          .filter(Boolean);
        const line = [vehicle.depot, ...stops.map((stop) => stop.location)].map((point) =>
          project(point.lat, point.lon)
        );
        return {
          vehicle,
          color: vehicleColor(scenario.vehicles, vehicle.id),
          stops,
          line,
        };
      })
      .filter((route) => route.stops.length > 0);

    return { project, assignmentByRequest, unassignedByRequest, routes };
  }, [scenario, plan]);

  if (!geometry) {
    return <div className="placeholder">Run a comparison to draw the routes.</div>;
  }

  const { project, assignmentByRequest, unassignedByRequest, routes } = geometry;

  return (
    <div className="map">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Scenario map">
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} className="map-bg" />

        {routes.map((route) => (
          <polyline
            key={`line-${route.vehicle.id}`}
            points={route.line.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
            fill="none"
            stroke={route.color}
            strokeWidth="2"
            strokeOpacity="0.75"
          />
        ))}

        {scenario.vehicles.map((vehicle) => {
          const [x, y] = project(vehicle.depot.lat, vehicle.depot.lon);
          const color = vehicleColor(scenario.vehicles, vehicle.id);
          const stopCount = ((plan && plan.routes[vehicle.id]) || []).length;
          return (
            <g key={vehicle.id}>
              <rect
                x={x - 6}
                y={y - 6}
                width="12"
                height="12"
                transform={`rotate(45 ${x} ${y})`}
                fill={color}
                stroke="#0f172a"
                strokeWidth="1.5"
              />
              <text x={x + 10} y={y + 4} className="map-label" fill={color}>
                {vehicle.id} · {stopCount} stops · {formatNumber(vehicle.capacity_kg, 0)} kg
              </text>
              <title>
                {`${vehicle.id}: depot ${vehicle.depot.lat.toFixed(4)}, ${vehicle.depot.lon.toFixed(4)} · shift ${formatClock(
                  vehicle.shift_start_min
                )}-${formatClock(vehicle.shift_end_min)} · ${stopCount} stops · ${
                  vehicle.capabilities.length ? vehicle.capabilities.join(", ") : "no special capability"
                }`}
              </title>
            </g>
          );
        })}

        {scenario.requests.map((request) => {
          const [x, y] = project(request.location.lat, request.location.lon);
          const assignment = assignmentByRequest.get(request.id);
          const refusal = unassignedByRequest.get(request.id);
          const color = assignment ? vehicleColor(scenario.vehicles, assignment.vehicle_id) : "#64748b";
          const radius = 3.4 + Math.sqrt(request.demand_kg) / 2.6;
          return (
            <g key={request.id}>
              <circle
                cx={x}
                cy={y}
                r={radius}
                fill={color}
                fillOpacity={assignment ? 0.95 : 0.35}
                stroke={refusal ? "#f87171" : "#0f172a"}
                strokeWidth={refusal ? 1.8 : 1}
                strokeDasharray={refusal ? "2 2" : undefined}
              />
              <title>
                {[
                  `${request.id} · ${formatNumber(request.demand_kg, 0)} kg · ${request.volume_m3} m3 · priority ${request.priority}`,
                  `window ${formatClock(request.window_open_min)}-${formatClock(request.window_close_min)}`,
                  assignment
                    ? `served by ${assignment.vehicle_id} as stop ${assignment.sequence + 1}, arriving ${assignment.arrival_clock}, load ${formatNumber(assignment.load_pct, 0)}%`
                    : `refused: ${refusal ? refusal.reason : "not part of this plan"}`,
                  assignment ? assignment.reason : "",
                ]
                  .filter(Boolean)
                  .join("\n")}
              </title>
            </g>
          );
        })}
      </svg>

      <div className="map-legend">
        <div className="map-legend-title">
          {selectedAlgorithm ? `${selectedAlgorithm} plan` : "no plan selected"}
        </div>
        {routes.map((route) => (
          <div key={`legend-${route.vehicle.id}`} className="map-legend-row">
            <span className="swatch" style={{ background: route.color }} />
            <span>
              {route.vehicle.id}: {route.stops.length} stops
            </span>
          </div>
        ))}
        <div className="map-legend-row">
          <span className="swatch swatch-refused" />
          <span>
            {plan ? `${plan.unassigned.length} refused` : "refused"} · marker size ~ demand · hover
            a marker for the decision
          </span>
        </div>
        <div className="map-legend-note">
          SVG Web-Mercator fitted to the scenario bounds. Deliberately offline: no tiles, no keys.
        </div>
      </div>
    </div>
  );
}
