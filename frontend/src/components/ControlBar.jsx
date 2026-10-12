import { formatNumber, formatPercent } from "../lib/format.js";

/** Scenario knobs, strategy selection and the run button. */
export default function ControlBar({
  presets,
  cities,
  algorithms,
  controls,
  onChange,
  onRun,
  busy,
  error,
  summary,
}) {
  const available = algorithms.filter((entry) => entry.available);

  const toggleAlgorithm = (name) => {
    const selected = controls.selectedAlgorithms.includes(name)
      ? controls.selectedAlgorithms.filter((entry) => entry !== name)
      : [...controls.selectedAlgorithms, name];
    onChange({ selectedAlgorithms: selected });
  };

  return (
    <section className="controls">
      <div className="control-grid">
        <label>
          <span>Scenario preset</span>
          <select
            value={controls.preset}
            onChange={(event) => onChange({ preset: event.target.value })}
          >
            {presets.map((preset) => (
              <option key={preset.name} value={preset.name} title={preset.description}>
                {preset.name} · {preset.vehicles} vehicles / {preset.requests} requests
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Seed</span>
          <span className="inline">
            <input
              type="number"
              min="0"
              value={controls.seed}
              onChange={(event) => onChange({ seed: Number(event.target.value) })}
            />
            <button
              type="button"
              className="ghost"
              onClick={() => onChange({ seed: Math.floor(Math.random() * 100000) })}
            >
              shuffle
            </button>
          </span>
        </label>

        <label>
          <span>City box</span>
          <select value={controls.city} onChange={(event) => onChange({ city: event.target.value })}>
            {cities.map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Fleet capacity / demand</span>
          <input
            type="number"
            step="0.05"
            min="0.3"
            placeholder="preset"
            value={controls.capacityRatio}
            onChange={(event) => onChange({ capacityRatio: event.target.value })}
          />
        </label>

        <label>
          <span>Window width scale</span>
          <input
            type="number"
            step="0.05"
            min="0.2"
            placeholder="preset"
            value={controls.windowScale}
            onChange={(event) => onChange({ windowScale: event.target.value })}
          />
        </label>

        <label>
          <span>Lateness grace (min)</span>
          <input
            type="number"
            step="5"
            min="0"
            value={controls.graceMinutes}
            onChange={(event) => onChange({ graceMinutes: Number(event.target.value) })}
          />
        </label>
      </div>

      <div className="strategies">
        <span className="field-label">Strategies</span>
        {algorithms.map((entry) =>
          entry.available ? (
            <label key={entry.name} className="check" title={entry.description}>
              <input
                type="checkbox"
                checked={controls.selectedAlgorithms.includes(entry.name)}
                onChange={() => toggleAlgorithm(entry.name)}
              />
              <span>
                {entry.name}
                {entry.is_default ? " (default)" : ""}
              </span>
            </label>
          ) : (
            <span key={entry.name} className="check disabled" title={entry.description}>
              {entry.name} unavailable
            </span>
          )
        )}
        <label className="check" title="Apply the shared relocate polish to the learned plan">
          <input
            type="checkbox"
            checked={controls.polishRL}
            onChange={(event) => onChange({ polishRL: event.target.checked })}
          />
          <span>polish the learned plan</span>
        </label>
      </div>

      <div className="run-row">
        <button type="button" className="primary" onClick={onRun} disabled={busy}>
          {busy ? "allocating…" : `Compare ${controls.selectedAlgorithms.length} strategies`}
        </button>
        <span className="muted small">
          {available.length} strategies available
          {algorithms.length !== available.length ? " (some are disabled in this image)" : ""}
        </span>
      </div>

      {error ? <div className="banner error">{error}</div> : null}

      {summary ? (
        <div className="chips">
          <span className="chip">{summary.name}</span>
          <span className="chip">{summary.vehicles} vehicles</span>
          <span className="chip">{summary.requests} requests</span>
          <span className="chip">
            {formatNumber(summary.total_demand_kg, 0)} kg demand ·{" "}
            {formatNumber(summary.total_capacity_kg, 0)} kg capacity (
            {formatPercent(summary.capacity_ratio * 100, 0)})
          </span>
          <span className="chip">grace {formatNumber(summary.grace_minutes, 0)} min</span>
        </div>
      ) : null}
    </section>
  );
}
