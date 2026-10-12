/**
 * Thin fetch wrappers around the allocation API.
 *
 * Every call surfaces the backend's error message (FastAPI puts it in `detail`)
 * so the UI can show e.g. "strategy 'exact' is unavailable: exhaustive search is
 * limited to 6 requests" instead of a generic failure.
 */

async function request(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (error) {
      payload = { detail: text.slice(0, 400) };
    }
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export function fetchHealth() {
  return request("/api/health");
}

export function fetchAlgorithms() {
  return request("/api/algorithms");
}

export function fetchPresets() {
  return request("/api/scenario/presets");
}

export function fetchPolicy(includeCurve = true) {
  return request(`/api/rl/policy?include_curve=${includeCurve ? "true" : "false"}`);
}

export function compareStrategies(body) {
  return request("/api/compare", { method: "POST", body: JSON.stringify(body) });
}

export function allocate(body) {
  return request("/api/allocate", { method: "POST", body: JSON.stringify(body) });
}

export function buildScenario(body) {
  return request("/api/scenario", { method: "POST", body: JSON.stringify(body) });
}
