/**
 * Web-Mercator projection fitted to the scenario's own bounding box.
 *
 * The map is drawn as plain SVG on purpose: no tile server, no API key and no
 * network access at runtime, which keeps the container reproducible and makes
 * the rendered plan identical every time.
 */

export function mercator(lat, lon) {
  const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat));
  const x = (lon + 180) / 360;
  const sinLat = Math.sin((clamped * Math.PI) / 180);
  const y = 0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI);
  return [x, y];
}

export function boundsOf(points) {
  const lats = points.map((point) => point.lat);
  const lons = points.map((point) => point.lon);
  return {
    south: Math.min(...lats),
    west: Math.min(...lons),
    north: Math.max(...lats),
    east: Math.max(...lons),
  };
}

/** Returns a `(lat, lon) -> [x, y]` projector for a WIDTH x HEIGHT viewport. */
export function makeProjector(points, width, height, padding = 36) {
  const bounds = boundsOf(points);
  const [westX, northY] = mercator(bounds.north, bounds.west);
  const [eastX, southY] = mercator(bounds.south, bounds.east);
  const spanX = Math.max(1e-9, eastX - westX);
  const spanY = Math.max(1e-9, southY - northY);
  const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
  const offsetX = padding + (width - 2 * padding - spanX * scale) / 2;
  const offsetY = padding + (height - 2 * padding - spanY * scale) / 2;

  const project = (lat, lon) => {
    const [mx, my] = mercator(lat, lon);
    return [offsetX + (mx - westX) * scale, offsetY + (my - northY) * scale];
  };
  project.bounds = bounds;
  return project;
}
