"""Geospatial helpers: great-circle distance, travel time and bounding boxes.

Documented assumption (see README): vehicles move along straight geodesic lines
at a constant cruising speed - no live traffic feed and no paid routing service,
so the engine runs fully offline and deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Point:
    """A WGS84 coordinate pair in degrees."""

    lat: float
    lon: float

    def to_dict(self) -> dict:
        return {"lat": round(self.lat, 6), "lon": round(self.lon, 6)}

    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        return cls(lat=float(data["lat"]), lon=float(data["lon"]))


def haversine_km(a: Point, b: Point) -> float:
    """Great-circle distance between two points, in kilometres."""
    lat1, lon1 = math.radians(a.lat), math.radians(a.lon)
    lat2, lon2 = math.radians(b.lat), math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def travel_minutes(a: Point, b: Point, speed_kmh: float) -> float:
    """Travel time in minutes at a constant cruising speed."""
    if speed_kmh <= 0.0:
        raise ValueError("speed_kmh must be positive")
    return haversine_km(a, b) / speed_kmh * 60.0


def bbox(points: Sequence[Point]) -> tuple:
    """Return the ``(south, west, north, east)`` bounding box of the points."""
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return (min(lats), min(lons), max(lats), max(lons))


def mercator(lat: float, lon: float) -> tuple:
    """Web-Mercator projection into the unit square (x right, y down, 0..1)."""
    x = (lon + 180.0) / 360.0
    clamped = max(-85.05112878, min(85.05112878, lat))
    sin_lat = math.sin(math.radians(clamped))
    y = 0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)
    return (x, y)


def route_distance_km(points: Sequence[Point]) -> float:
    """Total distance along an ordered list of points."""
    return sum(haversine_km(points[i - 1], points[i]) for i in range(1, len(points)))
