"""Route feasibility evaluation and best-insertion search.

Hard constraints - a violation makes an allocation invalid:

* capability    - the request's required capabilities must be a subset of the
                  vehicle's capabilities;
* capacity      - cumulative demand on the route must stay inside the vehicle's
                  kg and m3 capacity (everything is loaded at the depot);
* time window   - service must start no later than ``window_close_min + grace``;
* shift         - the route must finish inside the vehicle's shift window;
* route length  - route duration must not exceed ``max_route_minutes``.

Soft behaviour: arriving before ``window_open_min`` means waiting (allowed, the
wait counts towards route duration) and arriving after ``window_close_min`` but
within ``grace_minutes`` is tolerated and reported as lateness. Leaving a
request unassigned is always legal and is priced by the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .domain import DEFAULT_GRACE_MINUTES, DeliveryRequest, Vehicle
from .geo import haversine_km

EPS = 1e-9


@dataclass(frozen=True)
class Stop:
    """One planned visit; carries everything the UI needs to justify a choice."""

    request_id: str
    leg_km: float
    arrival_min: float
    service_start_min: float
    service_end_min: float
    waiting_min: float
    lateness_min: float
    load_kg: float
    load_m3: float

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "leg_km": round(self.leg_km, 3),
            "arrival_min": round(self.arrival_min, 2),
            "service_start_min": round(self.service_start_min, 2),
            "service_end_min": round(self.service_end_min, 2),
            "waiting_min": round(self.waiting_min, 2),
            "lateness_min": round(self.lateness_min, 2),
            "load_kg": round(self.load_kg, 2),
            "load_m3": round(self.load_m3, 4),
        }


@dataclass(frozen=True)
class RouteEvaluation:
    """Result of simulating one vehicle route end to end."""

    vehicle_id: str
    stops: tuple
    feasible: bool
    violations: tuple
    distance_km: float
    travel_minutes: float
    waiting_minutes: float
    lateness_minutes: float
    service_minutes: float
    finish_min: float
    peak_load_kg: float
    peak_load_m3: float

    @property
    def stop_count(self) -> int:
        return len(self.stops)

    @property
    def busy_minutes(self) -> float:
        """On duty time: ``finish_min - shift_start_min`` (travel + wait + service)."""
        return self.travel_minutes + self.waiting_minutes + self.service_minutes

    def to_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "stop_count": self.stop_count,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "distance_km": round(self.distance_km, 3),
            "travel_minutes": round(self.travel_minutes, 2),
            "waiting_minutes": round(self.waiting_minutes, 2),
            "lateness_minutes": round(self.lateness_minutes, 2),
            "service_minutes": round(self.service_minutes, 2),
            "busy_minutes": round(self.busy_minutes, 2),
            "finish_min": round(self.finish_min, 2),
            "peak_load_kg": round(self.peak_load_kg, 2),
            "peak_load_m3": round(self.peak_load_m3, 4),
            "stops": [stop.to_dict() for stop in self.stops],
        }


def capability_ok(vehicle: Vehicle, request: DeliveryRequest) -> bool:
    """Cheap hard filter, applied before any geometry is computed."""
    return vehicle.supports(request)


def evaluate_route(
    vehicle: Vehicle,
    requests: Sequence[DeliveryRequest],
    grace_minutes: float = DEFAULT_GRACE_MINUTES,
) -> RouteEvaluation:
    """Simulate ``vehicle`` visiting ``requests`` in the given order."""
    clock = vehicle.shift_start_min
    position = vehicle.depot
    distance = travel = waiting = lateness = service_total = 0.0
    load_kg = load_m3 = 0.0
    violations: list = []
    stops: list = []

    for request in requests:
        leg_km = haversine_km(position, request.location)
        leg_min = leg_km / vehicle.speed_kmh * 60.0
        arrival = clock + leg_min
        if not vehicle.supports(request):
            violations.append("capability:%s" % request.id)
        load_kg += request.demand_kg
        load_m3 += request.volume_m3
        if load_kg > vehicle.capacity_kg + EPS:
            violations.append("capacity_kg:%s" % request.id)
        if load_m3 > vehicle.capacity_m3 + EPS:
            violations.append("capacity_m3:%s" % request.id)
        start = max(arrival, request.window_open_min)
        stop_lateness = max(0.0, arrival - request.window_close_min)
        if arrival > request.window_close_min + grace_minutes + EPS:
            violations.append("window:%s" % request.id)
        end = start + request.service_minutes
        stops.append(
            Stop(
                request_id=request.id,
                leg_km=leg_km,
                arrival_min=arrival,
                service_start_min=start,
                service_end_min=end,
                waiting_min=start - arrival,
                lateness_min=stop_lateness,
                load_kg=load_kg,
                load_m3=load_m3,
            )
        )
        clock = end
        position = request.location
        distance += leg_km
        travel += leg_min
        waiting += start - arrival
        lateness += stop_lateness
        service_total += request.service_minutes

    if clock > vehicle.shift_end_min + EPS:
        violations.append("shift_end")
    if travel + waiting + service_total > vehicle.max_route_minutes + EPS:
        violations.append("route_length")

    return RouteEvaluation(
        vehicle_id=vehicle.id,
        stops=tuple(stops),
        feasible=not violations,
        violations=tuple(violations),
        distance_km=distance,
        travel_minutes=travel,
        waiting_minutes=waiting,
        lateness_minutes=lateness,
        service_minutes=service_total,
        finish_min=clock,
        peak_load_kg=load_kg,
        peak_load_m3=load_m3,
    )


def route_distance_cost(vehicle: Vehicle, evaluation: RouteEvaluation) -> float:
    """Distance component of the objective for a single route."""
    return evaluation.distance_km * vehicle.cost_per_km


@dataclass(frozen=True)
class Insertion:
    """A candidate position for a request inside an existing route."""

    vehicle_id: str
    position: int
    delta_km: float
    leg_in_km: float
    leg_out_km: float
    arrival_min: float
    service_start_min: float
    lateness_min: float
    lateness_delta_min: float
    waiting_delta_min: float
    finish_min: float
    cost: float

    def to_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id,
            "position": self.position,
            "delta_km": round(self.delta_km, 3),
            "leg_in_km": round(self.leg_in_km, 3),
            "leg_out_km": round(self.leg_out_km, 3),
            "arrival_min": round(self.arrival_min, 2),
            "lateness_min": round(self.lateness_min, 2),
            "lateness_delta_min": round(self.lateness_delta_min, 2),
            "waiting_delta_min": round(self.waiting_delta_min, 2),
            "finish_min": round(self.finish_min, 2),
            "cost": round(self.cost, 3),
        }


def _suffix_max(values: Sequence[float]) -> list:
    """``out[i] = max(values[i:])``; empty input yields an empty list."""
    out = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = max(running, values[index])
        out[index] = running
    return out


def best_insertion(
    vehicle: Vehicle,
    route: Sequence[DeliveryRequest],
    request: DeliveryRequest,
    grace_minutes: float = DEFAULT_GRACE_MINUTES,
    base: Optional[RouteEvaluation] = None,
    key: Optional[Callable[[Insertion], tuple]] = None,
) -> Optional[Insertion]:
    """Cheapest feasible position for ``request`` in ``route``, else ``None``.

    ``route`` is the vehicle's current ordered request list and ``base`` its
    cached :class:`RouteEvaluation` (recomputed when omitted) so that callers
    can reuse one evaluation across many candidate requests. Each position is
    checked in O(1): distance and capacity deltas are closed form, and the time
    propagation through the tail stops as soon as existing waiting absorbs the
    delay (classic forward time slack), so long routes stay cheap to score.
    """
    if not vehicle.supports(request):
        return None
    stop_count = len(route)
    if base is None:
        base = evaluate_route(vehicle, route, grace_minutes)
    if not base.feasible:
        return None

    tail_kg = _suffix_max([stop.load_kg for stop in base.stops])
    tail_m3 = _suffix_max([stop.load_m3 for stop in base.stops])
    rank = key or (lambda ins: (round(ins.cost, 6), ins.position))
    best: Optional[Insertion] = None

    for position in range(stop_count + 1):
        prefix = base.stops[position - 1] if position > 0 else None
        prefix_kg = prefix.load_kg if prefix is not None else 0.0
        prefix_m3 = prefix.load_m3 if prefix is not None else 0.0
        if prefix_kg + request.demand_kg > vehicle.capacity_kg + EPS:
            continue
        if prefix_m3 + request.volume_m3 > vehicle.capacity_m3 + EPS:
            continue
        if position < stop_count:
            if tail_kg[position] + request.demand_kg > vehicle.capacity_kg + EPS:
                continue
            if tail_m3[position] + request.volume_m3 > vehicle.capacity_m3 + EPS:
                continue

        previous_point = vehicle.depot if position == 0 else route[position - 1].location
        leg_in = haversine_km(previous_point, request.location)
        leg_in_minutes = leg_in / vehicle.speed_kmh * 60.0
        previous_end = prefix.service_end_min if prefix is not None else vehicle.shift_start_min
        arrival = previous_end + leg_in_minutes
        if arrival > request.window_close_min + grace_minutes + EPS:
            continue
        service_start = max(arrival, request.window_open_min)
        service_end = service_start + request.service_minutes
        waiting_delta = service_start - arrival
        lateness_delta = max(0.0, arrival - request.window_close_min)
        own_lateness = lateness_delta
        leg_out = 0.0

        # The leg *into* the stop after the insertion point changes identity (it
        # now starts at the new request), so that arrival is re-derived instead
        # of being shifted by a cached delta. The remaining tail is then shifted
        # by ``carry`` until existing waiting absorbs the delay.
        if position < stop_count:
            next_point = route[position].location
            leg_out = haversine_km(request.location, next_point)
            delta_km = leg_in + leg_out - haversine_km(previous_point, next_point)
            shifted_arrival = service_end + leg_out / vehicle.speed_kmh * 60.0
            following = base.stops[position]
            following_request = route[position]
            if shifted_arrival > following_request.window_close_min + grace_minutes + EPS:
                continue
            waiting_delta += (
                max(0.0, following_request.window_open_min - shifted_arrival) - following.waiting_min
            )
            lateness_delta += (
                max(0.0, shifted_arrival - following_request.window_close_min) - following.lateness_min
            )
            carry = (
                max(shifted_arrival, following_request.window_open_min)
                + following_request.service_minutes
                - following.service_end_min
            )
            tail_start = position + 1
        else:
            delta_km = leg_in
            carry = service_end - base.finish_min
            tail_start = stop_count

        feasible = True
        for index in range(tail_start, stop_count):
            if carry <= EPS:
                break
            stop = base.stops[index]
            tail_request = route[index]
            shifted_arrival = stop.arrival_min + carry
            if shifted_arrival > tail_request.window_close_min + grace_minutes + EPS:
                feasible = False
                break
            waiting_delta += max(0.0, tail_request.window_open_min - shifted_arrival) - stop.waiting_min
            lateness_delta += (
                max(0.0, shifted_arrival - tail_request.window_close_min) - stop.lateness_min
            )
            if stop.waiting_min > EPS:
                carry = max(0.0, carry - stop.waiting_min)
        if not feasible:
            continue

        finish = base.finish_min + carry
        if finish > vehicle.shift_end_min + EPS:
            continue
        if finish - vehicle.shift_start_min > vehicle.max_route_minutes + EPS:
            continue

        candidate = Insertion(
            vehicle_id=vehicle.id,
            position=position,
            delta_km=delta_km,
            leg_in_km=leg_in,
            leg_out_km=leg_out,
            arrival_min=arrival,
            service_start_min=service_start,
            lateness_min=own_lateness,
            lateness_delta_min=lateness_delta,
            waiting_delta_min=waiting_delta,
            finish_min=finish,
            cost=delta_km * vehicle.cost_per_km,
        )
        if best is None or rank(candidate) < rank(best):
            best = candidate

    return best
