"""Course geometry model.

Design note: shot frames
------------------------
The physics integrator works in a local Cartesian frame (x downrange along
the target line, y right, z up). The course is stored in geographic
coordinates. :class:`ShotFrame` is the bridge -- it converts a lat/lon/elev
origin and a target bearing into that local frame and back, so a trajectory
can be flown in metres and the landing point resolved back to a place on the
golf course.

We use a local tangent-plane (East-North-Up) projection. Over the ~600 m span
of a golf hole its distortion is far below the accuracy of the underlying
data, and unlike a global projection it introduces no scale factor to
remember.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
E2 = 2 * WGS84_F - WGS84_F**2
YARD_M = 0.9144


def local_metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """(metres per degree latitude, metres per degree longitude) at a latitude."""
    lat = math.radians(lat_deg)
    denom = 1 - E2 * math.sin(lat) ** 2
    m_lat = math.pi / 180.0 * WGS84_A * (1 - E2) / denom**1.5
    m_lon = math.pi / 180.0 * WGS84_A * math.cos(lat) / math.sqrt(denom)
    return m_lat, m_lon


@dataclass(frozen=True)
class GeoPoint:
    """A point on the course."""

    lat: float
    lon: float
    elevation_m: float | None = None

    def distance_to(self, other: "GeoPoint") -> float:
        """Horizontal distance, metres."""
        m_lat, m_lon = local_metres_per_degree((self.lat + other.lat) / 2.0)
        return math.hypot(
            (other.lat - self.lat) * m_lat, (other.lon - self.lon) * m_lon
        )

    def distance_yards_to(self, other: "GeoPoint") -> float:
        return self.distance_to(other) / YARD_M

    def bearing_to(self, other: "GeoPoint") -> float:
        """Initial bearing, degrees clockwise from true north."""
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dl = math.radians(other.lon - self.lon)
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        return math.degrees(math.atan2(y, x)) % 360.0

    def elevation_change_to(self, other: "GeoPoint") -> float | None:
        if self.elevation_m is None or other.elevation_m is None:
            return None
        return other.elevation_m - self.elevation_m

    def plays_like_yards_to(self, other: "GeoPoint", rule_of_thumb: bool = True) -> float:
        """Naive 'plays like' yardage: horizontal + elevation change.

        Included because every caddie book quotes it, and because it is the
        baseline the physics engine should be expected to BEAT. The rule of
        thumb adds one yard per yard of elevation gain. It ignores the fact
        that a downhill shot also has longer to be pushed around by wind and
        lands at a shallower effective angle, which is precisely why we
        integrate the real trajectory against real terrain instead.
        """
        flat = self.distance_yards_to(other)
        dz = self.elevation_change_to(other)
        if dz is None or not rule_of_thumb:
            return flat
        return flat + dz / YARD_M


class ShotFrame:
    """Local ENU-style frame for one shot.

    x -> along ``bearing_deg`` from ``origin``
    y -> 90 degrees clockwise of x (the player's right)
    z -> up, relative to the origin's elevation
    """

    def __init__(self, origin: GeoPoint, bearing_deg: float) -> None:
        self.origin = origin
        self.bearing_deg = bearing_deg
        self._m_lat, self._m_lon = local_metres_per_degree(origin.lat)
        b = math.radians(bearing_deg)
        self._cos_b, self._sin_b = math.cos(b), math.sin(b)

    def to_local(self, point: GeoPoint) -> tuple[float, float, float]:
        north = (point.lat - self.origin.lat) * self._m_lat
        east = (point.lon - self.origin.lon) * self._m_lon
        # Rotate so +x lies along the bearing.
        x = north * self._cos_b + east * self._sin_b
        y = -north * self._sin_b + east * self._cos_b
        z = 0.0
        if point.elevation_m is not None and self.origin.elevation_m is not None:
            z = point.elevation_m - self.origin.elevation_m
        return (x, y, z)

    def to_geo(self, x: float, y: float, z: float = 0.0) -> GeoPoint:
        north = x * self._cos_b - y * self._sin_b
        east = x * self._sin_b + y * self._cos_b
        elev = None
        if self.origin.elevation_m is not None:
            elev = self.origin.elevation_m + z
        return GeoPoint(
            lat=self.origin.lat + north / self._m_lat,
            lon=self.origin.lon + east / self._m_lon,
            elevation_m=elev,
        )


@dataclass
class Hole:
    number: int
    name: str
    par: int
    card_yardage: int
    tee: GeoPoint
    pin: GeoPoint
    aim_points: list[GeoPoint] = field(default_factory=list)

    @property
    def route_points(self) -> list[GeoPoint]:
        return [self.tee, *self.aim_points, self.pin]

    @property
    def route_yards(self) -> float:
        """Playing length along the tee -> aim point -> pin route."""
        pts = self.route_points
        return sum(a.distance_yards_to(b) for a, b in zip(pts, pts[1:]))

    @property
    def straight_yards(self) -> float:
        return self.tee.distance_yards_to(self.pin)

    @property
    def elevation_change_m(self) -> float | None:
        return self.tee.elevation_change_to(self.pin)

    @property
    def dogleg_deg(self) -> float:
        """How much the route bends, degrees. 0 for a straight hole.

        Sum of absolute bearing changes at each aim point -- a compact measure
        of how much the hole denies you a direct line at the pin.
        """
        pts = self.route_points
        if len(pts) < 3:
            return 0.0
        total = 0.0
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            d = (b.bearing_to(c) - a.bearing_to(b) + 180.0) % 360.0 - 180.0
            total += abs(d)
        return total

    def frame_from(self, origin: GeoPoint, target: GeoPoint | None = None) -> ShotFrame:
        """A shot frame aimed from ``origin`` at ``target`` (default: the pin)."""
        return ShotFrame(origin, origin.bearing_to(target or self.pin))

    def tee_frame(self) -> ShotFrame:
        """Frame for the tee shot, aimed at the first aim point if there is one."""
        target = self.aim_points[0] if self.aim_points else self.pin
        return self.frame_from(self.tee, target)

    def describe(self) -> str:
        dz = self.elevation_change_m
        dz_s = "n/a" if dz is None else f"{dz / 0.3048:+.0f} ft"
        return (
            f"#{self.number:2d} {self.name:<22} par {self.par}  "
            f"{self.card_yardage:3d} yd card / {self.route_yards:5.1f} route  "
            f"elev {dz_s:>8}  dogleg {self.dogleg_deg:4.1f}deg"
        )


@dataclass
class Course:
    name: str
    holes: list[Hole]
    par: int
    card_yardage_total: int
    sources: dict = field(default_factory=dict)
    known_gaps: list[str] = field(default_factory=list)

    def hole(self, number: int) -> Hole:
        for h in self.holes:
            if h.number == number:
                return h
        raise KeyError(f"no hole {number}")

    def __iter__(self) -> Iterable[Hole]:
        return iter(self.holes)

    def __len__(self) -> int:
        return len(self.holes)

    def elevation_range_m(self) -> tuple[float, float]:
        elevs = [
            p.elevation_m
            for h in self.holes
            for p in h.route_points
            if p.elevation_m is not None
        ]
        return (min(elevs), max(elevs))

    def summary(self) -> str:
        lines = [
            f"{self.name} -- par {self.par}, {self.card_yardage_total} yd",
        ]
        lo, hi = self.elevation_range_m()
        lines.append(
            f"elevation {lo:.0f}-{hi:.0f} m MSL "
            f"({(hi - lo) / 0.3048:.0f} ft of relief across the property)"
        )
        lines.append("")
        lines.extend(h.describe() for h in self.holes)
        return "\n".join(lines)


def terrain_from_hole(hole: Hole, frame: ShotFrame):
    """A terrain callback for the flight integrator, built from the hole.

    We only have point elevations (tee, aim points, pin), so this does
    inverse-distance-weighted interpolation between them. That is honest about
    the data we have: it captures the big picture -- the 100 ft drop on 10, the
    climb up 18 -- but it is NOT a green contour model and will not tell you
    which way a putt breaks.
    """
    pts = [frame.to_local(p) for p in hole.route_points]

    def terrain(x: float, y: float) -> float:
        num = 0.0
        den = 0.0
        for px, py, pz in pts:
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < 1e-6:
                return pz
            w = 1.0 / d2
            num += w * pz
            den += w
        return num / den if den else 0.0

    return terrain
