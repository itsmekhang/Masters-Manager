"""Georeferencing the illustrated hole maps.

The maps in ``hole diagrams/`` (project root) are plan-view illustrations, near enough
orthographic to draw a trajectory on. To do that we need to turn course
coordinates into image pixels, which takes a georeference: two points per hole
whose real-world position we already know from the course model, located by eye
in the image. Two correspondences pin down scale, rotation and translation
exactly, which is all a plan view needs.

The transform is a similarity WITH A REFLECTION, and the reflection is the part
that is easy to get wrong. Local ENU coordinates are right-handed with north up;
image coordinates run y DOWNWARD. So::

    px = A*E + B*N + c1
    py = B*E - A*N + c2

with ``A = s cos(theta)``, ``B = s sin(theta)``. The determinant is -s^2, i.e.
orientation-reversing -- correct for mapping a y-up frame onto a y-down raster.
Fit it with a plain rotation and the map comes out mirrored, which on a dogleg
looks almost plausible and is completely wrong.

ACCURACY, honestly. The two reference pixels per hole are placed by eye on an
*illustration*, not a survey. The tee is usually unambiguous; "the pin" is taken
as the centre of the green, which is the same convention the course model's
single representative pin point uses. Expect the overlay to be good enough to
show which side of a bunker a ball finishes and useless for anything finer.
These drawings are also not guaranteed to be perfectly to scale -- they are
marketing art -- so treat a disagreement between the overlay and the numbers as
the drawing being wrong, not the physics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .model import GeoPoint, local_metres_per_degree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
# Moved out of the caddie/ package and up to the project root.
DIAGRAM_DIR = PROJECT_ROOT / "hole diagrams"


def enu_offset(origin: GeoPoint, point: GeoPoint) -> tuple[float, float]:
    """(east, north) metres of ``point`` relative to ``origin``."""
    m_lat, m_lon = local_metres_per_degree(origin.lat)
    return (
        (point.lon - origin.lon) * m_lon,
        (point.lat - origin.lat) * m_lat,
    )


def geo_from_enu(origin: GeoPoint, east: float, north: float) -> GeoPoint:
    """Inverse of :func:`enu_offset`: (east, north) metres relative to
    ``origin`` -> a GeoPoint, at ``origin``'s elevation (no polygon here --
    water-hazard rings, the main caller -- carries its own height)."""
    m_lat, m_lon = local_metres_per_degree(origin.lat)
    return GeoPoint(
        lat=origin.lat + north / m_lat,
        lon=origin.lon + east / m_lon,
        elevation_m=origin.elevation_m,
    )


@dataclass(frozen=True)
class HoleMap:
    """A hole illustration plus the two reference points that locate it.

    ``tee_uv`` and ``pin_uv`` are NORMALISED image coordinates -- fractions of
    width and height, origin top-left -- so they survive the image being resized
    or re-encoded.
    """

    hole: int
    image: str
    tee_uv: tuple[float, float]
    pin_uv: tuple[float, float]
    green_image: str | None = None
    note: str = ""

    @property
    def image_path(self) -> Path:
        return DIAGRAM_DIR / self.image

    @property
    def green_image_path(self) -> Path | None:
        return None if self.green_image is None else DIAGRAM_DIR / self.green_image

    def exists(self) -> bool:
        return self.image_path.is_file()


class MapProjection:
    """Maps local ENU metres (relative to the hole's tee) onto image pixels.

    Built from a :class:`HoleMap`, the image's pixel size, and the tee and pin
    as the course model knows them.
    """

    def __init__(
        self,
        hole_map: HoleMap,
        width: int,
        height: int,
        tee: GeoPoint,
        pin: GeoPoint,
    ) -> None:
        self.hole_map = hole_map
        self.width = width
        self.height = height
        self.tee = tee

        # Reference pixels.
        px_t = hole_map.tee_uv[0] * width
        py_t = hole_map.tee_uv[1] * height
        px_p = hole_map.pin_uv[0] * width
        py_p = hole_map.pin_uv[1] * height

        # The pin in ENU metres relative to the tee.
        e_p, n_p = self.enu(pin)
        d2 = e_p * e_p + n_p * n_p
        if d2 <= 0:
            raise ValueError(f"hole {hole_map.hole}: tee and pin coincide")

        dpx = px_p - px_t
        dpy = py_p - py_t
        self._a = (e_p * dpx - n_p * dpy) / d2
        self._b = (n_p * dpx + e_p * dpy) / d2
        self._c1 = px_t
        self._c2 = py_t

    def enu(self, point: GeoPoint) -> tuple[float, float]:
        """(east, north) metres of ``point`` relative to the tee."""
        return enu_offset(self.tee, point)

    @property
    def metres_per_pixel(self) -> float:
        return 1.0 / math.hypot(self._a, self._b)

    @property
    def rotation_deg(self) -> float:
        """Map rotation: the compass bearing that points along image +x."""
        return math.degrees(math.atan2(self._a, self._b)) % 360.0

    def to_pixels(self, east: float, north: float) -> tuple[float, float]:
        return (
            self._a * east + self._b * north + self._c1,
            self._b * east - self._a * north + self._c2,
        )

    def geo_to_pixels(self, point: GeoPoint) -> tuple[float, float]:
        return self.to_pixels(*self.enu(point))

    def pixels_to_enu(self, px: float, py: float) -> tuple[float, float]:
        """Inverse of :meth:`to_pixels`: image pixels -> (east, north) metres
        relative to the tee. Used to push colour-segmented artwork (bunkers,
        water, tree cover -- see ``caddie/course/features.py``) back into the
        same real-world frame the rest of the course model uses."""
        dpx = px - self._c1
        dpy = py - self._c2
        det = self._a * self._a + self._b * self._b
        east = (self._a * dpx + self._b * dpy) / det
        north = (self._b * dpx - self._a * dpy) / det
        return (east, north)

    def pixel_dir_to_bearing(self, dpx: float, dpy: float) -> float:
        """Compass bearing (degrees clockwise from north) that an on-image
        direction vector ``(dpx, dpy)`` (image pixels, y downward) points to,
        under this hole's map rotation. Direction only -- no translation, no
        scale -- so it's valid for a vector measured in ANY image sharing this
        map's camera azimuth, not just this one (see
        ``features.extract_slope_arrows``)."""
        det = self._a * self._a + self._b * self._b
        east = (self._a * dpx + self._b * dpy) / det
        north = (self._b * dpx - self._a * dpy) / det
        return math.degrees(math.atan2(east, north)) % 360.0

    def extent_metres(self) -> tuple[float, float]:
        """Image footprint in metres (width, height). A scale sanity check."""
        mpp = self.metres_per_pixel
        return (self.width * mpp, self.height * mpp)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_hole_maps(path: Path | None = None) -> dict[int, HoleMap]:
    """Read the georeference table. Missing file -> empty dict, not an error.

    The overlay is optional decoration: the explorer must still work with no
    diagrams present, so a missing or partial table degrades rather than raises.
    """
    path = path or (DATA_DIR / "hole_maps.json")
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    maps: dict[int, HoleMap] = {}
    for entry in raw.get("maps", []):
        hm = HoleMap(
            hole=entry["hole"],
            image=entry["image"],
            tee_uv=tuple(entry["tee_uv"]),
            pin_uv=tuple(entry["pin_uv"]),
            green_image=entry.get("green_image"),
            note=entry.get("note", ""),
        )
        maps[hm.hole] = hm
    return maps


def load_hole_info(path: Path | None = None) -> dict[int, dict]:
    """Read ``hole diagrams/hole info.txt`` -- par, yardage and scoring difficulty.

    The historical scoring average and difficulty rank are genuinely new
    information: nothing else in this project knows which holes actually play
    hard, only how long they are and how much they climb.
    """
    path = path or (DIAGRAM_DIR / "hole info.txt")
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {h["hole"]: h for h in raw.get("holes", [])}
