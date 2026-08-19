"""Course features extracted from the illustrated hole maps.

``caddie/course/maps.py`` gives us a calibrated pixel <-> real-world transform
per hole (:class:`~caddie.course.maps.MapProjection`, fit from the tee and
pin). This module colour-segments the artwork through that same transform to
recover the things the geometry model doesn't have: bunkers, water, tree
cover and the tee box, as real polygons in metres -- the gap the README
calls out ("No green contours, bunker polygons, fairway boundaries or tree
positions").

Two different pieces of art, two different confidence levels
--------------------------------------------------------------
The full hole map (``hole-map-N.jpg``) has the calibrated projection, so
colour-segmenting it and pushing the result through ``MapProjection`` gives
bunker/water/tree footprints in real ENU metres, with the same "good enough
to show which side of a bunker" caveat as the route overlay in ``maps.py``.

The green close-up (``hole N green.avif``) is a SEPARATE crop with no
independent georeference of its own. We assume -- checked by eye against the
bunker/hazard layout shared by both images -- that it uses the same camera
azimuth as the full map, just cropped and zoomed. Under that assumption an
arrow's on-image direction converts to a real compass bearing the same way
the hole map's rotation does (:meth:`MapProjection.pixel_dir_to_bearing`),
even though we never learn the close-up's scale. So a slope arrow gives a
real "which way is uphill" and a magnitude *relative to the other arrows on
that green* -- not a grade in percent, and not a green polygon in metres.

Everything here is descriptive geometry, same spirit as the rest of the
course package: cheap, honest about its error bars, and meant to be looked
at (``scripts/render_hole_map.py``) before it's trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .maps import HoleMap, MapProjection
from .model import Hole

Polygon = list[tuple[float, float]]
"""A closed ring of (east, north) metres, relative to the hole's tee."""


# ---------------------------------------------------------------------------
# Colour model
# ---------------------------------------------------------------------------
# All thresholds are on HSV of the artwork, hand-tuned against holes 1, 5, 8,
# 12, 16 and 18 (a spread of shaded/unshaded, watered/dry, wooded/open) and
# then sanity-checked on all 18 by area plausibility in
# scripts/check_hole_features.py. They will not be exact on every hole --
# same caveat as everywhere else in this package.

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32) / 255.0
    maxc = a.max(-1)
    minc = a.min(-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = (maxc - r) / np.where(delta > 0, delta, 1)
        gc = (maxc - g) / np.where(delta > 0, delta, 1)
        bc = (maxc - b) / np.where(delta > 0, delta, 1)
    h = np.zeros_like(maxc)
    h = np.where(maxc == r, bc - gc, h)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta > 0, h, 0.0)
    return np.stack([h, s, v], axis=-1)


def bunker_mask(hsv: np.ndarray) -> np.ndarray:
    """Sand: low saturation, bright. Distinct from everything except cart
    paths and the pale tee markers, which are excluded by the caller."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (s < 0.16) & (v > 0.68)


def water_mask(hsv: np.ndarray) -> np.ndarray:
    """Rae's Creek, the ponds: a distinct blue hue absent from grass/sand.

    Sampled directly off the artwork (hole 13's creek, which was badly
    under-detected before this): open, sunlit water sits around h=0.55-0.56,
    s=0.64. Shaded creek -- a narrow ribbon under trees, not the open ponds
    -- drifts down toward h=0.40-0.45 and s as low as 0.17, close enough to
    grass_mask's own range (h 0.16-0.45) that this can't be pushed much
    wider without starting to catch shaded grass instead."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h > 0.42) & (h < 0.68) & (s > 0.17)


def grass_mask(hsv: np.ndarray) -> np.ndarray:
    h, s = hsv[..., 0], hsv[..., 1]
    return (h > 0.16) & (h < 0.45) & (s > 0.10)


def tree_mask(hsv: np.ndarray) -> np.ndarray:
    """Canopy: dark, mottled grass-hue pixels. Local variance (not colour
    alone) is what actually separates a tree crown from a shaded fairway,
    since both can be equally dark."""
    from scipy import ndimage as ndi

    v = hsv[..., 2]
    mean = ndi.uniform_filter(v, 5)
    mean2 = ndi.uniform_filter(v * v, 5)
    tex = np.sqrt(np.clip(mean2 - mean * mean, 0, None))
    return grass_mask(hsv) & (v < 0.60) & (tex > 0.035)


# ---------------------------------------------------------------------------
# Mask -> polygons, via matplotlib's contour tracer (no extra dependency)
# ---------------------------------------------------------------------------
def _mask_to_pixel_polygons(mask: np.ndarray, min_area_px: float) -> list[np.ndarray]:
    if not mask.any():
        return []
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(mask.astype(np.float32), levels=[0.5])
    polys = []
    for seg in cs.allsegs[0]:
        if len(seg) < 4:
            continue
        # shoelace area in pixel^2
        x, y = seg[:, 0], seg[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area >= min_area_px:
            polys.append(seg)
    plt.close(fig)
    return polys


def _simplify(poly: np.ndarray, max_vertices: int = 60) -> np.ndarray:
    if len(poly) <= max_vertices:
        return poly
    idx = np.linspace(0, len(poly) - 1, max_vertices).astype(int)
    return poly[idx]


# ---------------------------------------------------------------------------
# Full hole-map extraction (georeferenced, real metres)
# ---------------------------------------------------------------------------
@dataclass
class MapFeatures:
    bunkers: list[Polygon] = field(default_factory=list)
    water: list[Polygon] = field(default_factory=list)
    trees: list[Polygon] = field(default_factory=list)
    tee_box: Polygon | None = None
    downscale: int = 1


def extract_map_features(
    hole: Hole, hole_map: HoleMap, downscale: int = 4
) -> MapFeatures | None:
    """Colour-segment the full hole-map illustration into real-metre polygons.

    ``downscale`` trades resolution for speed -- the source art is ~5000 px
    wide and a full-resolution contour trace of 18 holes is minutes, not
    seconds. 4x keeps sub-metre-scale bunker detail while running in a
    couple of seconds a hole.
    """
    if not hole_map.exists():
        return None
    im = Image.open(hole_map.image_path).convert("RGB")
    W, H = im.size
    proj = MapProjection(hole_map, W, H, hole.tee, hole.pin)
    small = im.resize((W // downscale, H // downscale), Image.LANCZOS)
    rgb = np.asarray(small)
    hsv = _rgb_to_hsv(rgb)
    mpp = proj.metres_per_pixel * downscale

    def to_enu_polys(mask: np.ndarray, min_area_m2: float, max_area_m2: float,
                      max_n: int | None = None) -> list[Polygon]:
        min_area_px = min_area_m2 / (mpp * mpp)
        max_area_px = max_area_m2 / (mpp * mpp)
        pix_polys = _mask_to_pixel_polygons(mask, min_area_px)
        # The artwork fades to a soft white/transparent edge outside the
        # illustrated corridor (see the raw JPGs) -- bunker/tee colour
        # thresholds catch that fringe as one giant contour. An upper area
        # bound throws it out; a real bunker or a single tree crown never
        # gets anywhere near it.
        pix_polys = [p for p in pix_polys if _poly_area(p) <= max_area_px]
        pix_polys.sort(key=lambda p: -_poly_area(p))
        if max_n is not None:
            pix_polys = pix_polys[:max_n]
        out = []
        for p in pix_polys:
            p = _simplify(p)
            ring = [proj.pixels_to_enu(px * downscale, py * downscale) for px, py in p]
            out.append(ring)
        return out

    bunkers = to_enu_polys(bunker_mask(hsv), min_area_m2=8.0, max_area_m2=800.0)
    # A narrow, shaded creek (hole 13's, worst offender) breaks into a
    # dotted line under color thresholding alone -- tree shadow drops its
    # saturation below what a single hue/sat cutoff can separate from
    # grass. Closing bridges those small gaps back into one shape using
    # the pixels already classified as water, rather than trying to widen
    # the color thresholds further to close them (that starts catching
    # shaded grass instead -- see water_mask).
    #
    # Strong enough closing to bridge the creek's worst gaps also invents
    # small false ponds elsewhere (a single stray shadow pixel, dilated by
    # closing into something just over the area floor below) -- on holes
    # with no water at all. The fix isn't less closing (that just brings
    # the creek gaps back); it's requiring that a surviving blob actually
    # BE mostly real water, not mostly gap-filled by closing. A manufactured
    # false pond is nearly all closing, almost none of it raw pixels; a
    # real creek segment with a few shadow-dropped gaps is still mostly
    # raw. Erosion afterward trims the anti-aliased fringe back down,
    # mainly around the open ponds.
    #
    # Two area floors, not one: a handful of the creek's segments near the
    # green are only a few square metres, entirely raw (never touched by
    # closing) -- real, just small. The single-stray-pixel false positives
    # this same fraction check catches elsewhere top out at 0.3 m^2 once
    # dilated, nowhere near the smallest genuine fragment (5.5 m^2), so a
    # low floor for near-100%-raw blobs is safe: it only ever lets through
    # things that were already water before closing did anything.
    from scipy import ndimage as ndi
    water_raw = water_mask(hsv)
    water_closed = ndi.binary_closing(water_raw, iterations=9)
    water_eroded = ndi.binary_erosion(water_closed, iterations=1)
    labeled, n_components = ndi.label(water_eroded)
    water_mpp = proj.metres_per_pixel * downscale
    water_filtered = np.zeros_like(water_eroded)
    for i in range(1, n_components + 1):
        component = labeled == i
        raw_fraction = water_raw[component].sum() / component.sum()
        area_m2 = component.sum() * water_mpp * water_mpp
        high_confidence = raw_fraction >= 0.9 and area_m2 >= 2.0
        gap_bridged = raw_fraction >= 0.3 and area_m2 >= 20.0
        if high_confidence or gap_bridged:
            water_filtered |= component
    water = to_enu_polys(water_filtered, min_area_m2=1.0, max_area_m2=6000.0)
    trees = to_enu_polys(tree_mask(hsv), min_area_m2=25.0, max_area_m2=600.0, max_n=40)

    tee_box = _extract_tee_box(hsv, proj, hole, downscale)

    return MapFeatures(bunkers=bunkers, water=water, trees=trees, tee_box=tee_box)


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _extract_tee_box(hsv, proj, hole, downscale, search_radius_m: float = 45.0) -> Polygon | None:
    """The tee marker(s): a small pale, low-texture patch near the tee point."""
    from scipy import ndimage as ndi

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pale = (s < 0.22) & (s > 0.03) & (v > 0.68)
    mpp = proj.metres_per_pixel * downscale
    tpx, tpy = proj.geo_to_pixels(hole.tee)
    tpx, tpy = tpx / downscale, tpy / downscale
    r = int(search_radius_m / mpp)
    H, W = v.shape
    y0, y1 = max(0, int(tpy - r)), min(H, int(tpy + r))
    x0, x1 = max(0, int(tpx - r)), min(W, int(tpx + r))
    if y1 <= y0 or x1 <= x0:
        return None
    window = pale[y0:y1, x0:x1]
    lab, n = ndi.label(window, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndi.sum(window, lab, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == best)
    if len(xs) < 6:
        return None
    poly_px = np.stack([xs + x0, ys + y0], axis=1).astype(float)
    hull = _convex_hull(poly_px)
    return [proj.pixels_to_enu(px * downscale, py * downscale) for px, py in hull]


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Small inputs only (tee-box blobs)."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


# ---------------------------------------------------------------------------
# Green close-up: slope arrows (direction is real, scale is not)
# ---------------------------------------------------------------------------
@dataclass
class SlopeArrow:
    u: float
    """Position within the green image, 0..1 left-right."""
    v: float
    """Position within the green image, 0..1 top-bottom."""
    bearing_deg: float
    """Compass bearing the arrow points, i.e. the uphill direction, under the
    shared-orientation assumption documented on this module."""
    relative_length: float
    """Arrow pixel length / longest arrow on this green. A rough proxy for
    "how steep here relative to the rest of this green" -- not a grade."""


def _detect_arrow_components(hsv: np.ndarray):
    from scipy import ndimage as ndi

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (h > 0.11) & (h < 0.19) & (s > 0.75) & (v > 0.70)
    lab, n = ndi.label(yellow, structure=np.ones((3, 3)))
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 12:
            continue
        ys, xs = np.nonzero(m)
        pts = np.stack([xs, ys], axis=1).astype(float)
        centre = pts.mean(0)
        cov = np.cov((pts - centre).T)
        evals, evecs = np.linalg.eigh(cov)
        if evals[0] <= 1e-9 or evals[1] / evals[0] < 4:
            continue  # not elongated enough to be an arrow shaft
        major = evecs[:, 1]
        proj = (pts - centre) @ major
        lo, hi = pts[np.argmin(proj)], pts[np.argmax(proj)]
        # The head is the wider end: compare local blob thickness near each tip.
        def thickness(tip):
            d = np.linalg.norm(pts - tip, axis=1)
            return (d < 4.0).sum()

        head, tail = (hi, lo) if thickness(hi) >= thickness(lo) else (lo, hi)
        length = float(np.linalg.norm(head - tail))
        comps.append(dict(cx=float(centre[0]), cy=float(centre[1]),
                           dx=float(head[0] - tail[0]), dy=float(head[1] - tail[1]),
                           length=length))
    return comps


def extract_slope_arrows(hole_map: HoleMap, proj: MapProjection) -> list[SlopeArrow]:
    """Read the yellow slope arrows off ``hole N green.avif``.

    ``proj`` is the full hole map's calibrated projection. Only its rotation
    (``pixel_dir_to_bearing``) is used -- reused under the shared-orientation
    assumption in this module's docstring, since the green close-up has no
    georeference of its own.
    """
    if hole_map.green_image_path is None or not hole_map.green_image_path.is_file():
        return []
    im = Image.open(hole_map.green_image_path).convert("RGB")
    W, H = im.size
    hsv = _rgb_to_hsv(np.asarray(im))
    comps = _detect_arrow_components(hsv)
    if not comps:
        return []
    max_len = max(c["length"] for c in comps) or 1.0
    arrows = []
    for c in comps:
        bearing = proj.pixel_dir_to_bearing(c["dx"], c["dy"])
        arrows.append(SlopeArrow(
            u=c["cx"] / W, v=c["cy"] / H,
            bearing_deg=bearing,
            relative_length=c["length"] / max_len,
        ))
    return arrows


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class HoleFeatures:
    map: MapFeatures | None
    slope_arrows: list[SlopeArrow] = field(default_factory=list)


def extract_hole_features(hole: Hole, hole_map: HoleMap, downscale: int = 4) -> HoleFeatures:
    mf = extract_map_features(hole, hole_map, downscale=downscale)
    arrows: list[SlopeArrow] = []
    if hole_map.exists():
        im = Image.open(hole_map.image_path)
        proj = MapProjection(hole_map, *im.size, hole.tee, hole.pin)
        arrows = extract_slope_arrows(hole_map, proj)
    return HoleFeatures(map=mf, slope_arrows=arrows)
