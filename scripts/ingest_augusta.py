"""Build the Augusta National course model from real, citable sources.

Sources
-------
1. Geometry (tee / aim point / pin lat-lon for all 18 holes)
     provisualizer.com/3dlink.php?id=1 -> 3dplanner.php, which emits the
     decoded coordinates inline as setCourseHole*(...) calls.
2. Par and official yardage
     PGA Tour API via pgatourPY, pga_course_stats("R2025014").
     Also used to VALIDATE the geometry: route length through the aim points
     should reproduce the card yardage.
3. Elevation
     USGS Elevation Point Query Service (epqs.nationalmap.gov), 1/3 arc-second
     3DEP data. Free, no key. This is what makes Augusta's ~175 ft of relief
     enter the physics properly rather than as a fudge factor.

Run:  python scripts/ingest_augusta.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "caddie" / "course" / "data" / "augusta_national.json"
CACHE = ROOT / "scripts" / ".cache"
CACHE.mkdir(exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

OFFICIAL_PAR = [4, 5, 4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4]
OFFICIAL_YARDAGE = [445, 585, 350, 240, 495, 180, 450, 570, 460,
                    495, 520, 155, 545, 440, 550, 170, 440, 465]
HOLE_NAMES = [
    "Tea Olive", "Pink Dogwood", "Flowering Peach", "Flowering Crab Apple",
    "Magnolia", "Juniper", "Pampas", "Yellow Jasmine", "Carolina Cherry",
    "Camellia", "White Dogwood", "Golden Bell", "Azalea", "Chinese Fir",
    "Firethorn", "Redbud", "Nandina", "Holly",
]

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
YARD_M = 0.9144


# --------------------------------------------------------------------------
# Geodesy
# --------------------------------------------------------------------------
def geodesic_m(lat1, lon1, lat2, lon2) -> float:
    """Distance on the WGS84 ellipsoid, metres.

    Over golf-hole distances an equirectangular approximation with the proper
    local radii of curvature is accurate to well under a centimetre, and is
    numerically better behaved than Vincenty for near-coincident points.
    """
    lat_m = math.radians((lat1 + lat2) / 2.0)
    e2 = 2 * WGS84_F - WGS84_F**2
    sin_lat = math.sin(lat_m)
    denom = 1 - e2 * sin_lat**2
    # Meridional and normal radii of curvature at the mean latitude.
    m_per_deg_lat = math.pi / 180.0 * WGS84_A * (1 - e2) / denom**1.5
    m_per_deg_lon = math.pi / 180.0 * WGS84_A * math.cos(lat_m) / math.sqrt(denom)
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    return math.hypot(dx, dy)


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing, degrees clockwise from true north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


# --------------------------------------------------------------------------
# 1. Geometry
# --------------------------------------------------------------------------
def fetch_geometry() -> dict:
    cache_file = CACHE / "provisualizer_augusta.html"
    if cache_file.exists():
        html = cache_file.read_text(encoding="utf-8", errors="replace")
    else:
        s = requests.Session()
        s.headers.update(UA)
        r = s.get("https://www.provisualizer.com/3dlink.php?id=1", timeout=60)
        r.raise_for_status()
        html = r.text
        cache_file.write_text(html, encoding="utf-8")

    pat_tee = re.compile(r"setCourseHoleTee(Lat|Lon)\(0,(\d+),(-?[\d.]+)\)")
    pat_pin = re.compile(r"setCourseHolePin(Lat|Lon)\(0,(\d+),(-?[\d.]+)\)")
    pat_tgt = re.compile(r"setCourseHoleTarget(Lat|Lon)\(0,(\d+),(\d+),(-?[\d.]+)\)")

    holes: dict[int, dict] = {h: {"targets": {}} for h in range(1, 19)}

    for m in pat_tee.finditer(html):
        axis, hole, val = m.group(1), int(m.group(2)), float(m.group(3))
        holes[hole].setdefault("tee", {})[axis.lower()] = val
    for m in pat_pin.finditer(html):
        axis, hole, val = m.group(1), int(m.group(2)), float(m.group(3))
        holes[hole].setdefault("pin", {})[axis.lower()] = val
    for m in pat_tgt.finditer(html):
        axis, hole, idx, val = (m.group(1), int(m.group(2)),
                                int(m.group(3)), float(m.group(4)))
        holes[hole]["targets"].setdefault(idx, {})[axis.lower()] = val

    result = {}
    for h in range(1, 19):
        d = holes[h]
        if "tee" not in d or "pin" not in d:
            raise RuntimeError(f"hole {h}: missing tee or pin in source")
        targets = [d["targets"][k] for k in sorted(d["targets"])]
        result[h] = {
            "tee": (d["tee"]["lat"], d["tee"]["lon"]),
            "targets": [(t["lat"], t["lon"]) for t in targets],
            "pin": (d["pin"]["lat"], d["pin"]["lon"]),
        }
    return result


# --------------------------------------------------------------------------
# 3. Elevation (USGS 3DEP)
# --------------------------------------------------------------------------
def fetch_elevations(points: list[tuple[float, float]]) -> dict:
    """Query USGS EPQS for each unique point. Cached to disk."""
    cache_file = CACHE / "usgs_elevation.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    session = requests.Session()
    session.headers.update(UA)
    missing = [p for p in points if f"{p[0]:.7f},{p[1]:.7f}" not in cache]
    if missing:
        print(f"  querying USGS EPQS for {len(missing)} points "
              f"({len(points) - len(missing)} cached) ...")
    for i, (lat, lon) in enumerate(missing, 1):
        key = f"{lat:.7f},{lon:.7f}"
        url = (
            "https://epqs.nationalmap.gov/v1/json"
            f"?x={lon}&y={lat}&units=Meters&wkid=4326&includeDate=False"
        )
        for attempt in range(4):
            try:
                r = session.get(url, timeout=45)
                if r.status_code == 200 and r.text.strip():
                    cache[key] = float(r.json()["value"])
                    break
            except Exception as exc:  # noqa: BLE001 - report and retry
                if attempt == 3:
                    print(f"    ! {key}: {type(exc).__name__}: {exc}")
            time.sleep(1.5 * (attempt + 1))
        else:
            cache[key] = None
        if i % 10 == 0:
            print(f"    {i}/{len(missing)}")
            cache_file.write_text(json.dumps(cache, indent=1))
    cache_file.write_text(json.dumps(cache, indent=1))
    return cache


# --------------------------------------------------------------------------
def main() -> None:
    print("1. fetching geometry from provisualizer ...")
    geom = fetch_geometry()
    print(f"   got {len(geom)} holes")

    # Validate: route length through aim points vs official card yardage.
    print()
    print("2. validating geometry against official PGA Tour yardages")
    print(f"   {'hole':>4} {'par':>4} {'card':>6} {'straight':>9} {'route':>7} "
          f"{'route-card':>11} {'aim pts':>8}")
    print("   " + "-" * 60)
    rows = []
    for h in range(1, 19):
        g = geom[h]
        pts = [g["tee"], *g["targets"], g["pin"]]
        route = sum(
            geodesic_m(*a, *b) for a, b in zip(pts, pts[1:])
        ) / YARD_M
        straight = geodesic_m(*g["tee"], *g["pin"]) / YARD_M
        card = OFFICIAL_YARDAGE[h - 1]
        rows.append((h, card, straight, route))
        print(f"   {h:>4} {OFFICIAL_PAR[h-1]:>4} {card:>6} {straight:9.1f} "
              f"{route:7.1f} {route - card:+11.1f} {len(g['targets']):>8}")

    route_err = [abs(r[3] - r[1]) for r in rows]
    straight_err = [abs(r[2] - r[1]) for r in rows]
    print(f"   route   vs card: mean |err| {sum(route_err)/18:5.1f} yd, "
          f"max {max(route_err):5.1f} yd")
    print(f"   straight vs card: mean |err| {sum(straight_err)/18:5.1f} yd, "
          f"max {max(straight_err):5.1f} yd")

    print()
    print("3. fetching real elevation from USGS 3DEP ...")
    all_pts = []
    for h in range(1, 19):
        g = geom[h]
        all_pts.extend([g["tee"], *g["targets"], g["pin"]])
    elev = fetch_elevations(all_pts)

    def elev_of(p):
        return elev.get(f"{p[0]:.7f},{p[1]:.7f}")

    print()
    print("4. elevation profile (tee -> pin)")
    print(f"   {'hole':>4} {'tee m':>7} {'pin m':>7} {'rise m':>7} {'rise ft':>8}")
    print("   " + "-" * 40)
    holes_out = []
    for h in range(1, 19):
        g = geom[h]
        te, pe = elev_of(g["tee"]), elev_of(g["pin"])
        rise = None if (te is None or pe is None) else pe - te
        print(f"   {h:>4} {te if te is None else f'{te:7.1f}'} "
              f"{pe if pe is None else f'{pe:7.1f}'} "
              f"{'   n/a' if rise is None else f'{rise:7.1f}'} "
              f"{'   n/a' if rise is None else f'{rise / 0.3048:8.1f}'}")

        pts = [g["tee"], *g["targets"], g["pin"]]
        route_yd = sum(geodesic_m(*a, *b) for a, b in zip(pts, pts[1:])) / YARD_M
        holes_out.append({
            "number": h,
            "name": HOLE_NAMES[h - 1],
            "par": OFFICIAL_PAR[h - 1],
            "card_yardage": OFFICIAL_YARDAGE[h - 1],
            "tee": {"lat": g["tee"][0], "lon": g["tee"][1], "elevation_m": te},
            "aim_points": [
                {"lat": t[0], "lon": t[1], "elevation_m": elev_of(t)}
                for t in g["targets"]
            ],
            "pin": {"lat": g["pin"][0], "lon": g["pin"][1], "elevation_m": pe},
            "derived": {
                "straight_tee_to_pin_yd": round(
                    geodesic_m(*g["tee"], *g["pin"]) / YARD_M, 1),
                "route_yd": round(route_yd, 1),
                "tee_to_pin_bearing_deg": round(
                    bearing_deg(*g["tee"], *g["pin"]), 2),
                "elevation_change_m": None if rise is None else round(rise, 2),
            },
        })

    doc = {
        "course": "Augusta National Golf Club",
        "city": "Augusta",
        "state": "GA",
        "par": sum(OFFICIAL_PAR),
        "card_yardage_total": sum(OFFICIAL_YARDAGE),
        "sources": {
            "geometry": {
                "url": "https://www.provisualizer.com/3dlink.php?id=1",
                "what": "tee / aim point / pin lat-lon per hole",
                "confidence": "single third-party source; validated against "
                              "official yardages (see validation report)",
            },
            "par_and_yardage": {
                "url": "PGA Tour API via pgatourPY pga_course_stats('R2025014')",
                "what": "official par and card yardage, 2025 Masters",
                "confidence": "authoritative",
            },
            "elevation": {
                "url": "https://epqs.nationalmap.gov/v1/json",
                "what": "USGS 3DEP point elevations, metres",
                "confidence": "authoritative bare-earth DEM (~1-3 m vertical); "
                              "NOT a green-surface contour model",
            },
        },
        "known_gaps": [
            "No green surface contours, bunker polygons, fairway boundaries, "
            "or tree positions. Only point geometry is available.",
            "No shot-tracking data exists for Augusta in the PGA Tour API "
            "(ShotLink is not deployed at the Masters), so dispersion and "
            "strokes-gained baselines must come from other venues.",
            "Pin position is a single representative point, not the four "
            "daily Masters pin placements.",
        ],
        "holes": holes_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print()
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
