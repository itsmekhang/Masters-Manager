"""Build a hole into the running Unreal Editor over its live MCP server.

This is the bridge from the two things this repo already validates -- real
hole geometry (``caddie.course``) and artwork-derived features (bunkers,
water, trees, tee box, green slope arrows -- ``caddie.course.features``) --
into an actual 3D layout in the open Unreal level. It does NOT sculpt a
Landscape or paint foliage; the project has neither the Water plugin nor any
mesh content beyond its sky assets, so every feature is a coloured primitive
(box/cylinder/cone) sized and placed from real numbers. Schematic, not
photoreal -- same honesty rule as the rest of this repo: this is a first
pass meant to be looked at (it screenshots itself) and iterated on.

Requires the Unreal Editor to be running with the ModelContextProtocol
plugin's MCP server reachable (checked via ``.mcp.json`` in the project:
``http://127.0.0.1:8000/mcp``).

Usage:
    python scripts/build_ue_course.py --hole 1
    python scripts/build_ue_course.py --hole 1 2 3 --url http://127.0.0.1:8000/mcp
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caddie.course import load_augusta
from caddie.course.maps import load_hole_maps
from caddie.course.features import extract_hole_features
from caddie.course.model import GeoPoint

YARD_M = 0.9144
M_TO_UU = 100.0  # Unreal: 1 unit = 1 cm


# ---------------------------------------------------------------------------
# MCP client -- minimal JSON-RPC-over-HTTP, no extra dependency
# ---------------------------------------------------------------------------
class MCPError(RuntimeError):
    pass


class UnrealMCP:
    def __init__(self, url: str = "http://127.0.0.1:8000/mcp"):
        self.url = url
        self.session_id: str | None = None
        self._id = 0
        self._initialize()

    def _post(self, payload: dict) -> tuple[dict, dict]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            body = resp.read().decode("utf-8")
        return (json.loads(body) if body.strip() else {}), dict(headers)

    def _initialize(self) -> None:
        self._id += 1
        resp, _ = self._post({
            "jsonrpc": "2.0", "id": self._id, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-caddie-course-builder", "version": "0.1"},
            },
        })
        if "error" in resp:
            raise MCPError(resp["error"])
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call(self, toolset: str | None, tool: str, arguments: dict) -> dict:
        self._id += 1
        params = {"tool_name": tool, "arguments": arguments}
        if toolset is not None:
            params["toolset_name"] = toolset
        resp, _ = self._post({
            "jsonrpc": "2.0", "id": self._id, "method": "tools/call",
            "params": {"name": "call_tool", "arguments": params},
        })
        if "error" in resp:
            raise MCPError(f"{tool}: {resp['error']}")
        content = resp["result"]["content"][0]["text"]
        if resp["result"].get("isError"):
            raise MCPError(f"{tool}: {content}")
        parsed = json.loads(content)
        return parsed.get("returnValue", parsed)


# ---------------------------------------------------------------------------
# Small geometry helpers (plain ENU, no per-hole rotation -- see module docstring)
# ---------------------------------------------------------------------------
def ref(path: str) -> dict:
    return {"refPath": path}


def vec(x: float, y: float, z: float) -> dict:
    return {"x": x, "y": y, "z": z}


def rot(pitch: float, yaw: float, roll: float) -> dict:
    return {"pitch": pitch, "yaw": yaw, "roll": roll}


def xform(loc: tuple[float, float, float], yaw_deg: float = 0.0,
          scale: tuple[float, float, float] = (1, 1, 1)) -> dict:
    return {
        "location": vec(*loc),
        "rotation": rot(0.0, yaw_deg, 0.0),
        "scale": vec(*scale),
    }


def poly_area_centroid(poly: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Shoelace area + a plain vertex-average centroid.

    The textbook shoelace centroid (weighted by cross products, divided by
    area) is numerically unstable on the thin, near-degenerate contours this
    pipeline occasionally produces (the artwork's soft fade-to-white border
    traced as a sliver ring: near-zero net area but vertices spanning the
    whole image) -- dividing by a near-zero area there sent a bunker's
    centroid hundreds of metres off the hole. A plain vertex mean has no such
    division and is accurate enough for placing a representative marker."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if poly[0] == poly[-1] and len(poly) > 1:
        xs, ys = xs[:-1], ys[:-1]
    a = 0.0
    n = len(xs)
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[(i + 1) % n], ys[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    a = abs(a) * 0.5
    return (sum(xs) / n, sum(ys) / n, a)


# ---------------------------------------------------------------------------
# Colours -- one shared parametric material, one instance per feature kind
# ---------------------------------------------------------------------------
COLORS = {
    "Fairway": (0.09, 0.30, 0.10),
    "Rough": (0.24, 0.26, 0.07),
    "Green": (0.15, 0.44, 0.17),
    "Tee": (0.12, 0.36, 0.14),
    "Bunker": (0.86, 0.78, 0.55),
    "Water": (0.04, 0.22, 0.42),
    "TreeCanopy": (0.05, 0.16, 0.055),
    "TreeTrunk": (0.28, 0.18, 0.09),
    "FlagPole": (0.92, 0.92, 0.92),
    "Flag": (0.78, 0.04, 0.04),
    "Cup": (0.02, 0.02, 0.02),
    "SlopeArrow": (0.95, 0.80, 0.05),
}
MAT_FOLDER = "/Game/Golf/Materials"
BASE_MATERIAL = f"{MAT_FOLDER}/M_ColorBase.M_ColorBase"


def ensure_materials(mcp: UnrealMCP) -> dict[str, str]:
    """Create (or reuse) one MaterialInstanceConstant per colour above.

    Assumes ``M_ColorBase`` already exists with a "Color" vector parameter
    wired to base colour -- built once by hand the first time this project's
    MCP server was probed; see the plan / README for the one-off setup.
    """
    out = {}
    for name, (r, g, b) in COLORS.items():
        path = f"{MAT_FOLDER}/MI_{name}.MI_{name}"
        try:
            mcp.call("editor_toolset.toolsets.material_instance.MaterialInstanceTools",
                     "create", {"folder_path": MAT_FOLDER, "asset_name": f"MI_{name}",
                                "parent": ref(BASE_MATERIAL)})
        except MCPError:
            pass  # already exists
        mcp.call("editor_toolset.toolsets.material_instance.MaterialInstanceTools",
                  "set_vector_parameter",
                  {"instance": ref(path), "name": "Color",
                   "value": {"r": r, "g": g, "b": b, "a": 1.0}})
        out[name] = path
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
@dataclass
class Placed:
    actor_path: str


class HoleBuilder:
    # Real Augusta relief is ~50m tee-to-tee (hole 1 near the property's high
    # point, Amen Corner near its low point). There is no sculpted Landscape
    # yet -- the ground is one flat "Floor" actor -- so displaying that relief
    # at true scale means most holes are either buried tens of metres under
    # the floor or floating tens of metres above it (both happened during
    # development). Compressing it keeps hole-to-hole relief as a visual cue
    # -- Amen Corner still reads as the low ground -- without anything
    # floating or sinking relative to the flat floor it has to sit on.
    ELEVATION_COMPRESSION = 0.02

    def __init__(self, mcp: UnrealMCP, materials: dict[str, str], master_origin: GeoPoint,
                 master_elev_m: float):
        self.mcp = mcp
        self.mat = materials
        self.origin = master_origin
        self.origin_elev = master_elev_m
        self._comp_counts: dict[str, int] = {}

    # -- coordinate helpers --------------------------------------------------
    def enu(self, p: GeoPoint) -> tuple[float, float]:
        from caddie.course.maps import enu_offset
        return enu_offset(self.origin, p)

    def terrain_z_m(self, elevation_m: float | None) -> float:
        """Real elevation (metres, absolute) -> compressed world Z (metres),
        relative to ``origin_elev`` (the course's lowest point -- see main())."""
        if elevation_m is None:
            return 0.0
        return (elevation_m - self.origin_elev) * self.ELEVATION_COMPRESSION

    def world(self, p: GeoPoint) -> tuple[float, float, float]:
        e, n = self.enu(p)
        z = self.terrain_z_m(p.elevation_m)
        return (e * M_TO_UU, n * M_TO_UU, z * M_TO_UU)

    def world_en(self, east_m: float, north_m: float, hole_tee: GeoPoint,
                 dz_m: float = 0.0) -> tuple[float, float, float]:
        """A point given as metres (east, north) relative to a HOLE's tee
        (the frame caddie.course.features polygons are in) -> world cm.

        Features (bunkers/water/trees) don't carry their own elevation, so
        they sit at their hole's tee elevation plus a small ``dz_m`` nudge
        (sink a bunker, lift a tree) -- an approximation, but one that keeps
        them next to their own fairway instead of floating at a fixed Z
        unrelated to the hole's real height."""
        te, tn = self.enu(hole_tee)
        z = self.terrain_z_m(hole_tee.elevation_m) + dz_m
        return ((te + east_m) * M_TO_UU, (tn + north_m) * M_TO_UU, z * M_TO_UU)

    # -- scene plumbing -------------------------------------------------------
    def spawn_group(self, label: str, folder: str) -> str:
        a = self.mcp.call("editor_toolset.toolsets.scene.SceneTools", "add_to_scene_from_class",
                           {"actor_type": ref("/Script/Engine.Actor"), "name": label,
                            "xform": xform((0, 0, 0))})["refPath"]
        self.mcp.call("editor_toolset.toolsets.actor.ActorTools", "set_label",
                       {"actor": ref(a), "label": label})
        self.mcp.call("editor_toolset.toolsets.scene.SceneTools", "set_actor_folder",
                       {"actor": ref(a), "folder_path": folder})
        self._comp_counts[a] = 0
        return a

    def _name(self, actor: str, prefix: str) -> str:
        self._comp_counts[actor] = self._comp_counts.get(actor, 0) + 1
        return f"{prefix}_{self._comp_counts[actor]}"

    def add_box(self, actor: str, size_m: tuple[float, float, float],
                loc_cm: tuple[float, float, float], yaw_deg: float, material: str,
                prefix: str = "Box") -> str:
        comp = self.mcp.call("editor_toolset.toolsets.primitive.PrimitiveTools", "add_cube", {
            "actor": ref(actor), "name": self._name(actor, prefix),
            "dimensions": vec(size_m[0] * M_TO_UU, size_m[1] * M_TO_UU, size_m[2] * M_TO_UU),
            "local_transform": xform(loc_cm, yaw_deg),
        })["refPath"]
        self._set_material(comp, material)
        return comp

    def add_cyl(self, actor: str, radius_m: float, height_m: float,
                loc_cm: tuple[float, float, float], material: str, prefix: str = "Cyl") -> str:
        comp = self.mcp.call("editor_toolset.toolsets.primitive.PrimitiveTools", "add_cylinder", {
            "actor": ref(actor), "name": self._name(actor, prefix),
            "radius": radius_m * M_TO_UU, "height": height_m * M_TO_UU,
            "local_transform": xform(loc_cm),
        })["refPath"]
        self._set_material(comp, material)
        return comp

    def add_cone(self, actor: str, radius_m: float, height_m: float,
                 loc_cm: tuple[float, float, float], material: str, prefix: str = "Cone") -> str:
        comp = self.mcp.call("editor_toolset.toolsets.primitive.PrimitiveTools", "add_cone", {
            "actor": ref(actor), "name": self._name(actor, prefix),
            "radius": radius_m * M_TO_UU, "height": height_m * M_TO_UU,
            "local_transform": xform(loc_cm),
        })["refPath"]
        self._set_material(comp, material)
        return comp

    def _set_material(self, component: str, material_key: str) -> None:
        self.mcp.call("editor_toolset.toolsets.object.ObjectTools", "set_properties", {
            "instance": ref(component),
            "values": json.dumps({"overrideMaterials": [ref(self.mat[material_key])]}),
        })


def fairway_width(t: float) -> float:
    """Corridor FULL width in metres: narrow at tee, wide mid-hole, narrow at
    green. ``t`` is fraction of route length travelled, 0..1. Schematic, not
    surveyed -- real Augusta fairways run roughly this range."""
    if t < 0.08:
        return 16 + (32 - 16) * (t / 0.08)
    if t > 0.85:
        return 32 - (32 - 20) * ((t - 0.85) / 0.15)
    return 32


def clean_folder(mcp: UnrealMCP, folder_path: str) -> None:
    try:
        actors = mcp.call("editor_toolset.toolsets.scene.SceneTools", "get_actors_in_folder",
                           {"folder_path": folder_path})
    except MCPError:
        return
    for a in actors:
        try:
            mcp.call("editor_toolset.toolsets.scene.SceneTools", "remove_from_scene",
                      {"actor": a})
        except MCPError as e:
            print(f"  (couldn't remove {a.get('refPath')}: {e})")


def build_hole(mcp: UnrealMCP, mats: dict[str, str], course, hole_number: int,
               builder: HoleBuilder, clean: bool = True) -> None:
    hole = course.hole(hole_number)
    hm_table = load_hole_maps()
    hole_map = hm_table.get(hole_number)
    folder_root = f"Golf/Hole {hole_number:02d} - {hole.name}"

    if clean:
        clean_folder(mcp, folder_root)

    print(f"[hole {hole_number}] {hole.name}, par {hole.par}, {hole.card_yardage} yd")

    route = hole.route_points
    world_pts = [builder.world(p) for p in route]
    seg_lengths = [
        math.dist(world_pts[i][:2], world_pts[i + 1][:2]) / M_TO_UU
        for i in range(len(world_pts) - 1)
    ]
    total_len = sum(seg_lengths) or 1.0

    # --- turf: rough corridor (wide, dark) then fairway corridor (narrower, on top)
    turf_actor = builder.spawn_group(f"Turf", f"{folder_root}")
    travelled = 0.0
    for i in range(len(world_pts) - 1):
        x0, y0, z0 = world_pts[i]
        x1, y1, z1 = world_pts[i + 1]
        seg_len = seg_lengths[i]
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
        mid = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        t_mid = (travelled + seg_len / 2) / total_len
        width_m = fairway_width(t_mid)
        # Rough: a generous fixed-width band under the fairway, same segmentation.
        # Both sit a bit proud of the pre-existing ground plane so their edges
        # actually read as a mown line instead of blending into it.
        builder.add_box(turf_actor, (seg_len + 4.0, width_m + 24.0, 0.30),
                         (mid[0], mid[1], mid[2] + 15), yaw, "Rough", prefix="Rough")
        builder.add_box(turf_actor, (seg_len + 4.0, width_m, 0.30),
                         (mid[0], mid[1], mid[2] + 30), yaw, "Fairway", prefix="Fairway")
        travelled += seg_len

    # --- tee box
    tee_yaw = math.degrees(math.atan2(world_pts[1][1] - world_pts[0][1],
                                       world_pts[1][0] - world_pts[0][0]))
    builder.add_box(turf_actor, (6.0, 10.0, 0.32), world_pts[0], tee_yaw, "Tee", prefix="TeeBox")

    # --- green + flagstick + cup
    green_actor = builder.spawn_group("Green", folder_root)
    px, py, pz = world_pts[-1]
    GREEN_RADIUS_M = 11.5
    builder.add_cyl(green_actor, GREEN_RADIUS_M, 0.40, (px, py, pz + 4), "Green", prefix="Green")
    builder.add_cyl(green_actor, 0.054, 0.03, (px, py, pz + 24), "Cup", prefix="Cup")
    builder.add_cyl(green_actor, 0.025, 1.9, (px, py, pz + 24 + 95), "FlagPole", prefix="Pole")
    builder.add_box(green_actor, (0.30, 0.01, 0.20), (px + 15, py, pz + 24 + 175),
                     0.0, "Flag", prefix="Flag")

    # --- artwork-derived features
    mf = None
    feats = None
    if hole_map is not None and hole_map.exists():
        feats = extract_hole_features(hole, hole_map, downscale=4)
        mf = feats.map
        if mf is not None:
            # Belt-and-braces on top of the centroid fix and features.py's own
            # area caps: no Augusta hole is within a mile of 700m, so a
            # feature whose centroid lands further than that from the tee is
            # a misdetection, not a distant bunker.
            def plausible(cx: float, cy: float) -> bool:
                return math.hypot(cx, cy) < 700.0

            if mf.bunkers:
                bunker_actor = builder.spawn_group("Bunkers", folder_root)
                for poly in mf.bunkers:
                    cx, cy, area = poly_area_centroid(poly)
                    if not plausible(cx, cy):
                        continue
                    radius_m = min(20.0, max(1.0, math.sqrt(max(area, 1.0) / math.pi)))
                    wx, wy, wz = builder.world_en(cx, cy, hole.tee, dz_m=-0.10)
                    builder.add_cyl(bunker_actor, radius_m, 0.20, (wx, wy, wz), "Bunker",
                                     prefix="Bunker")
            if mf.water:
                water_actor = builder.spawn_group("Water", folder_root)
                for poly in mf.water:
                    cx, cy, area = poly_area_centroid(poly)
                    if not plausible(cx, cy):
                        continue
                    radius_m = min(45.0, max(1.5, math.sqrt(max(area, 1.0) / math.pi)))
                    wx, wy, wz = builder.world_en(cx, cy, hole.tee, dz_m=-0.25)
                    builder.add_cyl(water_actor, radius_m, 0.15, (wx, wy, wz), "Water",
                                     prefix="Water")
            if mf.trees:
                tree_actor = builder.spawn_group("Trees", folder_root)
                for poly in mf.trees[:25]:
                    cx, cy, area = poly_area_centroid(poly)
                    if area < 4.0 or not plausible(cx, cy):
                        continue
                    wx, wy, wz = builder.world_en(cx, cy, hole.tee)
                    canopy_r = min(4.0, max(1.6, math.sqrt(area / math.pi) * 0.7))
                    builder.add_cyl(tree_actor, 0.28, 2.2, (wx, wy, wz), "TreeTrunk",
                                     prefix="Trunk")
                    builder.add_cone(tree_actor, canopy_r, 4.2, (wx, wy, wz + 200), "TreeCanopy",
                                      prefix="Canopy")

        # --- slope arrows on the green (direction real, position schematic)
        if feats.slope_arrows:
            arrow_actor = builder.spawn_group("GreenSlope", folder_root)
            overall_yaw = math.degrees(math.atan2(py - world_pts[-2][1], px - world_pts[-2][0])) \
                if len(world_pts) > 1 else 0.0
            us = [a.u for a in feats.slope_arrows]
            vs = [a.v for a in feats.slope_arrows]
            u0, u1 = min(us), max(us) or 1.0
            v0, v1 = min(vs), max(vs) or 1.0
            for a in feats.slope_arrows:
                fu = 0.5 if u1 == u0 else (a.u - u0) / (u1 - u0) - 0.5
                fv = 0.5 if v1 == v0 else (a.v - v0) / (v1 - v0) - 0.5
                ox = fu * GREEN_RADIUS_M * 1.3
                oy = -fv * GREEN_RADIUS_M * 1.3
                local_yaw = a.bearing_deg  # absolute compass; fine as a visual cue
                builder.add_cone(arrow_actor, 0.10 + 0.10 * a.relative_length, 0.5,
                                  (px + ox * M_TO_UU, py + oy * M_TO_UU, pz + 4 + 40),
                                  "SlopeArrow", prefix="Slope")

    print("  built: turf, tee, green" +
          (f", {len(mf.bunkers)} bunkers" if mf and mf.bunkers else "") +
          (f", {len(mf.water)} water" if mf and mf.water else "") +
          (f", {len(mf.trees[:25])} trees" if mf and mf.trees else "") +
          (f", {len(feats.slope_arrows)} slope arrows" if feats and feats.slope_arrows else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hole", type=int, nargs="+", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    ap.add_argument("--no-clean", action="store_true",
                     help="Don't delete the hole's existing actors before rebuilding.")
    args = ap.parse_args()

    course = load_augusta()
    mcp = UnrealMCP(args.url)
    mats = ensure_materials(mcp)

    origin = course.hole(1).tee
    # Z=0 is the pre-existing flat "Floor" actor's surface. origin_elev is
    # the course's real lowest point (just under it, for a small clearance
    # margin), so every hole's compressed elevation (see
    # HoleBuilder.terrain_z_m) comes out >= 0 -- nothing sinks below the
    # floor, and nothing floats far above it either now that the relief
    # itself is compressed rather than shown at true scale.
    course_lo_elev, _course_hi_elev = course.elevation_range_m()
    origin_elev = course_lo_elev - 1.0
    builder = HoleBuilder(mcp, mats, origin, origin_elev)

    for hn in args.hole:
        build_hole(mcp, mats, course, hn, builder, clean=not args.no_clean)

    print("done.")


if __name__ == "__main__":
    main()
