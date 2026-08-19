"""Interactive trajectory explorer.

Run:  streamlit run scripts/explorer.py

Exposes what the physics and course layers already do -- pick a hole, a shot
origin, launch conditions, wind and air, and see the integrated flight against
real Augusta terrain. It deliberately does NOT recommend a club: that needs the
dispersion model and expected-strokes surface which do not exist yet. See the
"What this does not know" panel, which is part of the UI on purpose.

The UI lives in main() under a __main__ guard -- Streamlit runs a script with
__name__ == "__main__", so this still works as a Streamlit app while leaving the
chart builders importable for tests and for rendering figures to PNG.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.path import Path as MplPath
from scipy.optimize import brentq

from caddie.course import GeoPoint, load_augusta, terrain_from_hole
from caddie.course.features import extract_map_features
from caddie.course.maps import (
    MapProjection,
    enu_offset,
    load_hole_info,
    load_hole_maps,
)
from caddie.physics import (
    BAG_LIMIT,
    EXTRA_VARIANTS,
    Atmosphere,
    FlightIntegrator,
    LaunchConditions,
    PGA_TOUR_AVERAGES,
    WindField,
)
from caddie.physics.constants import MPH_MS
from caddie.physics.roll import estimate_roll_yards, wind_component_along_shot
from caddie.physics.trajectory import flat_terrain
from caddie.physics.wind import AUGUSTA_OPEN, AUGUSTA_SHELTERED_CORRIDOR
from caddie.shot import (
    FLIGHT_HEIGHTS,
    DPlaneModel,
    ShotShape,
    plan_shot,
)

YARD_M = 0.9144
FT_M = 0.3048

# No green polygon exists (see README "Known gaps") -- this is a generous
# circular proxy centred on the pin, not the actual green outline, used only
# to decide when the main view switches to a zoomed-in read of the finish.
GREEN_RADIUS_YD = 20.0
# The zoom crop is wider than the trigger radius so a ball that just crossed
# the threshold still has margin around it instead of sitting on the edge.
GREEN_ZOOM_RADIUS_YD = 32.0
# A shot landing this close to the pin counts as holed outright -- no putting
# step needed. Generous on purpose: terrain is interpolated between a handful
# of surveyed points (see caveats), nowhere near precise enough for a real
# 4.25" cup, so this is "close enough that arguing the last foot is pointless."
HOLE_OUT_RADIUS_YD = 1.5

CUSTOM_AIM = "Custom aim point"

# Every club the explorer knows how to hit. The tour table stops at PW --
# EXTRA_VARIANTS extends it with 7-wood/2-iron and loft-numbered wedges for
# bag-building variety (see caddie.physics.ball for why those are
# extrapolated rather than sourced the same way as PGA_TOUR_AVERAGES). This
# is the full pool a player picks their actual bag FROM, not the bag itself.
# Sorted into realistic bag order (long to short) rather than tour-table-
# then-extras -- naive concatenation would bury 7-wood/2-iron at the end,
# nowhere near the 5-wood/3-iron they actually sit between.
_CLUB_ORDER = (
    "Driver", "3-wood", "5-wood", "7-wood", "Hybrid",
    "2-iron", "3-iron", "4-iron", "5-iron", "6-iron", "7-iron", "8-iron", "9-iron",
    "PW", "50°", "52°", "54°", "56°", "58°", "60°",
)
FULL_CLUB_POOL: tuple = tuple(
    sorted(PGA_TOUR_AVERAGES + EXTRA_VARIANTS, key=lambda c: _CLUB_ORDER.index(c.name))
)

DEFAULT_BAG_NAMES: tuple[str, ...] = (
    "Driver", "3-wood", "Hybrid",
    "4-iron", "5-iron", "6-iron", "7-iron", "8-iron", "9-iron",
    "PW", "52°", "56°", "60°",
)  # BAG_LIMIT (13) clubs -- a sensible bag out of the box, no setup required.

# ---------------------------------------------------------------------------
# Palette. Two categorical slots (shot / calm reference), each mode stepped for
# its own surface rather than flipped. Both modes pass the six palette checks:
# lightness band, chroma floor, CVD separation (worst adjacent dE 24.7 light /
# 26.8 dark), normal-vision floor (33.6 / 31.8) and 3:1 contrast vs surface.
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
        "terrain": "#e1e0d9",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "terrain": "#2c2c2a",
    },
}

SHELTER_PRESETS = {
    "Sheltered corridor (pines both sides)": AUGUSTA_SHELTERED_CORRIDOR,
    "Open / exposed": AUGUSTA_OPEN,
    "Free stream (no shelter)": dict(
        roughness_length=0.03, shelter_factor=1.0, treeline_height=1.0
    ),
}

WIND_CARDINALS = {
    "Into the face": 0,
    "From the right": 90,
    "Downwind": 180,
    "From the left": 270,
}


# ---------------------------------------------------------------------------
# Model plumbing. Everything cached below takes primitives only, so Streamlit
# can hash it -- the terrain callback and the integrator are rebuilt inside.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_course():
    return load_augusta()


@st.cache_resource
def get_hole_maps():
    return load_hole_maps()


@st.cache_resource
def get_hole_info():
    return load_hole_info()


@st.cache_resource
def get_map_image(path_str: str):
    """Load a hole illustration once. Returns (array, width, height)."""
    from PIL import Image

    im = Image.open(path_str).convert("RGB")
    return np.asarray(im), im.size[0], im.size[1]


@st.cache_resource
def get_hole_water(hole_number: int) -> list:
    """Water-hazard polygons for one hole, colour-segmented from the map
    illustration (see caddie.course.features.water_mask) and given back as
    real ENU metres. Cached per hole -- the contour trace is the slow part
    of extract_map_features, and water doesn't change between reruns."""
    hole = get_course().hole(hole_number)
    hole_map = get_hole_maps().get(hole_number)
    if hole_map is None or not hole_map.exists():
        return []
    features = extract_map_features(hole, hole_map)
    return features.water if features is not None else []


@st.cache_data(show_spinner=False)
def solve_speed_for_carry(
    launch_angle_deg: float, backspin_rpm: float, target_carry_yd: float,
) -> float:
    """Ball speed (mph) that gives this launch angle/spin a carry of
    ``target_carry_yd`` in a neutral, no-wind reference atmosphere.

    Turns "I carry my 7-iron 155 yd" -- the number a player actually knows --
    into the launch parameter the physics engine needs, holding angle and
    spin at the club's tour-average shape (only ball speed scales with the
    player). It's a real solve, not a lookup: same coarse-solve pattern as
    ``shot.plan_shot``, just a 1-D root find on carry instead of curve/aim.
    """
    integ = FlightIntegrator(atmosphere=Atmosphere(), dt=0.004)

    def carry_error(speed_mph: float) -> float:
        launch = LaunchConditions(
            ball_speed_mph=speed_mph, launch_angle_deg=launch_angle_deg,
            backspin_rpm=backspin_rpm,
        )
        return integ.integrate(launch, terrain=flat_terrain).carry_yards - target_carry_yd

    lo, hi = 40.0, 200.0
    e_lo, e_hi = carry_error(lo), carry_error(hi)
    if e_lo > 0:
        return lo  # even the slowest speed on the slider overshoots this carry
    if e_hi < 0:
        return hi  # even the fastest speed on the slider falls short
    return brentq(carry_error, lo, hi, xtol=0.1)


def waypoints(hole):
    """Named shot origins / targets for a hole, in route order."""
    pts = {"Tee": hole.tee}
    for i, p in enumerate(hole.aim_points, start=1):
        pts[f"Aim point {i}"] = p
    pts["Pin"] = hole.pin
    return pts


@st.cache_data(show_spinner=False)
def fly(
    hole_number: int,
    origin_point: tuple[float, float, float | None],
    target_point: tuple[float, float, float | None],
    launch_params: tuple,
    wind_params: tuple | None,
    air_params: tuple,
    shape_params: tuple,
    loft_deg: float = 30.0,
    firmness: float = 1.0,
):
    """Integrate one shot. Returns plain arrays + scalars so it caches cleanly.

    ``origin_point`` / ``target_point`` are (lat, lon, elevation_m) rather than
    named waypoints, so an origin can be an arbitrary saved landing spot from
    an earlier shot in the same session, not just the tee/aim points/pin.
    """
    course = get_course()
    hole = course.hole(hole_number)
    origin = GeoPoint(*origin_point)
    target = GeoPoint(*target_point)

    frame = hole.frame_from(origin, target)
    terrain = terrain_from_hole(hole, frame)

    temp_c, rh, altimeter_hpa = air_params
    # Air state is evaluated once, at the origin's elevation, and held constant
    # for the flight. Over the worst case on the property (the 101 ft drop on
    # 10) density varies ~0.4%, well inside the model's other uncertainties.
    altitude_m = origin.elevation_m if origin.elevation_m is not None else 45.0
    air = Atmosphere.from_conditions(
        temp_c=temp_c,
        altitude_m=altitude_m,
        relative_humidity=rh,
        altimeter_hpa=altimeter_hpa,
    )

    speed, angle, spin, azimuth, sidespin = launch_params
    stock = LaunchConditions(
        ball_speed_mph=speed,
        launch_angle_deg=angle,
        backspin_rpm=spin,
        azimuth_deg=azimuth,
        sidespin_rpm=sidespin,
    )

    wind = None
    if wind_params is not None:
        speed_mph, dir_deg, z0, shelter, treeline, veer = wind_params
        if speed_mph > 0:
            wind = WindField.from_mph(
                speed_mph,
                dir_deg,
                roughness_length=z0,
                shelter_factor=shelter,
                treeline_height=treeline,
                veer_deg_per_100m=veer,
            )

    integ = FlightIntegrator(atmosphere=air, dt=0.002)

    # --- Resolve the shot the three shaping modes describe ------------------
    mode, curve_yards, height, hand, face_deg, path_deg, face_w, axis_per_ftp = (
        shape_params
    )
    shaping: dict = {"mode": mode, "error": None}

    if mode == "shape":
        shape = ShotShape(curve_yards=curve_yards, height=height, hand=hand)
        if curve_yards == 0.0:
            # No curve dialled in -- no auto-aim/curve compensation either.
            # Plain stock swing at this flighting, straight aim, and wind is
            # free to push it wherever it pushes it: nothing here is solving
            # against it. Auto-aim only kicks in once you actually ask for a
            # curve (below), which is the one case that genuinely needs a
            # spin-axis + aim solve to know what shape you get.
            launch = FLIGHT_HEIGHTS[height].apply(stock)
            shaping.update(label=shape.describe())
        else:
            try:
                plan = plan_shot(integ, stock, shape, wind=wind, terrain=terrain)
            except ValueError as exc:  # curve not reachable with this club
                shaping["error"] = str(exc)
                launch = FLIGHT_HEIGHTS[height].apply(stock)
            else:
                launch = plan.launch
                shaping.update(
                    solved_aim_deg=plan.aim_deg,
                    spin_axis_deg=plan.spin_axis_deg,
                    curve_achieved=plan.curve_achieved_yards,
                    shaping_cost=plan.shaping_cost_yards,
                    converged=plan.converged,
                    label=shape.describe(),
                )
    elif mode == "swing":
        dplane = DPlaneModel(face_weight=face_w, axis_per_face_to_path=axis_per_ftp)
        launch = dplane.launch_from_swing(
            FLIGHT_HEIGHTS[height].apply(stock), face_deg, path_deg
        )
        shaping.update(
            solved_aim_deg=launch.azimuth_deg,
            spin_axis_deg=launch.spin_axis_deg,
            face_to_path=face_deg - path_deg,
            label=f"face {face_deg:+.1f}deg / path {path_deg:+.1f}deg",
        )
    else:  # "manual" -- the raw launch-monitor sliders
        launch = FLIGHT_HEIGHTS[height].apply(stock)
        shaping.update(label="launch conditions as set")

    # "Default Trajectory": the stock ball speed/launch angle/backspin, no
    # curve/flighting/face-path/manual sidespin, straight aim, and NO
    # compensation for wind -- flown through the same wind as "My Shot" so
    # it shows exactly how far wind alone pushes a plain, unshaped shot off
    # the target line.
    default_launch = LaunchConditions(
        ball_speed_mph=speed,
        launch_angle_deg=angle,
        backspin_rpm=spin,
        azimuth_deg=0.0,
        sidespin_rpm=0.0,
    )

    shot = integ.integrate(launch, wind=wind, terrain=terrain)
    calm = integ.integrate(default_launch, wind=wind, terrain=terrain)

    tx, ty, tz = frame.to_local(target)

    # Wind along the shot's own line -- W in caddie.physics.roll's convention
    # (tailwind positive), which is the opposite sign to WindField's
    # meteorological direction_from_deg (0 = headwind). 0 mph if calm.
    wind_component_mph = (
        wind_component_along_shot(wind_params[0], wind_params[1])
        if wind is not None else 0.0
    )

    def roll_offset(t):
        # Roll: an empirical estimate layered on top of the validated carry
        # integration (see caddie.physics.roll) -- the ONE deliberately
        # approximate step in an otherwise physics-derived flight, since
        # this engine doesn't simulate bounce/turf friction at all. Extended
        # along the ground track's direction at impact, not straight down
        # the target line, so a shot still curving sideways at landing rolls
        # on curving, not snapping back onto line.
        elevation_ft = float(t.positions[-1, 2]) / FT_M
        roll_yd = estimate_roll_yards(
            carry_yd=t.carry_yards, ball_speed_mph=speed, loft_deg=loft_deg,
            wind_component_mph=wind_component_mph, elevation_ft=elevation_ft,
            firmness=firmness,
        )
        vx, vy, _ = t.velocities[-1]
        horiz = float(np.hypot(vx, vy))
        landing_z = float(t.positions[-1, 2])
        if horiz > 1e-6 and roll_yd > 0:
            dx, dy = vx / horiz * roll_yd * YARD_M, vy / horiz * roll_yd * YARD_M
        else:
            dx, dy = 0.0, 0.0
        # Snap the endpoint to the REAL terrain height there, not a flat
        # continuation of the landing height -- on sloped ground a flat
        # guess can end up below the actual surface at the rolled-out spot
        # (uphill) or floating above it (downhill), and the below-terrain
        # guard in trajectory.integrate() rightly rejects the former if it
        # becomes the next shot's origin.
        end_z = float(terrain(t.positions[-1, 0] + dx, t.positions[-1, 1] + dy))
        return dx, dy, roll_yd, landing_z, end_z

    # Where the ball actually comes to REST, back in lat/lon -- this is what
    # a played shot hands to the next one as its origin, and what counts for
    # distance-to-pin. landing_geo (carry only, no roll) is kept alongside
    # for the "This shot leaves" preview line, which is about where the ball
    # first touches down, not its final position.
    shot_roll_dx, shot_roll_dy, shot_roll_yd, _, shot_roll_end_z = roll_offset(shot)
    lx, ly, lz = shot.positions[-1]
    landing_geo = frame.to_geo(float(lx), float(ly), float(lz))
    final_geo = frame.to_geo(
        float(lx + shot_roll_dx), float(ly + shot_roll_dy), shot_roll_end_z,
    )
    distance_to_pin_yd = landing_geo.distance_yards_to(hole.pin)
    total_distance_to_pin_yd = final_geo.distance_yards_to(hole.pin)
    # Where you're standing right NOW, independent of the club/target you
    # happen to have dialled in for the shot you're previewing -- see the
    # "Now" vs "This shot leaves" split below the fly/save buttons.
    origin_distance_to_pin_yd = origin.distance_yards_to(hole.pin)

    # Ground profile down the target line, out past whichever finishes longer
    # (including roll, so the ground panel doesn't clip the rolled-out ball).
    x_max = max(
        shot.positions[-1, 0] + shot_roll_dx, calm.positions[-1, 0], tx,
    ) * 1.06
    ground_x = np.linspace(0.0, x_max, 240)
    ground_z = np.array([terrain(float(x), 0.0) for x in ground_x])

    ROLL_ANIM_STEPS = 10  # extra samples appended so the roll animates too,
    # through the exact same until_idx slicing every chart already does for
    # flight -- no separate "rolling" code path needed anywhere downstream.

    def pack(t):
        roll_dx, roll_dy, roll_yd, landing_z, roll_end_z = roll_offset(t)
        x, y, z = t.positions[:, 0], t.positions[:, 1], t.positions[:, 2]
        speed = np.linalg.norm(t.velocities, axis=1)
        spin = t.spins_rpm
        times = t.times
        n_roll = ROLL_ANIM_STEPS if roll_yd > 0.5 else 0
        if n_roll:
            frac = np.linspace(1.0 / n_roll, 1.0, n_roll)  # excludes 0 -- landing point already the last flight sample
            x = np.concatenate([x, x[-1] + roll_dx * frac])
            y = np.concatenate([y, y[-1] + roll_dy * frac])
            # Interpolated to the real terrain height at the rollout point,
            # not held flat at the landing height -- see roll_offset.
            z = np.concatenate([z, landing_z + (roll_end_z - landing_z) * frac])
            # Decelerating roll, not a real speed -- just enough for the
            # animation and table to show something monotonically slowing
            # rather than a jump-cut. Roll takes ~1.5s, a rough ballpark.
            speed = np.concatenate([speed, speed[-1] * (1.0 - frac) * 0.5])
            spin = np.concatenate([spin, np.full(n_roll, spin[-1])])
            times = np.concatenate([times, times[-1] + frac * 1.5])
        return {
            "times": times,
            "x": x,
            "y": y,
            "z": z,
            "speed": speed,
            "spin": spin,
            "carry": t.carry_yards,
            "offline": t.offline_yards,
            "apex": t.apex_yards,
            "descent": t.descent_angle_deg,
            "flight_time": t.flight_time,
            "impact_speed": t.impact_speed,
            "impact_spin": t.impact_spin_rpm,
            "landed": t.landed,
            "roll_yd": roll_yd,
            "total_yd": t.carry_yards + roll_yd,
            "n_flight_samples": len(t.positions),
        }

    # Ground tracks in ENU metres relative to the HOLE'S TEE -- the frame the
    # hole-map georeference is defined in, so the overlay works from any
    # origin. Takes the already roll-extended x/y from pack() so the map
    # track includes the rolled-out segment too, animated the same way.
    def enu_track(packed):
        return np.array([
            enu_offset(hole.tee, frame.to_geo(float(x), float(y)))
            for x, y in zip(packed["x"], packed["y"])
        ])

    shot_packed = pack(shot)
    calm_packed = pack(calm)

    return {
        "shot": shot_packed,
        "calm": calm_packed,
        "shaping": shaping,
        "launch_describe": launch.describe(),
        "shot_enu": enu_track(shot_packed),
        "calm_enu": enu_track(calm_packed),
        "origin_enu": enu_offset(hole.tee, origin),
        "target_enu": enu_offset(hole.tee, target),
        "route_enu": [enu_offset(hole.tee, p) for p in hole.route_points],
        "target_local": (tx, ty, tz),
        "ground_x": ground_x,
        "ground_z": ground_z,
        "air_describe": air.describe(),
        "density_ratio": air.density_ratio,
        "wind_describe": wind.describe() if wind else "calm",
        "origin_elev_m": origin.elevation_m,
        "target_elev_m": target.elevation_m,
        "straight_yards": origin.distance_yards_to(target),
        "plays_like": origin.plays_like_yards_to(target),
        "landing_geo": landing_geo,
        "distance_to_pin_yd": distance_to_pin_yd,
        "origin_distance_to_pin_yd": origin_distance_to_pin_yd,
        "final_geo": final_geo,
        "total_distance_to_pin_yd": total_distance_to_pin_yd,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _style_axes(ax, p, xlabel, ylabel):
    ax.set_facecolor(p["surface"])
    ax.grid(True, which="major", color=p["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(p["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=p["muted"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=p["ink_2"], fontsize=10)
    ax.set_ylabel(ylabel, color=p["ink_2"], fontsize=10)


def _distance_label(packed, word: str = "carry") -> str:
    """'172 yd carry' when roll is negligible, '201 yd (172 carry + 29 roll)'
    when it isn't -- the label now describes where the ball actually ends
    up, not just where it first touches down."""
    if packed["roll_yd"] > 0.5:
        return f"{packed['total_yd']:.0f} yd ({packed['carry']:.0f} {word} + {packed['roll_yd']:.0f} roll)"
    return f"{packed['carry']:.0f} yd {word}"


def _endpoint_label(ax, x, y, text, color, x_span):
    """Label one series at its landing point, kept inside the axes.

    Past ~80% of the x-range the label would run off the right edge, so it
    flips to sit left of the point instead of being clipped.
    """
    flip = x > 0.8 * x_span
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(-8, 8) if flip else (8, 8),
        textcoords="offset points",
        ha="right" if flip else "left",
        color=color,
        fontsize=9,
        fontweight="bold",
        zorder=7,
    )


def _legend(ax, p):
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(p["ink_2"])


def elevation_chart(data, p, show_calm: bool = True, until_idx: int | None = None):
    """Side elevation: height against downrange, over the real ground profile.

    ``until_idx`` draws the shot trajectory only up to that sample, with a
    ball marker at the tip -- the mid-flight frame used by the animation.
    ``None`` (the default) draws the full, completed flight.
    """
    shot, calm = data["shot"], data["calm"]
    tx, ty, tz = data["target_local"]
    animating = until_idx is not None
    i = (until_idx + 1) if animating else len(shot["x"])

    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=p["surface"])
    _style_axes(ax, p, "downrange (yd)", "height above origin (ft)")

    gx = data["ground_x"] / YARD_M
    gz = data["ground_z"] / FT_M
    top = max(shot["z"].max(), calm["z"].max()) / FT_M
    floor = min(gz.min(), 0.0) - 0.08 * (top - min(gz.min(), 0.0))

    ax.fill_between(gx, floor, gz, color=p["terrain"], linewidth=0, zorder=1)
    ax.plot(gx, gz, color=p["axis"], linewidth=1.0, zorder=2)

    if show_calm and not animating:
        ax.plot(
            calm["x"] / YARD_M, calm["z"] / FT_M,
            color=p["series_2"], linewidth=2.0, zorder=3,
            label="Default Trajectory",
        )
    ax.plot(
        shot["x"][:i] / YARD_M, shot["z"][:i] / FT_M,
        color=p["series_1"], linewidth=2.0, zorder=4, label="My Shot",
    )
    if animating:
        ax.plot(
            [shot["x"][i - 1] / YARD_M], [shot["z"][i - 1] / FT_M],
            marker="o", markersize=7, color=p["series_1"],
            markeredgecolor=p["surface"], markeredgewidth=1.5, zorder=5,
        )

    # Target: distinct shape + ink colour, never a series hue.
    ax.plot(
        [tx / YARD_M], [tz / FT_M],
        marker="*", markersize=11, color=p["ink"],
        markeredgecolor=p["surface"], markeredgewidth=2,
        linestyle="none", zorder=6, label="Target",
    )

    def total_and_carry(packed) -> str:
        # Total up front, carry alongside it -- roll itself is left as the
        # difference between the two rather than spelled out as its own
        # "+ N roll" term.
        if packed["roll_yd"] > 0.5:
            return f"{packed['total_yd']:.0f} yd ({packed['carry']:.0f} yd carry)"
        return f"{packed['carry']:.0f} yd carry"

    x_span = gx.max()
    if not animating:
        _endpoint_label(
            ax, shot["x"][-1] / YARD_M, shot["z"][-1] / FT_M,
            total_and_carry(shot), p["series_1"], x_span,
        )
        # Label the reference only when it is far enough away to be legible on
        # its own -- in a pure crosswind both series land within a yard of
        # each other and two labels would collide into mush. The legend and
        # table still carry it. This is the "direct-label selectively" rule
        # doing real work.
        if show_calm and abs(calm["carry"] - shot["carry"]) >= 5.0:
            _endpoint_label(
                ax, calm["x"][-1] / YARD_M, calm["z"][-1] / FT_M,
                total_and_carry(calm), p["series_2"], x_span,
            )

    ax.set_ylim(floor, top * 1.22)
    ax.set_xlim(0.0, x_span)
    if not animating:  # legend text layout is real per-frame cost; skip while playing
        _legend(ax, p)
    fig.tight_layout()
    return fig


def plan_chart(data, p, show_calm: bool = True):
    """Plan view: lateral deviation against downrange."""
    shot, calm = data["shot"], data["calm"]
    tx, ty, tz = data["target_local"]

    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=p["surface"])
    _style_axes(ax, p, "downrange (yd)", "offline (yd) — right positive")

    ax.axhline(0.0, color=p["axis"], linewidth=0.8, zorder=2)

    if show_calm:
        ax.plot(
            calm["x"] / YARD_M, calm["y"] / YARD_M,
            color=p["series_2"], linewidth=2.0, zorder=3,
            label="Default Trajectory",
        )
    ax.plot(
        shot["x"] / YARD_M, shot["y"] / YARD_M,
        color=p["series_1"], linewidth=2.0, zorder=4, label="My Shot",
    )
    ax.plot(
        [tx / YARD_M], [ty / YARD_M],
        marker="*", markersize=11, color=p["ink"],
        markeredgecolor=p["surface"], markeredgewidth=2,
        linestyle="none", zorder=6, label="Target",
    )

    _endpoint_label(
        ax, shot["x"][-1] / YARD_M, shot["y"][-1] / YARD_M,
        f"{shot['offline']:+.1f} yd", p["series_1"],
        data["ground_x"].max() / YARD_M,
    )

    span = max(
        float(np.abs(shot["y"]).max() / YARD_M),
        float(np.abs(calm["y"]).max() / YARD_M),
        abs(ty / YARD_M),
        4.0,
    )
    # Inverted on purpose: a plain y-up axis puts "right" at the top of the
    # screen, which is backwards for a bird's-eye view with the tee on the
    # left and the green on the right -- facing the green, the golfer's
    # right hand points toward the BOTTOM of the screen (south, on a
    # north-up map with downrange running east). map_chart already gets this
    # right because MapProjection's transform includes the same reflection
    # (see its docstring); this view was the one still plotting it mirrored.
    ax.set_ylim(span * 1.45, -span * 1.45)
    ax.set_xlim(0.0, data["ground_x"].max() / YARD_M)
    _legend(ax, p)
    fig.tight_layout()
    return fig


# A flag-on-a-pole glyph, as a marker path -- pole from bottom to top, then a
# closed pennant triangle back down to the pole. Local marker coordinates
# span roughly [-1, 1]; matplotlib scales/positions the whole thing as one
# point-sized unit, same as the built-in "*"/"s" markers below.
_PIN_FLAG_MARKER = MplPath(
    vertices=[
        (0.0, -1.0), (0.0, 1.0),
        (0.0, 1.0), (0.85, 0.5), (0.0, 0.05), (0.0, 1.0),
    ],
    codes=[
        MplPath.MOVETO, MplPath.LINETO,
        MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY,
    ],
)


def _draw_pin_flag(ax, proj, hole, p):
    """The actual hole, marked with a flag -- always, regardless of what's
    being aimed at. "Target" (the star) is whatever waypoint or custom point
    a shot is aimed toward, which is frequently NOT the pin: a tee shot aims
    at a dogleg's aim point, a custom aim can go anywhere. Without a mark for
    the pin itself, there'd be no way to tell "where I'm aiming" from "where
    the hole actually is" when those two differ.

    Fixed on-screen size, like the target/origin markers, not drawn to real
    flagstick scale -- a to-scale flag is exactly the thing that vanishes at
    whole-hole zoom when the aim point is far from the pin, which is the one
    case this exists to make visible.
    """
    px, py = proj.geo_to_pixels(hole.pin)
    ax.plot(
        [px], [py], marker=_PIN_FLAG_MARKER, ms=15,
        markerfacecolor="#d94f3d", markeredgecolor=p["ink"], markeredgewidth=1.3,
        ls="none", zorder=8, label="Pin",
    )


def map_chart(
    data, p, hole, hole_map, show_calm: bool = True,
    until_idx: int | None = None, zoom_radius_yd: float | None = None,
    water: list | None = None, water_alpha: float = 0.2,
):
    """The shot drawn on the illustrated hole map.

    The map is the figure -- no axes, no grid, no invented chrome on top of
    someone's artwork. Only the marks that carry information go on: the flight
    path, where it finishes, the tee and pin as located by the georeference,
    and (if given) detected water hazards -- the one hazard colour-distinct
    enough from grass/sand/trees to segment reliably (see
    caddie.course.features.water_mask).

    ``until_idx`` truncates "this shot"'s track to that sample, with the ball
    marker riding the tip instead of sitting at the landing spot -- the
    mid-flight frame the animation steps through.

    ``zoom_radius_yd`` crops the view to that many yards around the pin,
    computed from the same georeference as everything else on this map -- so
    unlike the green-closeup illustration, the ball and pin markers in a
    zoomed frame are real plotted positions, not decoration.
    """
    img, W, H = get_map_image(str(hole_map.image_path))
    proj = MapProjection(hole_map, W, H, hole.tee, hole.pin)
    animating = until_idx is not None
    i = (until_idx + 1) if animating else len(data["shot_enu"])

    fig, ax = plt.subplots(figsize=(11, 11 * H / W), facecolor=p["surface"])
    # Nearest while playing -- cheaper resampling, and at animation speed the
    # difference isn't visible anyway. Bilinear for the static frame you
    # actually sit and look at.
    ax.imshow(img, interpolation="nearest" if animating else "bilinear")
    ax.set_axis_off()

    # Detected water hazards, drawn under everything else -- yellow/red like
    # real water-hazard stakes, not the sampled creek-blue, so it reads as a
    # highlight/warning rather than decoration. water_alpha is a continuous
    # 0 -> 0.5 -> 0 pulse driven by wall-clock time (see main()'s rerun loop
    # below), not a fixed value -- the "Pulse water hazards" checkbox is a
    # toggle, not a one-shot flash. Colour-segmented, not surveyed: expect
    # rough edges and the odd shadow-split creek into two blobs (see
    # caddie.course.features).
    for poly in water or []:
        wx, wy = zip(*[proj.to_pixels(e, n) for e, n in poly])
        ax.fill(wx, wy, facecolor="#f5d130", edgecolor="#d3271f",
                linewidth=1.4, alpha=water_alpha, zorder=2)

    def track(enu):
        px, py = zip(*[proj.to_pixels(e, n) for e, n in enu])
        return px, py

    if show_calm and not animating:
        px, py = track(data["calm_enu"])
        ax.plot(px, py, color=p["series_2"], lw=2.0, zorder=3,
                label="Default Trajectory")
        ax.plot(px[-1], py[-1], "o", ms=6, color=p["series_2"],
                mec=p["surface"], mew=2, zorder=5)

    px, py = track(data["shot_enu"][:i])
    ax.plot(px, py, color=p["series_1"], lw=2.5, zorder=4, label="My Shot")
    ax.plot(px[-1], py[-1], "o", ms=7, color=p["series_1"],
            mec=p["surface"], mew=2, zorder=6)

    # Origin and target: distinct shapes in ink, never a series hue.
    ox, oy = proj.to_pixels(*data["origin_enu"])
    txp, typ = proj.to_pixels(*data["target_enu"])
    ax.plot([ox], [oy], marker="s", ms=7, color=p["surface"],
            mec=p["ink"], mew=1.8, ls="none", zorder=7, label="Playing from")
    ax.plot([txp], [typ], marker="*", ms=13, color=p["surface"],
            mec=p["ink"], mew=1.8, ls="none", zorder=7, label="Target")

    _draw_pin_flag(ax, proj, hole, p)

    if not animating:  # legend text layout is real per-frame cost; skip while playing
        leg = ax.legend(loc="lower left", frameon=True, fontsize=9,
                        facecolor=p["surface"], edgecolor=p["axis"], framealpha=0.85)
        for t in leg.get_texts():
            t.set_color(p["ink_2"])

    if zoom_radius_yd is not None:
        cx, cy = proj.geo_to_pixels(hole.pin)
        r_px = (zoom_radius_yd * YARD_M) / proj.metres_per_pixel
        ax.set_xlim(cx - r_px, cx + r_px)
        ax.set_ylim(cy + r_px, cy - r_px)
    else:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
    fig.tight_layout(pad=0.2)
    return fig


def trajectory_table(data, every_s: float = 0.25):
    """The table-view twin. Every value on the charts is readable here."""
    shot = data["shot"]
    t = shot["times"]
    marks = np.arange(0.0, t[-1] + 1e-9, every_s)
    idx = np.unique(np.searchsorted(t, marks).clip(0, len(t) - 1))
    idx = np.unique(np.append(idx, len(t) - 1))
    return {
        "t (s)": np.round(t[idx], 2),
        "downrange (yd)": np.round(shot["x"][idx] / YARD_M, 1),
        "offline (yd)": np.round(shot["y"][idx] / YARD_M, 1),
        "height (yd)": np.round(shot["z"][idx] / YARD_M, 1),
        "speed (mph)": np.round(shot["speed"][idx] / MPH_MS, 1),
        "spin (rpm)": np.round(shot["spin"][idx], 0),
    }


def _round_scorecard(course) -> dict:
    """Wide scorecard: rows Par/Score/+/-, columns per hole plus OUT/IN/TOTAL.

    A hole only counts once its last saved shot is holed -- an in-progress
    hole shows as blank rather than a partial stroke count that would read
    as a finished score.
    """
    shots_by_hole = st.session_state.get("shots", {})
    holes = course.holes

    def hole_score(h) -> int | None:
        shots = shots_by_hole.get(h.number, [])
        if not shots or not shots[-1].get("holed", False):
            return None
        return sum(s.get("strokes", 1) for s in shots)

    def total(vals: list[int | None]) -> int | None:
        played = [v for v in vals if v is not None]
        return sum(played) if played else None

    def fmt(v: int | None) -> str:
        return "–" if v is None else str(v)

    def fmt_diff(score: int | None, par: int) -> str:
        return "–" if score is None else f"{score - par:+d}"

    scores = [hole_score(h) for h in holes]
    front, back = holes[:9], holes[9:]
    out_par, in_par = sum(h.par for h in front), sum(h.par for h in back)
    out_score, in_score = total(scores[:9]), total(scores[9:])
    total_par, total_score = out_par + in_par, total(scores)

    data: dict[str, list[str]] = {"": ["Par", "Score", "+/-"]}
    for h, score in zip(holes, scores):
        data[str(h.number)] = [str(h.par), fmt(score), fmt_diff(score, h.par)]
        if h.number == 9:
            data["OUT"] = [str(out_par), fmt(out_score), fmt_diff(out_score, out_par)]
    data["IN"] = [str(in_par), fmt(in_score), fmt_diff(in_score, in_par)]
    data["TOTAL"] = [str(total_par), fmt(total_score), fmt_diff(total_score, total_par)]
    return data


def _advance_to_next_hole(course, current_number: int) -> None:
    """Queue a jump to the next hole after holing out.

    Can't write ``st.session_state["hole_select"]`` directly here -- that
    widget already rendered earlier this run (inside ``sidebar()``), and
    Streamlit forbids mutating a widget's state after it has been
    instantiated in the same run. Stash the target instead; ``main()``
    applies it at the top of the next run, before the widget is created.
    """
    numbers = [h.number for h in course.holes]
    idx = numbers.index(current_number)
    if idx + 1 < len(numbers):
        st.session_state["_next_hole"] = numbers[idx + 1]


def _play_animation(build_figs, n_samples: int, n_frames: int = 50):
    """Step each of ``build_figs`` -- one or more ``(until_idx) -> Figure``
    callables -- across evenly time-spaced samples, in lockstep, each into
    its own placeholder. This is what makes "Let it fly!" animate the map
    AND the elevation view together instead of one playing while the other
    just sits on its finished, static frame.

    Real flight is 2-9 s; this compresses to a fixed frame count rather than
    trying to hit real-time. Render time (a full figure rebuild per frame,
    not the sleep below) still dominates -- see map_chart/elevation_chart's
    animating-only skips (legend, interpolation) for the other half of the
    speed budget.
    """
    if callable(build_figs):
        build_figs = [build_figs]
    idxs = np.linspace(0, n_samples - 1, min(n_frames, n_samples)).round().astype(int)
    placeholders = [st.empty() for _ in build_figs]
    for idx in idxs:
        for build_fig, placeholder in zip(build_figs, placeholders):
            fig = build_fig(int(idx))
            placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)  # one run can create dozens of figures fast; don't wait on GC
        time.sleep(0.005)
    return placeholders
    return placeholder


CAVEATS = """\
- **Carry only. No bounce or roll.** Total distance is not modelled, so *vs
  target* is where the ball lands, not where it stops.
- **No club recommendation.** This flies the shot you specify. Choosing a club
  needs the dispersion model and an expected-strokes surface, neither of which
  exists yet.
- **One shot, not a distribution.** A real shot is a spread; this is the centre
  of it with no error bars.
- **Apex runs high** on low-spin long clubs, so descent angle carries more
  uncertainty than carry does. Root cause is now identified: our fitted lift
  coefficient runs up to +48% high around spin ratios 0.14–0.20, measured
  against an independent C_L table (`aero.EMPIRICAL_CL_TABLE`).
- **Curvature inherits that same error.** Sideways force is `C_L·sin(tilt)`, so
  predicted curve is over-stated worst through the middle of the bag — woods and
  long irons — and is close for driver and wedges.
- **Flighting presets (knockdown / flighted) are hand-set**, not published
  launch data. The signs are right; the magnitudes are estimates.
- **Face & path uses an unvalidated D-plane model.** Its two constants are hand-
  set; `caddie.shot.calibrate_face_to_path` measures the important one from a
  single observed shot.
- **Terrain is interpolated between 3–5 surveyed points** per hole. It captures
  the 100 ft drop on 10; it has no green contours, bunkers, water or trees.
- **The pin is one representative point**, not a daily Masters placement.
- **Corridor exposure is hand-set, not measured.**
- **The hole map is an illustration, georeferenced by eye** from two points per
  hole (tee and green centre). Good enough to see which side of a bunker a ball
  finishes; not good for anything finer. The artwork is also not guaranteed to
  be to scale — if the overlay and the numbers disagree, trust the numbers.
- **"On the green" is a circle, not the real green outline.** No green polygon
  exists (see above), so the switch to the zoomed-in view is a generous 20 yd
  radius around the pin (`GREEN_RADIUS_YD`), not the actual putting surface.
  The zoom itself is the same georeferenced overlay, just cropped tighter; the
  green artwork shown alongside it is decorative only, with no ball or pin
  plotted on it.
- **Putting is declared, not simulated.** Landing within `HOLE_OUT_RADIUS_YD`
  (1.5 yd) of the pin holes out on the spot; anything else on the green offers
  a putt count to pick and record, not a modeled stroke — there is no green
  contour data (see above) to simulate one from."""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _slider_edge_labels(left: str, right: str) -> None:
    """A small left/right caption spanning a slider about to be drawn --
    Streamlit sliders can't label their own ends, so this sits just above
    one as the closest plain-Streamlit equivalent."""
    st.markdown(
        '<div style="display:flex; justify-content:space-between; '
        'font-size:0.75rem; opacity:0.65; margin-bottom:-0.5rem;">'
        f"<span>{left}</span><span>{right}</span></div>",
        unsafe_allow_html=True,
    )


def sidebar(course):
    """Draw every control. Returns the palette plus the model inputs."""
    st.subheader("🎨 Appearance")
    mode_choice = st.radio(
        "Theme", ["Auto", "Light", "Dark"], horizontal=True,
        label_visibility="collapsed",
    )
    if mode_choice == "Auto":
        try:  # available on newer Streamlit; fall back quietly
            detected = st.context.theme.type or "light"
        except Exception:
            detected = "light"
        mode = "dark" if detected == "dark" else "light"
    else:
        mode = mode_choice.lower()

    st.subheader("⛳ Hole")
    numbers = [h.number for h in course.holes]
    # Keyed so holing out can jump the selector to the next hole (see
    # _advance_to_next_hole) -- with a key, `index` only applies on the very
    # first render, so the default-to-1 lives in the session_state seed below
    # instead.
    st.session_state.setdefault("hole_select", 1)
    hole_number = st.selectbox(
        "Hole",
        numbers,
        key="hole_select",
        format_func=lambda n: f"{n} — {course.hole(n).name} (par {course.hole(n).par})",
        label_visibility="collapsed",
    )
    hole = course.hole(hole_number)

    # Playing a hole shot by shot: each saved shot's landing spot becomes an
    # origin option for the next one, most recent first, ahead of the static
    # tee/aim-point/pin waypoints.
    st.session_state.setdefault("shots", {})
    hole_shots = st.session_state["shots"].get(hole_number, [])
    origin_points: dict = {}
    for s in reversed(hole_shots):
        origin_points[s["label"]] = s["landing"]
    origin_points.update(waypoints(hole))
    names = list(origin_points.keys())

    pending = st.session_state.pop("pending_origin", None)
    if pending in names:
        default_origin = pending
    elif hole_shots:
        default_origin = hole_shots[-1]["label"]
    else:
        default_origin = "Tee"

    # Keyed per hole so switching holes re-defaults to the tee instead of
    # Streamlit carrying over whatever was selected on the previous hole.
    # A keyed widget prefers its persisted session-state value over `index`
    # on every rerun after the first, so `index` alone can't move it once a
    # shot is saved -- the selection has to be pushed into session_state
    # directly whenever the freshly computed default should win: a new hole,
    # a just-reset hole, or a shot that was just saved (pending).
    origin_key = f"origin_select::{hole_number}"
    if pending is not None or st.session_state.get(origin_key) not in names:
        st.session_state[origin_key] = default_origin

    origin_name = st.selectbox("Playing from", names, key=origin_key)
    origin_geo = origin_points[origin_name]

    # Wind first, before club or shape: it's the condition you're playing
    # into, not a choice you make, so it should be known before deciding
    # what club to hit or how to shape the shot -- not buried after them.
    st.subheader("💨 Wind")
    # Randomized once per hole rather than fixed -- keyed per hole number so
    # arriving at (or resetting) a hole rolls fresh speed/direction, but
    # rerunning the script mid-hole (new club, new slider) doesn't shuffle
    # the wind out from under an in-progress shot.
    wind_key, dir_key = f"wind_mph::{hole_number}", f"wind_dir::{hole_number}"
    if wind_key not in st.session_state:
        # Centred on a 5 mph average, tight spread -- the slider itself is
        # still open to 40 mph if you want to dial up something wild by hand.
        st.session_state[wind_key] = round(random.uniform(3.0, 7.0) * 2) / 2
    if dir_key not in st.session_state:
        st.session_state[dir_key] = random.choice(list(WIND_CARDINALS.keys()))
    wind_mph = st.slider("Speed at 10 m (mph)", 0.0, 40.0, step=0.5, key=wind_key)
    cardinal = st.selectbox(
        "Direction (blowing FROM)", list(WIND_CARDINALS.keys()), key=dir_key
    )
    fine = st.slider("…offset (deg)", -45, 45, 0, 5)
    wind_dir = (WIND_CARDINALS[cardinal] + fine) % 360

    exposure = st.selectbox(
        "Corridor exposure", list(SHELTER_PRESETS.keys()) + ["Custom"]
    )
    if exposure == "Custom":
        z0 = st.slider("Roughness length z0 (m)", 0.01, 1.5, 0.8, 0.01)
        shelter = st.slider("Shelter factor below treeline", 0.0, 1.0, 0.45, 0.05)
        treeline = st.slider("Treeline height (m)", 1.0, 45.0, 30.0, 1.0)
    else:
        preset = SHELTER_PRESETS[exposure]
        z0 = preset["roughness_length"]
        shelter = preset["shelter_factor"]
        treeline = preset["treeline_height"]
    veer = st.slider("Veer (deg per 100 m of height)", -30.0, 30.0, 0.0, 1.0)

    st.subheader("🎒 My Bag")
    st.caption(
        f"USGA limit: 14 clubs including a putter -- {BAG_LIMIT} here, since "
        "putting is a declared result in this app, not a simulated club "
        "(see the caveats panel)."
    )
    full_pool_names = [c.name for c in FULL_CLUB_POOL]
    st.session_state.setdefault("my_bag", list(DEFAULT_BAG_NAMES))
    bag_selection = st.multiselect(
        "Clubs in your bag", full_pool_names, max_selections=BAG_LIMIT,
        key="my_bag",
        help="Everything past PW is an experimental variant -- extrapolated "
             "numbers, not tour-sourced (see caddie.physics.ball). Pick "
             "whatever combo matches your real bag.",
    )
    if not bag_selection:
        bag_selection = list(DEFAULT_BAG_NAMES)  # an empty bag can't play a hole
    # Preserve pool order (woods -> irons -> wedges) regardless of click order.
    BAG = [c for c in FULL_CLUB_POOL if c.name in bag_selection]

    use_own_stats = st.checkbox(
        "Play with my own carry distances instead of PGA Tour averages",
        help="Club management under YOUR physical constraints, not a tour "
             "pro's -- distances drive club choice and layup decisions the "
             "same way, they're just yours.",
    )
    if use_own_stats:
        st.session_state.setdefault(
            "bag_carries", {c.name: c.published_carry_yd for c in FULL_CLUB_POOL}
        )
        bag_rows = st.data_editor(
            [
                {"Club": c.name, "My carry (yd)": st.session_state["bag_carries"].get(
                    c.name, c.published_carry_yd
                )}
                for c in BAG
            ],
            column_config={
                "Club": st.column_config.TextColumn(disabled=True),
                "My carry (yd)": st.column_config.NumberColumn(
                    min_value=20.0, max_value=350.0, step=1.0,
                ),
            },
            hide_index=True, key="bag_editor", width="stretch",
        )
        st.session_state["bag_carries"].update(
            {row["Club"]: float(row["My carry (yd)"]) for row in bag_rows}
        )

    # Club is picked before aiming, not after, because the aim distance below
    # is capped by what THIS club can carry -- a pitching wedge shouldn't be
    # allowed to aim 280 yd out just because a driver could.
    st.subheader("🏌️ Shot")
    # Off the tee, driver's fine. Anywhere else, it's locked out -- hitting
    # driver off the deck is a real shot but a low-percentage one, and this
    # is a blunt "not yet" rather than modeling who can pull it off.
    driver_locked = origin_name != "Tee"
    club_names = [
        c.name for c in BAG if not (driver_locked and c.name == "Driver")
    ]
    if not club_names:  # the bag really was Driver-only -- fall back rather than crash
        club_names = [c.name for c in BAG] or full_pool_names
    non_driver_fallback = next((n for n in club_names if n != "Driver"), club_names[0])
    # Keyed per hole and seeded to Driver -- every hole starts as a tee shot,
    # so that's the sane default club, not whatever was last hit on the
    # previous hole. Picking a different club afterward is remembered for
    # this hole same as origin_select is.
    club_key = f"club_select::{hole_number}"
    if club_key not in st.session_state:
        st.session_state[club_key] = (
            "Driver" if not driver_locked and "Driver" in club_names else non_driver_fallback
        )
    elif st.session_state[club_key] not in club_names:
        st.session_state[club_key] = non_driver_fallback
    club_name = st.selectbox(
        "Club", club_names, key=club_key,
        help="Driver is locked out once you're off the tee." if driver_locked else None,
    )
    club = next(c for c in BAG if c.name == club_name)
    is_wedge_estimate = club_name in {c.name for c in EXTRA_VARIANTS}
    if is_wedge_estimate:
        st.caption(
            f"{club_name} launch numbers are extrapolated, not tour-sourced "
            "-- see caddie.physics.ball.EXTRA_VARIANTS. Dial in My Bag above."
        )

    if use_own_stats:
        my_carry_yd = st.session_state["bag_carries"][club_name]
        my_speed = solve_speed_for_carry(club.launch_angle_deg, club.backspin_rpm, my_carry_yd)
        st.caption(
            f"→ solved ball speed ≈ {my_speed:.0f} mph for a {my_carry_yd:.0f} yd "
            f"{club_name}, holding tour-average launch angle/spin. No wind, "
            "sea level."
        )
    else:
        my_carry_yd = club.published_carry_yd
        my_speed = float(club.ball_speed_mph)

    # Trackman's published carry is the tour *average*; the same player's
    # longest ball with that club runs 10-15% past their own average --
    # applied to whichever carry (tour or personal) is actually in play.
    club_carry_cap_yd = my_carry_yd * 1.15

    # "Aiming at" is untouched by the custom-aim feature below -- it only
    # ever lists the hole's own waypoints (Aim point 1, Pin, ...), so picking
    # one is never confused with overriding it.
    targets = [n for n in names if n != origin_name]
    if origin_name == "Tee" and "Aim point 1" in targets:
        default_target = "Aim point 1"
    elif origin_geo.distance_yards_to(hole.pin) <= club_carry_cap_yd:
        # Close enough that the selected club could plausibly reach the pin
        # -- go straight at the flag rather than a layup waypoint.
        default_target = "Pin"
    else:
        # Still too far for this club -- default to the aim point nearest
        # the green among the ones still on offer, not straight at a pin
        # that's out of reach.
        remaining_aim_points = [n for n in targets if n.startswith("Aim point")]
        default_target = remaining_aim_points[-1] if remaining_aim_points else "Pin"
    target_name = st.selectbox(
        "Aiming at", targets, index=targets.index(default_target)
    )
    target_geo = origin_points[target_name]

    use_custom_aim = st.checkbox(
        "🎯 Override with a custom aim point",
        help="Aim anywhere, not just the waypoint picked above — combine "
             "with Curve below for shot shapes the hole itself doesn't "
             "suggest.",
    )
    if use_custom_aim:
        ref_frame = hole.frame_from(origin_geo, hole.pin)
        pin_x, _, _ = ref_frame.to_local(hole.pin)
        aim_dist_max = min(max(60.0, pin_x / YARD_M * 1.6), club_carry_cap_yd)
        aim_dist = st.slider(
            "Aim distance (yd)", 0.0, aim_dist_max,
            min(max(0.0, float(pin_x / YARD_M)), aim_dist_max), 1.0,
            help=f"Capped at {aim_dist_max:.0f} yd — roughly the longest "
                 f"{'you' if use_own_stats else 'a tour pro'} carr{'y' if use_own_stats else 'ies'} "
                 f"a {club_name}, not just the average. Change club above to "
                 "raise or lower this.",
        )
        _slider_edge_labels("⬅ Left", "Right ➡")
        aim_angle_deg = st.slider(
            "Aim offline (deg)", -60.0, 60.0, 0.0, 0.5,
            help="An angle off the aim line, not a raw yardage -- the same "
                 "5 deg swings a much bigger gap at 250 yd than it does at "
                 "50 yd, which a fixed yard offset can't express.",
        )
        aim_offline = aim_dist * float(np.tan(np.radians(aim_angle_deg)))
        target_geo = ref_frame.to_geo(aim_dist * YARD_M, aim_offline * YARD_M, 0.0)

    if use_custom_aim:
        # Custom aim sets a distance, not just a direction -- without this,
        # "aim 150 yd out" pointed the shot that way but still flew however
        # far the club's regular speed happened to carry, which read as the
        # aim distance doing nothing. Same solve My Bag uses for a personal
        # carry, just targeting the aim distance instead.
        my_speed = solve_speed_for_carry(club.launch_angle_deg, club.backspin_rpm, aim_dist)
        my_carry_yd = aim_dist

    # Keying the widgets on the club name makes changing club reset the sliders
    # to that club's launch numbers, which is the behaviour you want. The
    # speed key also folds in the personal carry (My Bag) or aim distance
    # (custom aim) when either is active, so editing either re-seeds the
    # slider instead of leaving it stuck on a value solved for whichever
    # number you just changed away from.
    if use_custom_aim:
        speed_key = f"speed::{club_name}::aim::{aim_dist:.0f}"
    elif use_own_stats:
        speed_key = f"speed::{club_name}::own::{my_carry_yd:.0f}"
    else:
        speed_key = f"speed::{club_name}"
    speed = st.slider(
        "Ball speed (mph)", 60.0, 190.0, float(my_speed), 0.5,
        key=speed_key,
    )
    if club_name == "Driver":
        # A live floating tooltip while dragging would need a custom JS
        # component -- Streamlit only sees the value once the drag commits.
        # This is the closest plain-Streamlit equivalent: a badge that
        # updates to whichever tier you've reached the moment you let go.
        DRIVER_SPEED_TIERS = [
            (175.0, "🏆 PGA Tour speed"),
            (166.0, "⛳ Scratch golfer"),
            (133.0, "🙂 Amateur average"),
        ]
        tier_label = next(
            (label for threshold, label in DRIVER_SPEED_TIERS if speed >= threshold),
            "🌱 Below amateur average",
        )
        st.markdown(f"**{tier_label}**")
        st.caption("Driver ball speed reference: 133 amateur → 166 scratch → 175+ tour.")
    angle = st.slider(
        "Launch angle (deg)", 2.0, 50.0, float(club.launch_angle_deg), 0.1,
        key=f"angle::{club_name}",
    )
    spin = st.slider(
        "Backspin (rpm)", 1000.0, 13000.0, float(club.backspin_rpm), 50.0,
        key=f"spin::{club_name}",
    )
    st.subheader("🌀 Shape")
    hand = st.radio("Handedness", ["RH", "LH"], horizontal=True)
    height = st.selectbox(
        "Trajectory", list(FLIGHT_HEIGHTS), index=list(FLIGHT_HEIGHTS).index("stock"),
        help="Flighting magnitudes are hand-set, not published data.",
    )
    mode_label = st.radio(
        "Specify the shape by",
        ["Curve + auto-aim", "Face & path", "Raw launch numbers"],
        help=(
            "Curve + auto-aim solves the spin axis and start line from the "
            "validated physics. Face & path uses the D-plane model, whose two "
            "constants are hand-set."
        ),
    )
    shape_mode = {
        "Curve + auto-aim": "shape",
        "Face & path": "swing",
        "Raw launch numbers": "manual",
    }[mode_label]

    curve_yards, azimuth, sidespin = 0.0, 0.0, 0.0
    face_deg, path_deg = 0.0, 0.0
    face_w, axis_per_ftp = 0.78, 3.0

    if shape_mode == "shape":
        curve_yards = st.slider(
            "Curve (yd, + = to the player's right)", -40.0, 40.0, 0.0, 1.0,
            help="Bend away from the start line. It solves the aim so the ball "
                 "still finishes on the target line.",
        )
        shape_preview = ShotShape(curve_yards=curve_yards, height=height, hand=hand)
        st.caption(f"→ {shape_preview.describe()}")
    elif shape_mode == "swing":
        _slider_edge_labels("⬅ Closed", "Open ➡")
        face_deg = st.slider("Club face (deg)", -8.0, 8.0, 0.0, 0.25)
        _slider_edge_labels("⬅ Out-to-in", "In-to-out ➡")
        path_deg = st.slider("Club path (deg)", -12.0, 12.0, 0.0, 0.25)
        st.caption(f"→ face-to-path {face_deg - path_deg:+.2f}deg")
        with st.expander("D-plane constants (hand-set)"):
            face_w = st.slider("Start line from face", 0.5, 1.0, 0.78, 0.01)
            axis_per_ftp = st.slider(
                "Spin axis deg per deg face-to-path", 0.5, 8.0, 3.0, 0.1,
                help="The one number with no published value. "
                     "caddie.shot.calibrate_face_to_path measures it from a "
                     "single observed shot.",
            )
    else:
        azimuth = st.slider("Start direction (deg, + = right)", -15.0, 15.0, 0.0, 0.5)
        sidespin = st.slider(
            "Sidespin (rpm, + = curves right)", -1500.0, 1500.0, 0.0, 25.0
        )

    st.subheader("🌤️ Air")
    temp_c = st.slider("Temperature (°C)", -5.0, 40.0, 22.0, 0.5)
    rh = st.slider("Relative humidity", 0.0, 1.0, 0.55, 0.05)
    altimeter = st.slider("Sea-level pressure (hPa)", 970.0, 1050.0, 1013.25, 0.25)

    firmness_presets = {
        "Soft / wet": 0.5, "Normal fairway": 1.0, "Firm / dry": 1.45,
    }
    firmness_label = st.select_slider(
        "Ground firmness (roll)", list(firmness_presets.keys()), value="Normal fairway",
        help="Multiplies the estimated roll after landing -- see "
             "caddie.physics.roll. Roll itself is a rough empirical model, "
             "not a simulation: spin, landing angle, slope and moisture all "
             "matter and none of them feed into it.",
    )
    firmness = firmness_presets[firmness_label]

    show_calm = st.checkbox(
        "Overlay the unshaped Default Trajectory", value=True,
        help="The plain stock swing, aimed at the same target, with no "
             "curve/flighting/face-path applied and no correction for wind.",
    )
    show_map = st.checkbox(
        "Draw the shot on the hole map", value=True,
        help="Uncheck for the abstract plan view, which has a readable "
             "offline scale. The map is an illustration, georeferenced by eye.",
    )
    show_water = st.checkbox(
        "Pulse water hazards", value=True,
        help="Colour-segmented from the map artwork, not surveyed -- expect "
             "rough edges and the odd creek split into a couple of pieces by "
             "shadow. See caddie.course.features.water_mask. While this is "
             "on, the page auto-refreshes to animate the pulse, which costs "
             "some responsiveness elsewhere -- leave it off unless you're "
             "actively looking for hazards.",
    )

    def to_point(geo):
        return (geo.lat, geo.lon, geo.elevation_m)

    return dict(
        mode=mode,
        hole=hole,
        origin_name=origin_name,
        target_name=CUSTOM_AIM if use_custom_aim else target_name,
        origin_point=to_point(origin_points[origin_name]),
        target_point=to_point(target_geo),
        club_name=club_name,
        hole_shots=hole_shots,
        launch_params=(speed, angle, spin, azimuth, sidespin),
        wind_params=(wind_mph, wind_dir, z0, shelter, treeline, veer),
        air_params=(temp_c, rh, altimeter),
        shape_params=(
            shape_mode, curve_yards, height, hand, face_deg, path_deg,
            face_w, axis_per_ftp,
        ),
        loft_deg=club.loft_deg if club.loft_deg is not None else club.launch_angle_deg,
        firmness=firmness,
        show_calm=show_calm,
        show_map=show_map,
        show_water=show_water,
    )


def _inject_style() -> None:
    """Light CSS polish -- spacing, rounded cards, a calmer sidebar. Doesn't
    touch chart colors (those come from PALETTES) or Streamlit's own
    light/dark chrome (that stays with the browser/OS preference; only
    ``primaryColor`` is pinned, in .streamlit/config.toml)."""
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap"
              rel="stylesheet">
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.15);
        }
        section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
        h3 {
            padding-bottom: 0.3rem;
            margin-top: 1.1rem !important;
            border-bottom: 1px solid rgba(128, 128, 128, 0.18);
        }
        div[data-testid="stMetric"] {
            background: rgba(47, 122, 79, 0.07);
            border: 1px solid rgba(47, 122, 79, 0.18);
            border-radius: 0.6rem;
            padding: 0.6rem 0.9rem;
        }
        div[data-testid="stMetricValue"] {font-weight: 700;}
        .stButton > button, .stDownloadButton > button {
            border-radius: 0.5rem;
            transition: transform 0.05s ease-in-out;
        }
        .stButton > button:active {transform: scale(0.98);}
        div[data-testid="stExpander"] {
            border-radius: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border-radius: 0.5rem; overflow: hidden;
        }
        .hole-badge {
            vertical-align: middle;
            font-size: 1.5rem; font-weight: 800; letter-spacing: 0.02em;
            color: #3a9463;
        }
        .hole-name {
            font-family: 'Libre Baskerville', serif;
            font-style: italic; font-size: 1.5rem; font-weight: 700;
            vertical-align: middle;
        }
        .par-badge {
            display: inline-flex; align-items: center;
            background: rgba(47, 122, 79, 0.16);
            border: 1px solid rgba(47, 122, 79, 0.4);
            border-radius: 999px; padding: 0.15rem 0.7rem;
            font-weight: 700; font-size: 0.85rem; vertical-align: middle;
        }
        .stat-row {
            display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.55rem;
        }
        .stat-chip {
            display: inline-flex; align-items: center; gap: 0.3rem;
            background: rgba(128, 128, 128, 0.09);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 999px; padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
        }
        .stat-chip b {font-variant-numeric: tabular-nums; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "logo.png"


def main() -> None:
    st.set_page_config(
        page_title="Masters Manager — trajectory explorer",
        page_icon=str(LOGO_PATH) if LOGO_PATH.is_file() else "⛳",
        layout="wide",
    )
    _inject_style()
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=340)
    else:
        st.markdown(
            '<div style="display:flex; align-items:baseline; gap:0.6rem; '
            'margin-bottom:0.2rem;">'
            '<span style="font-size:1.7rem;">⛳</span>'
            '<span style="font-size:1.7rem; font-weight:800;">Masters Manager</span>'
            '<span style="opacity:0.6; font-size:0.95rem;">— trajectory explorer</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    course = get_course()

    # Apply a hole jump queued by _advance_to_next_hole before the "Hole"
    # widget (inside sidebar()) is instantiated for this run.
    next_hole = st.session_state.pop("_next_hole", None)
    if next_hole is not None:
        st.session_state["hole_select"] = next_hole

    with st.sidebar:
        cfg = sidebar(course)

    p = PALETTES[cfg["mode"]]
    hole = cfg["hole"]

    data = fly(
        hole.number,
        cfg["origin_point"],
        cfg["target_point"],
        cfg["launch_params"],
        cfg["wind_params"],
        cfg["air_params"],
        cfg["shape_params"],
        cfg["loft_deg"],
        cfg["firmness"],
    )

    shot, calm = data["shot"], data["calm"]
    tx, ty, tz = data["target_local"]
    target_yards = tx / YARD_M
    vs_target = shot["carry"] - target_yards
    dz_m = (data["target_elev_m"] or 0.0) - (data["origin_elev_m"] or 0.0)

    inf = get_hole_info().get(hole.number, {})
    difficulty_chip = ""
    if inf.get("historicalRank"):
        # Scoring difficulty is the one thing here measured from actual play
        # rather than derived from geometry, so it is worth showing.
        difficulty_chip = (
            '<span class="stat-chip">📊 '
            f"<b>{inf['historicalAverage']:.3f}</b> strokes "
            f"({inf['historicalAverage'] - hole.par:+.3f} to par) · "
            f"<b>{inf['historicalRank']}</b><sup>th</sup> hardest</span>"
        )

    header_col, scorecard_col = st.columns([6, 1])
    with scorecard_col:
        with st.popover("📇 Scorecard", use_container_width=True):
            # st.table, not st.dataframe: the dataframe widget virtualizes
            # into a fixed-size viewport with its own scrollbars no matter
            # how wide the popover grows. st.table renders the full static
            # HTML table at its natural size instead.
            st.table(_round_scorecard(course))
    with header_col:
        st.markdown(
            f'<span class="hole-badge">No. {hole.number}</span> '
            f'<span class="hole-name">{hole.name}</span> '
            f'<span class="par-badge">Par {hole.par}</span>'
            '<div class="stat-row">'
            f'<span class="stat-chip">🎯 {cfg["origin_name"]} → {cfg["target_name"]}</span>'
            f'<span class="stat-chip">📏 <b>{target_yards:.0f}</b> yd on the line</span>'
            f'<span class="stat-chip">⛰️ <b>{dz_m / FT_M:+.0f}</b> ft elevation</span>'
            f'<span class="stat-chip">🃏 card <b>{hole.card_yardage}</b> yd</span>'
            f'<span class="stat-chip">↩️ dogleg <b>{hole.dogleg_deg:.0f}°</b></span>'
            f"{difficulty_chip}"
            "</div>",
            unsafe_allow_html=True,
        )

    sh = data["shaping"]
    if sh.get("error"):
        st.warning(f"Shape not reachable — {sh['error']}")
    elif sh["mode"] == "shape" and not sh.get("converged", True):
        st.warning(
            "The curve/aim solve did not fully converge; the numbers below are "
            "the last iterate."
        )

    if sh["mode"] != "manual" and sh.get("spin_axis_deg") is not None:
        s = st.columns(4)
        s[0].metric("Solved aim", f"{sh['solved_aim_deg']:+.1f}°")
        s[1].metric("Spin axis", f"{sh['spin_axis_deg']:+.1f}°")
        if sh["mode"] == "shape":
            s[2].metric("Curve achieved", f"{sh['curve_achieved']:+.1f} yd")
            s[3].metric(
                "Cost of shaping", f"{sh['shaping_cost']:+.1f} yd",
                help="Carry given up versus the same club hit straight. A "
                     "consequence of tilting lift out of the vertical, not an "
                     "applied penalty.",
            )
        else:
            s[2].metric("Face-to-path", f"{sh['face_to_path']:+.2f}°")
            s[3].metric("Start line", f"{sh['solved_aim_deg']:+.1f}°")

    c = st.columns(8)
    c[0].metric("Carry", f"{shot['carry']:.1f} yd", f"{vs_target:+.1f} vs target")
    c[1].metric(
        "Roll (est.)", f"{shot['roll_yd']:.1f} yd",
        help="Empirical estimate, not simulated -- see caddie.physics.roll "
             "and the ground firmness control in Air.",
    )
    c[2].metric("Total distance", f"{shot['total_yd']:.1f} yd")
    c[3].metric(
        "Offline", f"{shot['offline']:+.1f} yd",
        f"{shot['offline'] - calm['offline']:+.1f} from wind",
    )
    c[4].metric("Apex", f"{shot['apex']:.1f} yd")
    c[5].metric("Descent", f"{shot['descent']:.1f}°")
    c[6].metric("Flight time", f"{shot['flight_time']:.2f} s")
    c[7].metric(
        "Wind effect", f"{shot['carry'] - calm['carry']:+.1f} yd",
        f"ρ {data['density_ratio'] * 100:.1f}% ISA", delta_color="off",
    )

    # --- Play the hole shot by shot -----------------------------------------
    hole_shots = cfg["hole_shots"]
    holed_out = bool(hole_shots) and hole_shots[-1].get("holed", False)
    about_to_hole = data["total_distance_to_pin_yd"] <= HOLE_OUT_RADIUS_YD

    btn = st.columns([2.6, 2.6, 1.6, 3.4])
    play_clicked = btn[0].button("▶ Let it fly!")
    # Deliberately separate from "Let it fly!" -- this saves immediately,
    # with no animation, for anyone who just wants the result.
    save_label = "⛳ Holes out — save it!" if about_to_hole else "✅ Shot completed"
    save_clicked = btn[1].button(save_label)
    reset_clicked = btn[2].button("↺ Reset hole") if hole_shots else False
    # Always the REAL distance from where you're actually standing (the
    # chosen origin) to the pin -- never the previewed/not-yet-saved shot's
    # landing spot. That preview distance briefly lived here (see git
    # history) and it read as flatly wrong off the tee: a 323 yd hole would
    # show something like "120 yd" because that's where the currently
    # dialled-in club happens to land, not where you're standing.
    btn[3].metric("Remaining to pin", f"{data['origin_distance_to_pin_yd']:.0f} yd")

    if reset_clicked:
        st.session_state["shots"][hole.number] = []
        st.session_state.pop(f"origin_select::{hole.number}", None)
        # Restarting the hole rerolls its wind too -- popped, not reassigned,
        # since the widgets already rendered this run (see sidebar()); the
        # seeding check there rolls fresh values next run.
        st.session_state.pop(f"wind_mph::{hole.number}", None)
        st.session_state.pop(f"wind_dir::{hole.number}", None)
        st.session_state.pop(f"club_select::{hole.number}", None)
        st.rerun()

    if save_clicked:
        n = len(hole_shots) + 1
        if about_to_hole:
            label = f"Shot {n}: {cfg['club_name']} → holed! ({shot['total_yd']:.0f} yd)"
        else:
            side = (
                "right" if shot["offline"] > 0.5
                else "left" if shot["offline"] < -0.5
                else "straight"
            )
            label = (
                f"Shot {n}: {cfg['club_name']} → {_distance_label(shot)} "
                f"({abs(shot['offline']):.0f} yd {side}), "
                f"{data['total_distance_to_pin_yd']:.0f} yd left"
            )
        st.session_state["shots"].setdefault(hole.number, []).append(dict(
            label=label,
            landing=data["final_geo"],  # rolled-out resting spot, not just carry landing
            club=cfg["club_name"],
            carry=shot["carry"],
            offline=shot["offline"],
            distance_to_pin_yd=data["total_distance_to_pin_yd"],
            holed=about_to_hole,
            strokes=1,
        ))
        if about_to_hole:
            _advance_to_next_hole(course, hole.number)
        else:
            st.session_state["pending_origin"] = label
        st.rerun()

    if holed_out:
        # A "3 putts" entry is one list item but three strokes, so the score
        # is the sum of `strokes`, not the shot count.
        total = sum(s.get("strokes", 1) for s in hole_shots)
        st.success(f"⛳ Holed out in {total} ({total - hole.par:+d} to par)")

    if hole_shots:
        shots_so_far = sum(s.get("strokes", 1) for s in hole_shots)
        with st.expander(f"Shots played this hole ({shots_so_far})"):
            st.dataframe({
                "#": list(range(1, len(hole_shots) + 1)),
                "club": [s["club"] for s in hole_shots],
                "strokes": [s.get("strokes", 1) for s in hole_shots],
                "carry (yd)": [round(s["carry"], 1) for s in hole_shots],
                "offline (yd)": [round(s["offline"], 1) for s in hole_shots],
                "left to pin (yd)": [
                    round(s["distance_to_pin_yd"], 1) for s in hole_shots
                ],
            }, width="stretch")

    # Stale gate removed: this used to hide "Default Trajectory" whenever
    # wind was 0, back when no-wind meant the two lines were always
    # identical. They can now diverge from shaping alone, wind or not, so
    # only the user's own checkbox should control visibility.
    show_calm = cfg["show_calm"]

    hole_map = get_hole_maps().get(hole.number)
    has_map = hole_map is not None and hole_map.exists() and cfg["show_map"]
    hole_water = get_hole_water(hole.number) if has_map and cfg["show_water"] else []
    # Continuous 0 -> peak -> 0 triangle wave, driven by wall-clock time so
    # its speed doesn't depend on how often the page happens to rerun.
    # Streamlit has no background animation -- the rerun loop at the bottom
    # of this function is what actually keeps this moving.
    if hole_water:
        _WATER_PULSE_PERIOD_S = 2.4
        _WATER_PULSE_PEAK = 0.2
        _phase = (time.time() % _WATER_PULSE_PERIOD_S) / _WATER_PULSE_PERIOD_S
        water_alpha = (1.0 - abs(2.0 * _phase - 1.0)) * _WATER_PULSE_PEAK
    else:
        water_alpha = 0.0
    on_green = data["total_distance_to_pin_yd"] <= GREEN_RADIUS_YD
    # Putting only kicks in for a shot you've actually committed (Save), not
    # just one you're still previewing -- it changes your score.
    on_green_committed = (
        bool(hole_shots) and not holed_out
        and hole_shots[-1]["distance_to_pin_yd"] <= GREEN_RADIUS_YD
    )
    green_path = hole_map.green_image_path if hole_map is not None else None

    # Animate over the full hole first -- the moving ball needs the whole
    # flight path in frame, not a crop around the pin.
    animated_map = animated_elevation = False
    if play_clicked:
        n_samples = len(shot["x"])
        # Both views animate together, in lockstep -- previously only
        # whichever one came first (the map, when available) actually
        # played, and the other just sat on its finished static frame.
        builders = []
        if has_map:
            builders.append(
                lambda idx: map_chart(data, p, hole, hole_map, show_calm, until_idx=idx)
            )
            animated_map = True
        builders.append(
            lambda idx: elevation_chart(data, p, show_calm, until_idx=idx)
        )
        animated_elevation = True
        _play_animation(builders, n_samples)
    if has_map and on_green:
        # Zoom the SAME georeferenced overlay in around the pin, rather than
        # swapping to the green-closeup illustration -- that illustration has
        # no georeference (see caveats), so it can't show where the ball or
        # the hole actually are. This can, because it's the real plotted
        # positions, just cropped tighter.
        st.pyplot(
            map_chart(
                data, p, hole, hole_map, show_calm, zoom_radius_yd=GREEN_ZOOM_RADIUS_YD,
                water=hole_water, water_alpha=water_alpha,
            ),
            clear_figure=True,
        )
        st.caption(
            f"On the green — {data['distance_to_pin_yd']:.0f} yd "
            f"({data['distance_to_pin_yd'] * 3:.0f} ft) to the pin. Zoomed to "
            f"{GREEN_ZOOM_RADIUS_YD:.0f} yd around the pin from the same "
            "georeferenced overlay above, not the marketing artwork."
        )
        if green_path is not None and green_path.is_file():
            with st.expander("Green artwork (illustration only — does not show your ball or the pin)"):
                st.image(str(green_path), width="stretch")
    elif has_map and not animated_map:
        st.pyplot(
            map_chart(data, p, hole, hole_map, show_calm, water=hole_water, water_alpha=water_alpha),
            clear_figure=True,
        )

    if on_green_committed:
        # Putting isn't modeled at all (no green contours -- see caveats), so
        # this doesn't simulate a stroke, it just lets you declare one and
        # move on, the same way you'd concede a round to a rules-of-golf
        # gimme rather than replaying it.
        last_dist_yd = hole_shots[-1]["distance_to_pin_yd"]
        if last_dist_yd <= 5:
            # Gimme range -- assume the putt is a single extra stroke.
            default_putts = 1
        elif last_dist_yd <= 10:
            # No green-contour model to lean on (see caveats), so treat it as
            # a coin flip between a made putt and a two-putt.
            default_putts = random.choice([1, 2])
        else:
            default_putts = 2
        pc = st.columns([2, 2, 4])
        putts = pc[0].selectbox(
            "Putts to finish", [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(default_putts), key="putts_select",
        )
        if pc[1].button("⛳ Hole out"):
            n = len(hole_shots) + 1
            st.session_state["shots"][hole.number].append(dict(
                label=f"Shot {n}: {putts} putt" + ("s" if putts > 1 else "") + " — holed",
                landing=hole.pin,
                club=f"{putts}-putt" + ("s" if putts > 1 else ""),
                carry=hole_shots[-1]["distance_to_pin_yd"],
                offline=0.0,
                distance_to_pin_yd=0.0,
                holed=True,
                strokes=putts,
            ))
            _advance_to_next_hole(course, hole.number)
            st.rerun()
        pc[2].caption(
            "Putting isn't modeled — see caveats. This records a declared "
            "result rather than simulating the green."
        )

    if not animated_elevation:
        st.pyplot(elevation_chart(data, p, show_calm), clear_figure=True)
    if not has_map:
        # The map already shows the ground track; the abstract plan view is a
        # fallback for holes with no diagram, and a way to read exact offline.
        st.pyplot(plan_chart(data, p, show_calm), clear_figure=True)

    st.caption(
        f"Shot: {sh.get('label', '')} · {data['launch_describe']} · "
        f"Air: {data['air_describe']} · Wind: {data['wind_describe']} · "
        "Lateral scale on the plan view is expanded relative to downrange so the "
        "curve is visible — read both axes."
    )

    left, right = st.columns([3, 2])

    with left:
        with st.expander("Table view — every plotted value", expanded=False):
            st.dataframe(trajectory_table(data), width="stretch", height=320)
            st.markdown(
                f"**My Shot** · carry {shot['carry']:.1f} yd · offline "
                f"{shot['offline']:+.1f} yd · apex {shot['apex']:.1f} yd · descent "
                f"{shot['descent']:.1f}° · impact {shot['impact_speed']:.1f} m/s, "
                f"{shot['impact_spin']:.0f} rpm  \n"
                f"**Default Trajectory** · carry {calm['carry']:.1f} yd · offline "
                f"{calm['offline']:+.1f} yd · apex {calm['apex']:.1f} yd · descent "
                f"{calm['descent']:.1f}°"
            )

    with right:
        with st.expander(
            "What this does not know — read before trusting a number", expanded=True
        ):
            st.markdown(CAVEATS)

    if hole_water:
        # What actually makes the pulse "continuous": Streamlit has no
        # background animation, so this reruns the whole script on a short
        # timer for as long as "Pulse water hazards" stays checked. That's a
        # real cost -- every rerun redraws the entire page, not just the
        # water -- which is why the checkbox defaults off and says so.
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
