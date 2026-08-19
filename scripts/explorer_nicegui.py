"""AI Caddie -- NiceGUI trajectory explorer.

Run:  python scripts/explorer_nicegui.py
Opens at http://localhost:8502 by default.

A redesigned front end over the exact same physics/course/shot-shaping
engine as scripts/explorer.py (the Streamlit app). Chart-building and
shot-resolution code is imported from there rather than duplicated, so both
UIs share one source of truth for what a shot actually does -- this file is
presentation only.

Why NiceGUI over Streamlit here: Streamlit reruns the whole script top to
bottom on every interaction, which is what forced a lot of the session_state
gymnastics in explorer.py (seed a widget's state before it renders, or a
stale value wins over a fresh default -- see origin_select/hole_select
there). NiceGUI keeps one long-lived Python object per browser tab, so state
is just... state. A slider's on_change updates an attribute and calls
render.refresh(); nothing needs to be pre-seeded into a framework-owned
store first.
"""
from __future__ import annotations

import asyncio
import base64
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nicegui import app, ui

from caddie.physics import PGA_TOUR_AVERAGES
from caddie.shot import FLIGHT_HEIGHTS, ShotShape
from scripts.explorer import (
    CAVEATS,
    FT_M,
    GREEN_RADIUS_YD,
    GREEN_ZOOM_RADIUS_YD,
    HOLE_OUT_RADIUS_YD,
    PALETTES,
    SHELTER_PRESETS,
    WIND_CARDINALS,
    YARD_M,
    _round_scorecard,
    elevation_chart,
    fly,
    get_course,
    get_hole_info,
    get_hole_maps,
    map_chart,
    plan_chart,
    trajectory_table,
    waypoints,
)

CUSTOM_AIM = "Custom aim point"

# ---------------------------------------------------------------------------
# Theme. A calm, editorial "course guide" palette rather than a generic
# admin-dashboard blue -- fairway green as the one accent, everything else
# neutral so the charts (which carry their own two-series palette) stay the
# loudest thing on the page.
# ---------------------------------------------------------------------------
ACCENT = "#2f7a4f"


def fig_to_uri(fig) -> str:
    """Render a matplotlib Figure to a data: URI and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


class State:
    """Everything one browser tab needs to remember between interactions.

    One instance per page load (see main_page below) -- plain attributes,
    no framework-managed store. Per-hole values (wind, origin, shots) are
    keyed by hole number so switching holes naturally lands on that hole's
    own state instead of needing an explicit reset dance.
    """

    def __init__(self) -> None:
        self.course = get_course()
        self.hole_number = 1
        self.shots: dict[int, list[dict]] = {}
        self.origin_name: dict[int, str] = {}
        self.wind_mph: dict[int, float] = {}
        self.wind_cardinal: dict[int, str] = {}

        self.dark = True
        self.target_name: str | None = None
        self.use_custom_aim = False
        self.aim_dist = 0.0
        self.aim_offline = 0.0

        self.club_name = "Driver"  # every hole starts as a tee shot
        self._speed: dict[str, float] = {}
        self._angle: dict[str, float] = {}
        self._spin: dict[str, float] = {}

        self.hand = "RH"
        self.height = "stock"
        self.shape_mode = "shape"
        self.curve_yards = 0.0
        self.face_deg = 0.0
        self.path_deg = 0.0
        self.face_w = 0.78
        self.axis_per_ftp = 3.0
        self.manual_azimuth = 0.0
        self.manual_sidespin = 0.0

        self.wind_offset_deg = 0
        self.exposure = next(iter(SHELTER_PRESETS))
        self.veer = 0.0

        self.temp_c = 22.0
        self.rh = 0.55
        self.altimeter = 1013.25

        self.show_default = True
        self.show_map = True
        self.pending_origin: str | None = None

    # -- hole-scoped helpers ------------------------------------------------
    @property
    def hole(self):
        return self.course.hole(self.hole_number)

    def hole_shots(self) -> list[dict]:
        return self.shots.get(self.hole_number, [])

    def named_points(self) -> dict:
        pts: dict = {}
        for s in reversed(self.hole_shots()):
            pts[s["label"]] = s["landing"]
        pts.update(waypoints(self.hole))
        return pts

    def ensure_hole_state(self) -> None:
        """Roll wind and pick a sane origin the first time a hole is seen
        this session -- equivalent to the tee-off reset in explorer.py."""
        n = self.hole_number
        if n not in self.wind_mph:
            self.wind_mph[n] = round(random.uniform(2.0, 22.0) * 2) / 2
            self.wind_cardinal[n] = random.choice(list(WIND_CARDINALS))
        names = list(self.named_points().keys())
        if self.pending_origin in names:
            self.origin_name[n] = self.pending_origin
        elif self.origin_name.get(n) not in names:
            hs = self.hole_shots()
            self.origin_name[n] = hs[-1]["label"] if hs else "Tee"
        self.pending_origin = None
        self.use_custom_aim = False

    def club(self):
        return next(c for c in PGA_TOUR_AVERAGES if c.name == self.club_name)

    def speed(self) -> float:
        return self._speed.setdefault(self.club_name, float(self.club().ball_speed_mph))

    def angle(self) -> float:
        return self._angle.setdefault(self.club_name, float(self.club().launch_angle_deg))

    def spin(self) -> float:
        return self._spin.setdefault(self.club_name, float(self.club().backspin_rpm))

    def advance_to_next_hole(self) -> None:
        numbers = [h.number for h in self.course.holes]
        idx = numbers.index(self.hole_number)
        if idx + 1 < len(numbers):
            self.hole_number = numbers[idx + 1]
        self.club_name = "Driver"  # every hole starts as a tee shot
        self.ensure_hole_state()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@ui.page("/")
def main_page() -> None:
    ui.colors(primary=ACCENT, secondary="#8a6d3b", positive="#2f7a4f", negative="#c0392b")
    s = State()
    s.ensure_hole_state()

    ui.dark_mode(s.dark)
    ui.add_head_html(
        "<style>.q-field__label{opacity:.85} .nicegui-content{padding-top:0}</style>"
    )

    header = ui.header().classes(
        "items-center justify-between px-4 py-2 shadow-sm"
    ).style(f"background:{ACCENT}")
    with header:
        with ui.row().classes("items-center gap-3"):
            ui.icon("golf_course").classes("text-2xl")
            ui.label("AI Caddie").classes("text-lg font-semibold")
        hole_select = ui.select(
            {h.number: f"{h.number} — {h.name} (par {h.par})" for h in s.course.holes},
            value=s.hole_number,
        ).classes("w-64 bg-white/10 rounded").props("dark dense outlined")
        with ui.row().classes("items-center gap-2"):
            scorecard_btn = ui.button("Scorecard", icon="list_alt").props("flat dark")
            dark_toggle = ui.switch("Dark", value=s.dark).props("dark dense")

    scorecard_dialog = ui.dialog()
    with scorecard_dialog, ui.card().classes("p-0 min-w-[70vw]"):
        ui.label("Scorecard").classes("text-lg font-semibold p-4 pb-0")
        scorecard_container = ui.column().classes("p-4 pt-2 w-full overflow-x-auto")

    def refresh_scorecard() -> None:
        scorecard_container.clear()
        data = _round_scorecard(s.course)
        rows = []
        for i, label in enumerate(data[""]):
            row = {"metric": label}
            for col in list(data.keys())[1:]:
                row[col] = data[col][i]
            rows.append(row)
        columns = [{"name": "metric", "label": "", "field": "metric", "align": "left"}]
        for col in list(data.keys())[1:]:
            columns.append({"name": col, "label": col, "field": col, "align": "center"})
        with scorecard_container:
            ui.table(rows=rows, columns=columns, pagination=0).classes("w-full").props(
                "dense flat"
            )

    scorecard_btn.on("click", lambda: (refresh_scorecard(), scorecard_dialog.open()))

    with ui.left_drawer(value=True).classes("bg-neutral-50 dark:bg-neutral-900 p-3 gap-2") as drawer:
        controls = ui.column().classes("w-full gap-1")

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-3"):
        content = ui.column().classes("w-full gap-3")

    # -- reactive rebuild ----------------------------------------------------
    @ui.refreshable
    def render_controls() -> None:
        hole = s.hole
        names = list(s.named_points().keys())

        with ui.expansion("Wind", icon="air", value=True).classes("w-full").props(
            "header-class=font-semibold"
        ):
            ui.slider(min=0.0, max=40.0, step=0.5, value=s.wind_mph[s.hole_number]).props(
                "label-always"
            ).on_value_change(
                lambda e: (s.wind_mph.__setitem__(s.hole_number, e.value), render_content.refresh())
            )
            ui.label("Speed at 10 m (mph)").classes("text-xs text-neutral-500 -mt-2")
            ui.select(
                list(WIND_CARDINALS.keys()), value=s.wind_cardinal[s.hole_number],
                label="Direction (blowing FROM)",
            ).classes("w-full").on_value_change(
                lambda e: (s.wind_cardinal.__setitem__(s.hole_number, e.value), render_content.refresh())
            )
            offset = ui.slider(min=-45, max=45, step=5, value=s.wind_offset_deg).props("label-always")
            offset.on_value_change(lambda e: (setattr(s, "wind_offset_deg", e.value), render_content.refresh()))
            ui.select(
                list(SHELTER_PRESETS.keys()), value=s.exposure, label="Corridor exposure",
            ).classes("w-full").on_value_change(
                lambda e: (setattr(s, "exposure", e.value), render_content.refresh())
            )

        with ui.expansion("Shot", icon="sports_golf", value=True).classes("w-full").props(
            "header-class=font-semibold"
        ):
            # Off the tee, driver's fine. Anywhere else it's locked out -- a
            # blunt "not yet" rather than modeling who could pull it off.
            driver_locked = s.origin_name.get(s.hole_number, "Tee") != "Tee"
            club_choices = [
                c.name for c in PGA_TOUR_AVERAGES
                if not (driver_locked and c.name == "Driver")
            ]
            if s.club_name not in club_choices:
                s.club_name = "3-wood"
            ui.select(
                club_choices, value=s.club_name, label="Club",
            ).classes("w-full").on_value_change(
                lambda e: (
                    setattr(s, "club_name", e.value),
                    render_controls.refresh(),  # speed/angle/spin sliders show this club's numbers
                    render_content.refresh(),
                )
            )
            targets = [n for n in names if n != s.origin_name.get(s.hole_number)]
            if s.target_name not in targets:
                s.target_name = "Aim point 1" if "Aim point 1" in targets else (
                    targets[0] if targets else None
                )
            ui.select(
                targets, value=s.target_name, label="Aiming at",
            ).classes("w-full").on_value_change(
                lambda e: (setattr(s, "target_name", e.value), render_content.refresh())
            )
            ui.select(
                names, value=s.origin_name.get(s.hole_number), label="Playing from",
            ).classes("w-full").on_value_change(
                lambda e: (
                    s.origin_name.__setitem__(s.hole_number, e.value),
                    render_controls.refresh(),  # driver lock depends on origin == "Tee"
                    render_content.refresh(),
                )
            )
            custom = ui.checkbox("Custom aim point", value=s.use_custom_aim)
            custom.on_value_change(lambda e: (
                setattr(s, "use_custom_aim", e.value),
                render_controls.refresh(),  # shows/hides the aim-dist/offline sliders
                render_content.refresh(),
            ))
            if s.use_custom_aim:
                ui.slider(min=0.0, max=350.0, step=1.0, value=s.aim_dist).props("label-always").on_value_change(
                    lambda e: (setattr(s, "aim_dist", e.value), render_content.refresh())
                )
                ui.slider(min=-200.0, max=200.0, step=1.0, value=s.aim_offline).props("label-always").on_value_change(
                    lambda e: (setattr(s, "aim_offline", e.value), render_content.refresh())
                )
            ui.slider(min=60.0, max=190.0, step=0.5, value=s.speed()).props("label-always").on_value_change(
                lambda e: (s._speed.__setitem__(s.club_name, e.value), render_content.refresh())
            )
            ui.label("Ball speed (mph)").classes("text-xs text-neutral-500 -mt-2")
            ui.slider(min=2.0, max=50.0, step=0.1, value=s.angle()).props("label-always").on_value_change(
                lambda e: (s._angle.__setitem__(s.club_name, e.value), render_content.refresh())
            )
            ui.label("Launch angle (deg)").classes("text-xs text-neutral-500 -mt-2")
            ui.slider(min=1000.0, max=13000.0, step=50.0, value=s.spin()).props("label-always").on_value_change(
                lambda e: (s._spin.__setitem__(s.club_name, e.value), render_content.refresh())
            )
            ui.label("Backspin (rpm)").classes("text-xs text-neutral-500 -mt-2")

        with ui.expansion("Shape", icon="gesture", value=False).classes("w-full").props(
            "header-class=font-semibold"
        ):
            ui.radio(["RH", "LH"], value=s.hand).props("inline").on_value_change(
                lambda e: (setattr(s, "hand", e.value), render_content.refresh())
            )
            ui.select(
                list(FLIGHT_HEIGHTS), value=s.height, label="Trajectory",
            ).classes("w-full").on_value_change(
                lambda e: (setattr(s, "height", e.value), render_content.refresh())
            )
            mode_labels = {"shape": "Curve + auto-aim", "swing": "Face & path", "manual": "Raw numbers"}
            ui.radio(
                mode_labels, value=s.shape_mode,
            ).on_value_change(lambda e: (
                setattr(s, "shape_mode", e.value),
                render_controls.refresh(),  # swaps in that mode's own sliders
                render_content.refresh(),
            ))
            if s.shape_mode == "shape":
                ui.slider(min=-40.0, max=40.0, step=1.0, value=s.curve_yards).props("label-always").on_value_change(
                    lambda e: (setattr(s, "curve_yards", e.value), render_content.refresh())
                )
                ui.label("Curve (yd, + = right). 0 = auto-hold straight against wind.").classes(
                    "text-xs text-neutral-500 -mt-2"
                )
            elif s.shape_mode == "swing":
                ui.slider(min=-8.0, max=8.0, step=0.25, value=s.face_deg).props("label-always").on_value_change(
                    lambda e: (setattr(s, "face_deg", e.value), render_content.refresh())
                )
                ui.label("Club face (deg)").classes("text-xs text-neutral-500 -mt-2")
                ui.slider(min=-12.0, max=12.0, step=0.25, value=s.path_deg).props("label-always").on_value_change(
                    lambda e: (setattr(s, "path_deg", e.value), render_content.refresh())
                )
                ui.label("Club path (deg)").classes("text-xs text-neutral-500 -mt-2")
            else:
                ui.slider(min=-15.0, max=15.0, step=0.5, value=s.manual_azimuth).props("label-always").on_value_change(
                    lambda e: (setattr(s, "manual_azimuth", e.value), render_content.refresh())
                )
                ui.label("Start direction (deg)").classes("text-xs text-neutral-500 -mt-2")
                ui.slider(min=-1500.0, max=1500.0, step=25.0, value=s.manual_sidespin).props("label-always").on_value_change(
                    lambda e: (setattr(s, "manual_sidespin", e.value), render_content.refresh())
                )
                ui.label("Sidespin (rpm)").classes("text-xs text-neutral-500 -mt-2")

        with ui.expansion("Air & display", icon="thermostat", value=False).classes("w-full").props(
            "header-class=font-semibold"
        ):
            ui.slider(min=-5.0, max=40.0, step=0.5, value=s.temp_c).props("label-always").on_value_change(
                lambda e: (setattr(s, "temp_c", e.value), render_content.refresh())
            )
            ui.label("Temperature (C)").classes("text-xs text-neutral-500 -mt-2")
            ui.slider(min=0.0, max=1.0, step=0.05, value=s.rh).props("label-always").on_value_change(
                lambda e: (setattr(s, "rh", e.value), render_content.refresh())
            )
            ui.label("Relative humidity").classes("text-xs text-neutral-500 -mt-2")
            ui.slider(min=970.0, max=1050.0, step=0.25, value=s.altimeter).props("label-always").on_value_change(
                lambda e: (setattr(s, "altimeter", e.value), render_content.refresh())
            )
            ui.label("Sea-level pressure (hPa)").classes("text-xs text-neutral-500 -mt-2")
            ui.checkbox("Overlay unshaped Default Trajectory", value=s.show_default).on_value_change(
                lambda e: (setattr(s, "show_default", e.value), render_content.refresh())
            )
            ui.checkbox("Draw on hole map", value=s.show_map).on_value_change(
                lambda e: (setattr(s, "show_map", e.value), render_content.refresh())
            )

    @ui.refreshable
    def render_content() -> None:
        hole = s.hole
        names = s.named_points()
        origin_name = s.origin_name.get(s.hole_number, "Tee")
        origin_geo = names[origin_name]

        if s.use_custom_aim:
            ref_frame = hole.frame_from(origin_geo, hole.pin)
            target_geo = ref_frame.to_geo(s.aim_dist * YARD_M, s.aim_offline * YARD_M, 0.0)
        else:
            target_name = s.target_name or "Pin"
            target_geo = names.get(target_name, hole.pin)

        wind_dir = (WIND_CARDINALS[s.wind_cardinal[s.hole_number]] + s.wind_offset_deg) % 360
        preset = SHELTER_PRESETS[s.exposure]

        def to_point(geo):
            return (geo.lat, geo.lon, geo.elevation_m)

        data = fly(
            hole.number,
            to_point(origin_geo),
            to_point(target_geo),
            (s.speed(), s.angle(), s.spin(), s.manual_azimuth, s.manual_sidespin),
            (s.wind_mph[s.hole_number], wind_dir, preset["roughness_length"],
             preset["shelter_factor"], preset["treeline_height"], s.veer),
            (s.temp_c, s.rh, s.altimeter),
            (s.shape_mode, s.curve_yards, s.height, s.hand, s.face_deg, s.path_deg,
             s.face_w, s.axis_per_ftp),
        )

        p = PALETTES["dark" if s.dark else "light"]
        hole_shots = s.hole_shots()
        holed_out = bool(hole_shots) and hole_shots[-1].get("holed", False)
        about_to_hole = data["distance_to_pin_yd"] <= HOLE_OUT_RADIUS_YD

        content.clear()
        with content:
            inf = get_hole_info().get(hole.number, {})
            difficulty = ""
            if inf.get("historicalRank"):
                difficulty = (
                    f" · historically {inf['historicalAverage']:.3f} strokes "
                    f"({inf['historicalAverage'] - hole.par:+.3f} to par), "
                    f"{inf['historicalRank']}th hardest"
                )
            with ui.card().classes("w-full"):
                ui.label(f"Hole {hole.number} — {hole.name} · par {hole.par}").classes(
                    "text-xl font-semibold"
                )
                ui.label(
                    f"{origin_name} → {s.target_name or 'custom'} · "
                    f"card {hole.card_yardage} yd · dogleg {hole.dogleg_deg:.0f}°"
                    f"{difficulty}"
                ).classes("text-sm text-neutral-500")

            with ui.row().classes("w-full gap-3 items-stretch"):
                with ui.card().classes("flex-1 items-center justify-center"):
                    ui.label("Remaining to pin").classes("text-xs text-neutral-500")
                    # Always the real distance from where you're standing --
                    # never the previewed shot's landing spot, which reads as
                    # flatly wrong off the tee (e.g. a 323 yd hole showing
                    # ~120 yd because that's just where the currently
                    # dialled-in club happens to land).
                    ui.label(f"{data['origin_distance_to_pin_yd']:.0f} yd").classes(
                        "text-2xl font-bold"
                    )

                def do_save() -> None:
                    n = len(hole_shots) + 1
                    shot = data["shot"]
                    if about_to_hole:
                        label = f"Shot {n}: {s.club_name} → holed! ({shot['carry']:.0f} yd)"
                    else:
                        side = ("right" if shot["offline"] > 0.5 else
                                "left" if shot["offline"] < -0.5 else "straight")
                        label = (
                            f"Shot {n}: {s.club_name} → {shot['carry']:.0f} yd "
                            f"({abs(shot['offline']):.0f} yd {side}), "
                            f"{data['distance_to_pin_yd']:.0f} yd left"
                        )
                    s.shots.setdefault(hole.number, []).append(dict(
                        label=label, landing=data["landing_geo"], club=s.club_name,
                        carry=shot["carry"], offline=shot["offline"],
                        distance_to_pin_yd=data["distance_to_pin_yd"],
                        holed=about_to_hole, strokes=1,
                    ))
                    if about_to_hole:
                        s.advance_to_next_hole()
                        hole_select.set_value(s.hole_number)
                    else:
                        s.pending_origin = label
                        s.ensure_hole_state()
                    render_controls.refresh()
                    render_content.refresh()
                    ui.notify(label, type="positive")

                def do_reset() -> None:
                    s.shots[hole.number] = []
                    s.club_name = "Driver"  # restarting the hole is a fresh tee shot
                    s.ensure_hole_state()
                    render_controls.refresh()
                    render_content.refresh()

                with ui.card().classes("flex-[2] justify-center gap-2"):
                    with ui.row().classes("gap-2 items-center flex-wrap"):
                        play_btn = ui.button("Watch it fly", icon="play_arrow").props("outline")
                        ui.button(
                            "Holes out — save!" if about_to_hole else "Save shot (no animation)",
                            icon="flag" if about_to_hole else "save",
                            on_click=do_save,
                        ).props("color=primary")
                        if hole_shots:
                            ui.button("Reset hole", icon="restart_alt", on_click=do_reset).props(
                                "flat color=negative"
                            )

            if holed_out:
                total = sum(sh.get("strokes", 1) for sh in hole_shots)
                ui.label(f"⛳ Holed out in {total} ({total - hole.par:+d} to par)").classes(
                    "text-positive font-semibold"
                )

            hole_map = get_hole_maps().get(hole.number)
            has_map = hole_map is not None and hole_map.exists() and s.show_map

            with ui.tabs().classes("w-full") as tabs:
                t_map = ui.tab("Map", icon="map")
                t_elev = ui.tab("Elevation", icon="landscape")
                t_plan = ui.tab("Plan view", icon="explore")
                t_table = ui.tab("Table", icon="table_chart")
            map_image = elev_image = None
            with ui.tab_panels(tabs, value=t_map if s.show_map else t_plan).classes("w-full"):
                with ui.tab_panel(t_map):
                    if has_map:
                        on_green = data["distance_to_pin_yd"] <= GREEN_RADIUS_YD
                        fig = map_chart(
                            data, p, hole, hole_map, s.show_default,
                            zoom_radius_yd=GREEN_ZOOM_RADIUS_YD if on_green else None,
                        )
                        map_image = ui.image(fig_to_uri(fig)).classes("w-full rounded")
                    else:
                        ui.label("No map illustration for this hole.").classes(
                            "text-neutral-500 p-4"
                        )
                with ui.tab_panel(t_elev):
                    fig = elevation_chart(data, p, s.show_default)
                    elev_image = ui.image(fig_to_uri(fig)).classes("w-full rounded")
                with ui.tab_panel(t_plan):
                    fig = plan_chart(data, p, s.show_default)
                    ui.image(fig_to_uri(fig)).classes("w-full rounded")
                with ui.tab_panel(t_table):
                    ui.table(
                        rows=[
                            dict(zip(trajectory_table(data).keys(), vals))
                            for vals in zip(*trajectory_table(data).values())
                        ],
                        columns=[
                            {"name": k, "label": k, "field": k}
                            for k in trajectory_table(data).keys()
                        ],
                        pagination=10,
                    ).classes("w-full").props("dense flat")

            async def play_animation() -> None:
                # Animate whichever view is actually on screen -- the map
                # overlay if this hole has one, the elevation profile
                # otherwise. No blocking sleep: NiceGUI keeps the rest of the
                # UI responsive between frames since this is a real asyncio
                # coroutine, not a Streamlit script rerun.
                anim_image = map_image if has_map else elev_image
                shot = data["shot"]
                n_samples = len(shot["x"])
                n_frames = 40
                idxs = sorted(set(
                    int(round(k * (n_samples - 1) / max(n_frames - 1, 1)))
                    for k in range(n_frames)
                ))
                play_btn.props("loading")
                try:
                    for idx in idxs:
                        if has_map:
                            fig = map_chart(data, p, hole, hole_map, s.show_default, until_idx=idx)
                        else:
                            fig = elevation_chart(data, p, s.show_default, until_idx=idx)
                        anim_image.set_source(fig_to_uri(fig))
                        await asyncio.sleep(0.02)
                    if has_map:
                        fig = map_chart(
                            data, p, hole, hole_map, s.show_default,
                            zoom_radius_yd=GREEN_ZOOM_RADIUS_YD
                            if data["distance_to_pin_yd"] <= GREEN_RADIUS_YD else None,
                        )
                    else:
                        fig = elevation_chart(data, p, s.show_default)
                    anim_image.set_source(fig_to_uri(fig))
                finally:
                    play_btn.props(remove="loading")

            if map_image is not None or elev_image is not None:
                play_btn.on("click", play_animation)
            else:
                play_btn.disable()

            if hole_shots:
                with ui.expansion(f"Shots played ({sum(sh.get('strokes', 1) for sh in hole_shots)})"):
                    ui.table(
                        rows=[
                            {
                                "#": i + 1, "club": sh["club"],
                                "strokes": sh.get("strokes", 1),
                                "carry (yd)": round(sh["carry"], 1),
                                "offline (yd)": round(sh["offline"], 1),
                                "left to pin (yd)": round(sh["distance_to_pin_yd"], 1),
                            }
                            for i, sh in enumerate(hole_shots)
                        ],
                        columns=[
                            {"name": c, "label": c, "field": c}
                            for c in ["#", "club", "strokes", "carry (yd)", "offline (yd)", "left to pin (yd)"]
                        ],
                        pagination=0,
                    ).classes("w-full").props("dense flat")

            on_green_committed = (
                bool(hole_shots) and not holed_out
                and hole_shots[-1]["distance_to_pin_yd"] <= GREEN_RADIUS_YD
            )
            if on_green_committed:
                last_dist_yd = hole_shots[-1]["distance_to_pin_yd"]
                default_putts = 1 if last_dist_yd <= 5 else (
                    random.choice([1, 2]) if last_dist_yd <= 10 else 2
                )
                with ui.card().classes("w-full"):
                    ui.label("On the green — putting isn't modeled, declare a result.").classes(
                        "text-sm text-neutral-500"
                    )
                    with ui.row().classes("items-center gap-3"):
                        putts_select = ui.select([1, 2, 3, 4], value=default_putts, label="Putts").classes("w-32")

                        def do_hole_out() -> None:
                            n = len(hole_shots) + 1
                            putts = putts_select.value
                            s.shots[hole.number].append(dict(
                                label=f"Shot {n}: {putts} putt" + ("s" if putts > 1 else "") + " — holed",
                                landing=hole.pin, club=f"{putts}-putt" + ("s" if putts > 1 else ""),
                                carry=hole_shots[-1]["distance_to_pin_yd"], offline=0.0,
                                distance_to_pin_yd=0.0, holed=True, strokes=putts,
                            ))
                            s.advance_to_next_hole()
                            hole_select.set_value(s.hole_number)
                            render_controls.refresh()
                            render_content.refresh()

                        ui.button("Hole out", icon="flag", on_click=do_hole_out).props("color=primary")

            with ui.expansion("What this does not know", icon="info").classes("w-full"):
                ui.markdown(CAVEATS)

    def on_hole_change(e) -> None:
        s.hole_number = e.value
        s.club_name = "Driver"  # every hole starts as a tee shot
        s.ensure_hole_state()
        render_controls.refresh()
        render_content.refresh()

    hole_select.on_value_change(on_hole_change)
    dark_toggle.on_value_change(lambda e: (setattr(s, "dark", e.value), ui.dark_mode(e.value)))

    with controls:
        render_controls()
    with content:
        render_content()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="AI Caddie", port=8502, reload=False, favicon="⛳")
