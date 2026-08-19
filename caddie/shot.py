"""Shot shaping: turning an *intent* into launch conditions.

``LaunchConditions`` is the launch-monitor parameter set -- six numbers
describing what the ball is doing off the face. This module is the layer above
it, the one ``ball.py`` scopes out as "the club model's job": mapping "a 5-yard
draw that finishes on the flag" onto those six numbers.

Three layers, and they are NOT equally trustworthy. Read this before using
anything below, because the whole point of this file is that the boundary
between physics and folklore should be visible in the code rather than buried.

Layer 1 -- curvature and aim.  VALIDATED PHYSICS, no new assumptions.
    Curving a golf ball is a tilted spin axis, and the integrator already does
    that exactly. Because ``AeroModel.spin_ratio`` is computed from the spin
    *magnitude*, tilting the axis leaves C_D and C_L untouched and only rotates
    the Magnus force: the vertical (lift) component scales as cos(tilt) and the
    lateral (curve) component as sin(tilt). So the carry cost of shaping is not
    a fudge factor -- it falls out of the same equations that were calibrated
    against the tour table. Nothing here is fitted or invented; the solvers
    just invert the existing model numerically.

    (Cross-check: MacDonald & Hanzely, "The physics of the drive in golf",
    Am. J. Phys. 59(3) 1991, write the same decomposition with the spin-axis
    angle entering the vertical Magnus term as cos and the lateral as sin.
    Independent implementation, same structure.)

    ONE CAVEAT INHERITED FROM THE AERO FIT. The lateral force is C_L * sin(tilt),
    so predicted curve is only as good as C_L -- and per
    ``aero.EMPIRICAL_CL_TABLE`` our C_L runs high through the middle of the bag:
    roughly +7% at driver spin ratios, +3% at wedge, but up to +48% around
    S = 0.14-0.20 (the woods and long irons). Expect this module to OVER-predict
    how much a 5-wood or 3-iron bends, by something of that order, while driver
    and wedge curvature are close. The solvers invert the model exactly; the
    model itself is the limit. Fixing the lift fit fixes the curve too, and
    nothing in this file needs to change when it does.

Layer 2 -- trajectory height.  HAND-SET, NOT MEASURED.
    A knockdown or a flighted shot changes ball speed, launch angle and spin
    together. The *signs* of those changes are well established; the
    *magnitudes* in ``FLIGHT_HEIGHTS`` are my estimates, not published data,
    and they are not in any tour-average table. Treat them exactly as you treat
    the per-hole wind shelter factors: directionally right, numerically
    unaudited. Replace them from your own launch monitor.

Layer 3 -- face and path (the D-plane).  UNVALIDATED MODEL.
    Mapping club face angle and club path onto a start line and a spin axis
    needs a spin-generation model, which needs impact data this project does
    not have. The relationships in ``DPlaneModel`` have the right form and are
    standard teaching, but the two constants are hand-set. If you use one thing
    from this layer, use :func:`calibrate_face_to_path` -- it back-solves the
    constant from one observed shot through the *validated* Layer 1 solver,
    which turns the guess into a measurement.

Sign conventions, consistent with ``LaunchConditions`` throughout:
    +y, +azimuth, +spin_axis, +curve  ==  the player's RIGHT.
Handedness never touches the physics. It only decides whether a rightward
curve is called a fade (right-handed) or a draw (left-handed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from scipy.optimize import brentq

from .physics.ball import ClubReference, LaunchConditions
from .physics.constants import YARD_M
from .physics.trajectory import FlightIntegrator, TerrainFn, Trajectory, flat_terrain
from .physics.wind import WindField

# Solver brackets. Past these a "shape" is a miss, not a shot.
MAX_SPIN_AXIS_DEG = 60.0
MAX_AIM_DEG = 25.0


# ---------------------------------------------------------------------------
# Layer 2: trajectory height. Hand-set -- see the module docstring.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlightHeight:
    """A multiplicative/additive tweak to launch conditions for flighting.

    Applied as: speed *= speed_scale, launch += launch_delta_deg,
    spin *= spin_scale.
    """

    name: str
    speed_scale: float = 1.0
    launch_delta_deg: float = 0.0
    spin_scale: float = 1.0
    note: str = ""

    def apply(self, launch: LaunchConditions) -> LaunchConditions:
        return replace(
            launch,
            ball_speed_mph=launch.ball_speed_mph * self.speed_scale,
            launch_angle_deg=launch.launch_angle_deg + self.launch_delta_deg,
            backspin_rpm=launch.backspin_rpm * self.spin_scale,
        )


FLIGHT_HEIGHTS: dict[str, FlightHeight] = {
    "knockdown": FlightHeight(
        "knockdown", 0.94, -3.0, 0.88,
        "Ball back, shorter finish. Lower and shorter, holds its line in wind.",
    ),
    "three-quarter": FlightHeight(
        "three-quarter", 0.97, -1.5, 0.94, "Controlled, slightly lower.",
    ),
    "stock": FlightHeight("stock", 1.0, 0.0, 1.0, "The club's normal flight."),
    "flighted": FlightHeight(
        "flighted", 1.0, 2.0, 1.06,
        "Higher and softer-landing, for carrying trouble or holding a firm green.",
    ),
}
"""Flighting presets. MAGNITUDES ARE HAND-SET, NOT PUBLISHED DATA.

The signs are not controversial -- a knockdown really is slower, lower-launching
and lower-spinning than a stock shot, and a flighted shot really is higher and
higher-spinning. The specific numbers are estimates. They are the only values in
this file not either derived from the calibrated physics or measurable by
:func:`calibrate_face_to_path`, and they are flagged here rather than quietly
embedded so that a wrong number is findable. Replace from a launch monitor
session and nothing else in this module needs to change."""


# ---------------------------------------------------------------------------
# Layer 1: the shot intent, and the solvers that realise it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShotShape:
    """What the player is trying to do.

    Attributes
    ----------
    curve_yards:
        Signed lateral bend away from the START LINE, in yards. Positive is to
        the player's right. This is the *observed* bend, so it includes what the
        wind does: asking for 0 yards of curve in a crosswind is asking to hold
        the ball against the wind, which is a real shot and the solver will find
        the axis tilt that does it.
    height:
        A key of :data:`FLIGHT_HEIGHTS`. See the Layer 2 warning.
    hand:
        "RH" or "LH". Naming only -- never enters the physics.
    """

    curve_yards: float = 0.0
    height: str = "stock"
    hand: str = "RH"

    def __post_init__(self) -> None:
        if self.height not in FLIGHT_HEIGHTS:
            raise ValueError(
                f"unknown height {self.height!r}; "
                f"expected one of {sorted(FLIGHT_HEIGHTS)}"
            )
        if self.hand not in ("RH", "LH"):
            raise ValueError("hand must be 'RH' or 'LH'")

    @property
    def flight_height(self) -> FlightHeight:
        return FLIGHT_HEIGHTS[self.height]

    @property
    def shape_name(self) -> str:
        """"draw", "fade" or "straight" -- the only handedness-aware thing here."""
        if abs(self.curve_yards) < 1.0:
            return "straight"
        rightward = self.curve_yards > 0
        if self.hand == "RH":
            return "fade" if rightward else "draw"
        return "draw" if rightward else "fade"

    def describe(self) -> str:
        if self.shape_name == "straight":
            shape = "straight"
        else:
            shape = f"{abs(self.curve_yards):.0f} yd {self.shape_name}"
        return shape if self.height == "stock" else f"{self.height} {shape}"


def with_spin_axis(launch: LaunchConditions, spin_axis_deg: float) -> LaunchConditions:
    """Set the spin-axis tilt, preserving total spin.

    Uses ``spin_axis_deg`` rather than ``sidespin_rpm`` deliberately: a real
    ball has ONE spin axis, and tilting it converts backspin into sidespin
    instead of adding spin from nowhere. Adding sidespin on top of unchanged
    backspin would raise the total spin, which would raise C_L, which would
    make a shaped shot fly *further* -- backwards from reality.
    """
    return replace(launch, spin_axis_deg=spin_axis_deg, sidespin_rpm=0.0)


def bend_yards(traj: Trajectory, azimuth_deg: float) -> float:
    """Lateral bend away from the start line at landing, yards, right-positive.

    Not the same as ``traj.offline_yards``, which is measured from the target
    line. A pushed straight shot has offline > 0 but zero bend; that distinction
    is the whole basis of separating "how much does it curve" from "where do I
    aim it".
    """
    dx = traj.positions[-1, 0] - traj.positions[0, 0]
    dy = traj.positions[-1, 1] - traj.positions[0, 1]
    return (dy - dx * math.tan(math.radians(azimuth_deg))) / YARD_M


@dataclass
class ShotPlan:
    """A realised shot: the launch conditions, the flight, and what it cost."""

    launch: LaunchConditions
    trajectory: Trajectory
    shape: ShotShape
    straight_reference: Trajectory
    converged: bool = True

    @property
    def spin_axis_deg(self) -> float:
        return self.launch.spin_axis_deg or 0.0

    @property
    def aim_deg(self) -> float:
        return self.launch.azimuth_deg

    @property
    def curve_achieved_yards(self) -> float:
        return bend_yards(self.trajectory, self.launch.azimuth_deg)

    @property
    def finish_offline_yards(self) -> float:
        """Where it ends up relative to the target line. The number that counts."""
        return self.trajectory.offline_yards

    @property
    def shaping_cost_yards(self) -> float:
        """Carry given up versus the same club hit straight, same conditions.

        Negative means the shape cost distance. This is a physical consequence
        of tilting lift out of the vertical, not an adjustment applied on top.
        """
        return self.trajectory.carry_yards - self.straight_reference.carry_yards

    def describe(self) -> str:
        return (
            f"{self.shape.describe()}: aim {self.aim_deg:+.1f}deg, "
            f"spin axis {self.spin_axis_deg:+.1f}deg -> "
            f"carry {self.trajectory.carry_yards:.1f} yd "
            f"({self.shaping_cost_yards:+.1f} vs straight), "
            f"curve {self.curve_achieved_yards:+.1f} yd, "
            f"finishes {self.finish_offline_yards:+.1f} yd offline"
        )


def _integrator_at(integ: FlightIntegrator, dt: float) -> FlightIntegrator:
    """Same physics, different step. Used to solve coarse and report fine."""
    if dt == integ.dt:
        return integ
    return FlightIntegrator(
        atmosphere=integ.atmosphere, aero=integ.aero, ball=integ.ball,
        dt=dt, max_time=integ.max_time,
    )


def solve_spin_axis(
    integ: FlightIntegrator,
    launch: LaunchConditions,
    curve_yards: float,
    wind: WindField | None = None,
    terrain: TerrainFn = flat_terrain,
    max_axis_deg: float = MAX_SPIN_AXIS_DEG,
) -> float:
    """Find the spin-axis tilt that bends the ball ``curve_yards`` off its start line.

    A 1-D root find. Bend is monotonic in tilt over the practical range -- more
    tilt puts more of a fixed-magnitude Magnus force sideways -- so bisection is
    safe and needs no derivative.
    """

    def bend_error(axis_deg: float) -> float:
        traj = integ.integrate(
            with_spin_axis(launch, axis_deg), wind=wind, terrain=terrain
        )
        return bend_yards(traj, launch.azimuth_deg) - curve_yards

    lo, hi = -max_axis_deg, max_axis_deg
    e_lo, e_hi = bend_error(lo), bend_error(hi)
    if e_lo > 0 or e_hi < 0:
        raise ValueError(
            f"cannot bend this shot {curve_yards:+.1f} yd: within +/-"
            f"{max_axis_deg:.0f}deg of axis tilt the reachable bend is "
            f"{e_lo + curve_yards:+.1f} to {e_hi + curve_yards:+.1f} yd. "
            "Slower clubs curve less -- ask for less shape, or more club."
        )
    return brentq(bend_error, lo, hi, xtol=1e-3)


def solve_aim(
    integ: FlightIntegrator,
    launch: LaunchConditions,
    wind: WindField | None = None,
    terrain: TerrainFn = flat_terrain,
    finish_offline_yards: float = 0.0,
    max_aim_deg: float = MAX_AIM_DEG,
) -> float:
    """Find the start direction that lands the ball on the target line.

    Same structure as :func:`solve_spin_axis`: landing offline is monotonic in
    the launch azimuth, so bisect.
    """

    def offline_error(azimuth_deg: float) -> float:
        traj = integ.integrate(
            replace(launch, azimuth_deg=azimuth_deg), wind=wind, terrain=terrain
        )
        return traj.offline_yards - finish_offline_yards

    lo, hi = -max_aim_deg, max_aim_deg
    e_lo, e_hi = offline_error(lo), offline_error(hi)
    if e_lo > 0 or e_hi < 0:
        raise ValueError(
            f"cannot finish {finish_offline_yards:+.1f} yd offline within +/-"
            f"{max_aim_deg:.0f}deg of aim; reachable range is "
            f"{e_lo + finish_offline_yards:+.1f} to "
            f"{e_hi + finish_offline_yards:+.1f} yd."
        )
    return brentq(offline_error, lo, hi, xtol=1e-3)


def plan_shot(
    integ: FlightIntegrator,
    club: ClubReference | LaunchConditions,
    shape: ShotShape,
    wind: WindField | None = None,
    terrain: TerrainFn = flat_terrain,
    finish_offline_yards: float = 0.0,
    solve_dt: float = 0.016,
    iterations: int = 4,
    tol_yards: float = 0.15,
) -> ShotPlan:
    """Solve for the shot that curves as asked AND finishes on the target line.

    Curve and aim are coupled: tilting the axis moves the landing point, and
    re-aiming changes where the bend ends up. Rather than a 2-D solve we
    alternate the two 1-D solves, which converges in a handful of passes
    because the coupling is weak (aim rotates the shot; it barely changes how
    much the ball bends).

    Solved at ``solve_dt`` and reported at the integrator's own ``dt`` -- the
    same coarse-solve/fine-report split ``calibrate.py`` uses. The root finds
    need tens of integrations; the answer needs one.
    """
    base = club.launch() if isinstance(club, ClubReference) else club
    launch = shape.flight_height.apply(base)

    solver = _integrator_at(integ, solve_dt)

    axis = 0.0
    aim = launch.azimuth_deg
    converged = False
    for _ in range(iterations):
        axis = solve_spin_axis(
            solver, replace(launch, azimuth_deg=aim), shape.curve_yards,
            wind=wind, terrain=terrain,
        )
        shaped = with_spin_axis(launch, axis)
        aim = solve_aim(
            solver, shaped, wind=wind, terrain=terrain,
            finish_offline_yards=finish_offline_yards,
        )
        check = solver.integrate(
            replace(shaped, azimuth_deg=aim), wind=wind, terrain=terrain
        )
        if (
            abs(bend_yards(check, aim) - shape.curve_yards) < tol_yards
            and abs(check.offline_yards - finish_offline_yards) < tol_yards
        ):
            converged = True
            break

    final = replace(with_spin_axis(launch, axis), azimuth_deg=aim)
    trajectory = integ.integrate(final, wind=wind, terrain=terrain)

    # Reference: same club, same flighting, no shape, aimed straight. Isolates
    # the cost of the shape from the cost of the flighting.
    straight = integ.integrate(
        replace(launch, azimuth_deg=0.0, spin_axis_deg=None, sidespin_rpm=0.0),
        wind=wind, terrain=terrain,
    )
    return ShotPlan(final, trajectory, shape, straight, converged)


# ---------------------------------------------------------------------------
# Layer 3: the D-plane. Unvalidated -- see the module docstring.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DPlaneModel:
    """Club face + club path -> start line and spin axis.

    Form
    ----
    Both relationships are standard ball-flight law:

        start line  ~  face_weight * face + (1 - face_weight) * path
        spin axis   ~  axis_per_face_to_path * (face - path)

    The ball starts close to where the face points, and it curves away from the
    path -- so a face closed relative to the path (negative face-to-path) tilts
    the axis left and draws the ball.

    Provenance
    ----------
    The FORM above is well established. Both CONSTANTS are hand-set:

    ``face_weight``
        Widely taught as roughly 0.85 for a driver and 0.75 for an iron. The
        physical reason is real (a longer club is struck with a shallower,
        faster face, so the face dominates the start line more), but the
        numbers are rules of thumb.
    ``axis_per_face_to_path``
        Degrees of spin-axis tilt per degree of face-to-path. This one has no
        good published value at all, because it depends on the ball, the loft
        and gear effect. The default is a placeholder.

    ``calibrate_face_to_path`` measures the second constant from a single
    observed shot, which is the intended way to use this class.
    """

    face_weight: float = 0.78
    axis_per_face_to_path: float = 3.0

    DRIVER_FACE_WEIGHT: float = 0.85
    IRON_FACE_WEIGHT: float = 0.75

    def start_line_deg(self, face_deg: float, path_deg: float) -> float:
        return self.face_weight * face_deg + (1.0 - self.face_weight) * path_deg

    def spin_axis_deg(self, face_deg: float, path_deg: float) -> float:
        return self.axis_per_face_to_path * (face_deg - path_deg)

    def launch_from_swing(
        self, base: LaunchConditions, face_deg: float, path_deg: float
    ) -> LaunchConditions:
        """Apply a face/path pair to a club's stock launch conditions."""
        return replace(
            base,
            azimuth_deg=self.start_line_deg(face_deg, path_deg),
            spin_axis_deg=self.spin_axis_deg(face_deg, path_deg),
            sidespin_rpm=0.0,
        )

    def face_path_for(
        self, spin_axis_deg: float, start_line_deg: float
    ) -> tuple[float, float]:
        """Invert: the face and path that would produce this axis and start line.

        Solving the two linear relationships simultaneously. Useful for turning
        a Layer 1 solution -- which is physics -- into something a player can
        actually rehearse on the range.
        """
        face_to_path = spin_axis_deg / self.axis_per_face_to_path
        # start = w*face + (1-w)*path  and  face - path = face_to_path
        #   => start = w*face + (1-w)*(face - face_to_path)
        #   => face  = start + (1-w)*face_to_path
        face = start_line_deg + (1.0 - self.face_weight) * face_to_path
        return face, face - face_to_path


def calibrate_face_to_path(
    integ: FlightIntegrator,
    club: ClubReference | LaunchConditions,
    face_to_path_deg: float,
    observed_curve_yards: float,
    wind: WindField | None = None,
    terrain: TerrainFn = flat_terrain,
    solve_dt: float = 0.004,
) -> float:
    """Measure ``axis_per_face_to_path`` from one observed shot.

    Give it a shot you actually hit -- "3 degrees of face-to-path curved it 12
    yards to the right" -- and it inverts the *validated* Layer 1 physics to
    find the axis tilt that produces that curve, then divides. The result is the
    one number Layer 3 cannot otherwise justify, turned into a measurement.

    Any launch monitor that reports club path and face angle gives you both
    inputs, so this is a two-minute job on a range session.
    """
    if abs(face_to_path_deg) < 1e-6:
        raise ValueError("face_to_path_deg must be non-zero to divide by it")
    base = club.launch() if isinstance(club, ClubReference) else club
    axis = solve_spin_axis(
        _integrator_at(integ, solve_dt), base, observed_curve_yards,
        wind=wind, terrain=terrain,
    )
    return axis / face_to_path_deg
