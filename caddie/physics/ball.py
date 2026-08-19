"""Ball properties and launch conditions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (
    BALL_INERTIA_UNIFORM,
    BALL_MASS_MAX_KG,
    BALL_RADIUS_M,
    MPH_MS,
    RPM_RADS,
)


@dataclass(frozen=True)
class Ball:
    """Physical ball. Defaults are a conforming tour ball at the rules limits."""

    mass: float = BALL_MASS_MAX_KG
    radius: float = BALL_RADIUS_M
    inertia: float = BALL_INERTIA_UNIFORM

    # Spin decay time constant, seconds. Spin bleeds off roughly
    # exponentially in flight; launch-monitor studies put the rate at
    # 3-7 %/s, i.e. tau ~ 14-33 s. Fitted by scripts/calibrate.py, which
    # bounds it to that band -- left free the optimiser pushes tau to ~60 s
    # and buys carry accuracy with an unphysical spin history.
    spin_decay_tau: float = 32.99

    @property
    def area(self) -> float:
        """Cross-sectional (frontal) area, m^2."""
        return math.pi * self.radius**2


@dataclass(frozen=True)
class LaunchConditions:
    """What the ball is doing the instant it leaves the face.

    This is deliberately the launch-monitor parameter set, not a swing model:
    the caddie's job is to pick a shot, and a shot is specified by these six
    numbers. Mapping "7-iron, smooth" onto them is the club model's job.

    Angles are in degrees for ergonomics; everything is converted on access.

    Attributes
    ----------
    ball_speed_mph:
        Initial ball speed.
    launch_angle_deg:
        Vertical launch angle above horizontal.
    azimuth_deg:
        Horizontal launch direction relative to the target line. Positive is
        right of target (a push), negative is left (a pull).
    backspin_rpm:
        Spin about the horizontal axis perpendicular to flight. Positive is
        normal backspin.
    sidespin_rpm:
        Spin about the vertical axis. Positive curves the ball to the RIGHT
        for a right-handed player (a slice/fade); negative curves it left.
        Note: real balls have a single tilted spin axis, which is exactly
        what backspin + sidespin compose into -- see :meth:`spin_vector`.
    spin_axis_deg:
        Optional alternative to ``sidespin_rpm``: the spin-axis tilt as
        reported by Trackman/GCQuad. If given, it overrides ``sidespin_rpm``
        and the total spin is taken from ``backspin_rpm``.
    """

    ball_speed_mph: float
    launch_angle_deg: float
    backspin_rpm: float
    azimuth_deg: float = 0.0
    sidespin_rpm: float = 0.0
    spin_axis_deg: float | None = None

    @property
    def speed(self) -> float:
        return self.ball_speed_mph * MPH_MS

    def velocity_vector(self) -> tuple[float, float, float]:
        """Initial velocity in the shot frame.

        Frame convention used throughout: ``x`` down the target line,
        ``y`` to the player's RIGHT, ``z`` up.
        """
        theta = math.radians(self.launch_angle_deg)
        psi = math.radians(self.azimuth_deg)
        horizontal = self.speed * math.cos(theta)
        return (
            horizontal * math.cos(psi),
            horizontal * math.sin(psi),
            self.speed * math.sin(theta),
        )

    def spin_vector(self) -> tuple[float, float, float]:
        """Initial angular velocity vector, rad/s, in the shot frame.

        Signs, worked through -- these are easy to get backwards, and getting
        sidespin backwards produces a model that recommends aiming at the
        wrong side of every pin.

        With the ball travelling downrange, u = (V, 0, 0), and spin vector
        omega = (0, -w_back, +w_side), the Magnus direction is::

            omega x u = (0, +w_side * V, +w_back * V)

        so backspin (w_back > 0) gives +z, i.e. lift, and positive sidespin
        gives +y, i.e. a curve to the player's right. Both as documented.

        The axis is referenced to the FLIGHT DIRECTION, not the target line
        --------------------------------------------------------------------
        The vector built above is then rotated about the vertical by
        ``azimuth_deg``, so it stays perpendicular to the direction the ball is
        actually launched rather than to the target line.

        This matters, and getting it wrong is subtle. Leave the axis fixed to
        the target line and a ball started 4 degrees right with "pure backspin"
        has a spin axis that is NOT square to its own flight; the Magnus force
        then acts in the wrong vertical plane and the ground track curves about
        1.5 yards over a 7-iron. That is real physics for that spin state, but
        it is not what "no sidespin" means: a launch monitor reports the spin
        axis relative to the ball's flight, so 0 degrees of axis tilt is a shot
        that does not curve, wherever it was started. With the rotation applied,
        ``azimuth_deg`` moves the start line and ``spin_axis_deg`` bends the
        ball, and the two decouple -- which is what ``caddie.shot`` relies on.

        No effect at all when ``azimuth_deg`` is zero, which is every straight
        shot and the whole calibration set.
        """
        if self.spin_axis_deg is not None:
            total = self.backspin_rpm * RPM_RADS
            tilt = math.radians(self.spin_axis_deg)
            # Rotate the pure-backspin axis about the flight axis, so a
            # positive tilt leans the axis into a rightward curve.
            spin = (0.0, -total * math.cos(tilt), total * math.sin(tilt))
        else:
            back = self.backspin_rpm * RPM_RADS
            side = self.sidespin_rpm * RPM_RADS
            spin = (0.0, -back, side)

        return self._align_with_launch(spin)

    def _align_with_launch(
        self, spin: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Rotate a flight-referenced spin vector into the shot frame.

        A rotation about +z by the launch azimuth, so the spin axis follows the
        start line. See :meth:`spin_vector` for why this is not optional.
        """
        psi = math.radians(self.azimuth_deg)
        if psi == 0.0:
            return spin
        wx, wy, wz = spin
        cos_p, sin_p = math.cos(psi), math.sin(psi)
        return (wx * cos_p - wy * sin_p, wx * sin_p + wy * cos_p, wz)

    @property
    def total_spin_rpm(self) -> float:
        wx, wy, wz = self.spin_vector()
        return math.hypot(math.hypot(wx, wy), wz) / RPM_RADS

    def describe(self) -> str:
        return (
            f"{self.ball_speed_mph:.1f} mph / {self.launch_angle_deg:.1f}deg / "
            f"{self.backspin_rpm:.0f} rpm back"
            + (f" / {self.sidespin_rpm:+.0f} rpm side" if self.sidespin_rpm else "")
            + (f" / axis {self.spin_axis_deg:+.1f}deg" if self.spin_axis_deg is not None else "")
        )


# ---------------------------------------------------------------------------
# PGA Tour average launch conditions, Trackman. These double as the
# calibration targets in calibrate.py and as a realistic default bag.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClubReference:
    """A club's tour-average launch numbers, plus its static loft.

    Attributes
    ----------
    loft_deg:
        Static clubface loft -- a DIFFERENT quantity from launch_angle_deg
        (dynamic loft and attack angle put the ball's actual launch a few
        degrees under the club's loft, more so for longer clubs). Standard
        manufacturer spec, not Trackman-sourced like the other fields;
        only used by caddie.physics.roll's ground-roll estimator, which
        explicitly wants real loft rather than launch angle.
    """

    name: str
    ball_speed_mph: float
    launch_angle_deg: float
    backspin_rpm: float
    published_carry_yd: float
    published_apex_yd: float | None = None
    published_descent_deg: float | None = None
    loft_deg: float | None = None

    def launch(self, **overrides) -> LaunchConditions:
        kwargs = dict(
            ball_speed_mph=self.ball_speed_mph,
            launch_angle_deg=self.launch_angle_deg,
            backspin_rpm=self.backspin_rpm,
        )
        kwargs.update(overrides)
        return LaunchConditions(**kwargs)


PGA_TOUR_AVERAGES: tuple[ClubReference, ...] = (
    ClubReference("Driver", 167.0, 10.9, 2686, 275.0, 32.0, 38.0, loft_deg=10.5),
    ClubReference("3-wood", 158.0, 9.2, 3655, 243.0, 30.0, 43.0, loft_deg=15.0),
    ClubReference("5-wood", 152.0, 9.4, 4350, 230.0, 31.0, 47.0, loft_deg=18.0),
    ClubReference("Hybrid", 146.0, 10.2, 4437, 225.0, 29.0, 47.0, loft_deg=19.0),
    ClubReference("3-iron", 142.0, 10.4, 4630, 212.0, 27.0, 46.0, loft_deg=21.0),
    ClubReference("4-iron", 137.0, 11.0, 4836, 203.0, 28.0, 48.0, loft_deg=24.0),
    ClubReference("5-iron", 132.0, 12.1, 5361, 194.0, 31.0, 49.0, loft_deg=27.0),
    ClubReference("6-iron", 127.0, 14.1, 6231, 183.0, 30.0, 50.0, loft_deg=30.0),
    ClubReference("7-iron", 120.0, 16.3, 7097, 172.0, 32.0, 50.0, loft_deg=34.0),
    ClubReference("8-iron", 115.0, 18.1, 7998, 160.0, 31.0, 50.0, loft_deg=38.0),
    ClubReference("9-iron", 109.0, 20.4, 8647, 148.0, 30.0, 51.0, loft_deg=42.0),
    ClubReference("PW", 102.0, 24.2, 9304, 136.0, 29.0, 52.0, loft_deg=46.0),
)
"""Trackman PGA Tour averages -- the full bag -- used as calibration targets.

PROVENANCE -- read before trusting a fit against these:
  * Every row is quoted from the one published Trackman "PGA Tour Averages"
    table: ball speed, launch angle, spin rate, max height (taken as apex) and
    land angle (taken as descent angle) all come from that single source.
    Keeping one source matters -- see the disagreement note below.
  * NO row is interpolated. An earlier version of this table carried five
    clubs with a 5-iron whose launch and spin were interpolated between the
    3-wood and 7-iron; the published 5-iron row (12.1 deg / 5361 rpm, not
    14.3 / 5280) replaces it, and it now constrains apex and descent too.
  * Independently republished tour-average tables disagree with this one by
    roughly 2-4%: a widely circulated alternative gives Driver 171 mph ball
    speed / 282 yd carry and 7-iron 123 / 176, against 167 / 275 and 120 / 172
    here. That spread is real and is the reason the model is fitted rather
    than assumed -- and the reason the note below matters more than the note
    above.
  * Apex and descent angle are reported less consistently than carry across
    sources. ``calibrate.py`` therefore weights carry hardest and treats apex
    and descent as soft constraints.

Replace this table with your own launch monitor session and re-run
``scripts/calibrate.py`` to personalise the model -- that is the intended
workflow, and it matters more than any published average, because a caddie
recommendation is only as good as its model of *your* ball flight."""


WEDGES: tuple[ClubReference, ...] = (
    ClubReference("50°", 95.4, 27.1, 9789.0, 120.0, loft_deg=50.0),
    ClubReference("52°", 92.1, 28.6, 10031.0, 112.0, loft_deg=52.0),
    ClubReference("54°", 88.9, 30.1, 10273.0, 104.0, loft_deg=54.0),
    ClubReference("56°", 85.6, 31.6, 10515.0, 96.0, loft_deg=56.0),
    ClubReference("58°", 82.3, 33.0, 10758.0, 88.0, loft_deg=58.0),
    ClubReference("60°", 79.0, 34.5, 11000.0, 80.0, loft_deg=60.0),
)
"""Wedges below PW, named by loft rather than GW/SW/LW -- those labels mean
different actual lofts to different players, loft numbers don't.

Kept OUT of PGA_TOUR_AVERAGES on purpose. The published Trackman table above
stops at PW; there is no single sourced row for anything shorter, so unlike
every entry above, these are EXTRAPOLATED, not quoted -- ball speed, launch
angle and spin are linear in loft along the same per-degree rate the table's
own 9-iron -> PW step (46 deg) already sets, and carry is this engine's own
prediction from those numbers, not an independent figure to fit against.
Real bags carry two or three of these (see any "what's in your bag" thread
for how much the exact combo varies), not all six -- see BAG_LIMIT and
EXTRA_VARIANTS. Replace with your own numbers (My Bag in the explorer, or
scripts/calibrate.py) before trusting these for anything that matters."""

OTHER_VARIANTS: tuple[ClubReference, ...] = (
    ClubReference("7-wood", 149.0, 9.8, 4390.0, 227.0, loft_deg=21.0),
    ClubReference("2-iron", 144.0, 9.8, 4500.0, 218.0, loft_deg=18.0),
)
"""7-wood and 2-iron -- same EXTRAPOLATED-not-quoted status as WEDGES, added
for bag-building variety (a 7-wood between 5-wood and Hybrid; a 2-iron
between Hybrid and 3-iron, lower-launching and lower-spin as a blade-style
long iron typically is)."""

EXTRA_VARIANTS: tuple[ClubReference, ...] = OTHER_VARIANTS + WEDGES
"""Everything offered beyond PGA_TOUR_AVERAGES, in one pool: 7-wood, 2-iron,
then the six wedge lofts. Order matches how a bag-builder UI should list
them -- woods/irons before wedges."""

# USGA Rule 4.1b: 14 clubs total including the putter. Putting in this
# project is a declared result, not a simulated club (see explorer.py's
# putting mechanic) -- there is no simulated putter to occupy a slot -- so
# the cap on clubs that actually get swung here is 13.
BAG_LIMIT = 13
