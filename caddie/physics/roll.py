"""Empirical ground-roll estimator.

``caddie.physics`` integrates carry precisely (see ``trajectory.py``) but
stops at landing -- "Carry only. No bounce or roll" is one of the explorer's
own caveats. This module fills that gap with a hand-fit empirical model, NOT
a physics simulation: exact roll depends on spin, landing angle, turf
firmness, slope and moisture, and this engine doesn't have good data for any
of those at the ball-turf interface. Treat it like ``FLIGHT_HEIGHTS`` in
shot.py -- directionally reasonable, numerically a starting estimate, meant
to be replaced by your own observed numbers once you have them.
"""
from __future__ import annotations

import math


def estimate_roll_yards(
    carry_yd: float,
    ball_speed_mph: float,
    loft_deg: float,
    wind_component_mph: float = 0.0,
    elevation_ft: float = 0.0,
    firmness: float = 1.0,
) -> float:
    """Roll after landing, in yards.

    Parameters
    ----------
    carry_yd:
        Y -- carry distance in the air.
    ball_speed_mph:
        V -- ball speed off the club.
    loft_deg:
        L -- club loft in degrees. ``ClubReference`` stores launch angle,
        not loft (they differ -- dynamic loft, attack angle); launch angle
        is the closest available stand-in and is what callers should pass
        unless they have real loft data.
    wind_component_mph:
        W -- wind ALONG the shot's line, tailwind positive, headwind
        negative. For a wind at an angle to the shot, project it first::

            wind_component_mph = wind_speed_mph * cos(radians(angle_from_shot_line))

        where 0 deg is a pure tailwind (this is NOT the same convention as
        ``WindField.direction_from_deg``, where 0 deg is a headwind -- see
        ``wind_component_along_shot`` below for the conversion).
    elevation_ft:
        E -- target elevation relative to the ball, uphill positive. Pass 0
        if unreliable, per the formula's own guidance.
    firmness:
        Multiplier applied AFTER the base estimate is capped (see below):
        ~0.5 for soft/wet greens, 1.0 for normal turf, 1.3-1.6 for firm, dry
        ground. Deliberately allowed to push the result back past the base
        cap -- firm ground legitimately rolls out more than 35% of carry on
        a low shot; the cap bounds the uncalibrated base formula, not the
        physical ceiling.

    Returns roll in yards.
    """
    Y, V, L, W, E = carry_yd, ball_speed_mph, loft_deg, wind_component_mph, elevation_ft
    if Y <= 0:
        return 0.0

    v_ref = 43.0 + 0.47 * Y

    slope_term_deg = math.degrees(math.atan(E / (3.0 * Y)))
    descent_deg = 31.0 + 0.45 * L - 0.12 * W - 0.30 * slope_term_deg
    descent_deg = max(32.0, min(55.0, descent_deg))

    speed_ratio = (V / v_ref) if v_ref > 0 else 1.0
    roll = (
        Y * 0.12
        * math.exp(-0.09 * (descent_deg - 35.0))
        * speed_ratio ** 1.7
        * math.exp(0.012 * W)
    )
    roll = max(0.0, min(roll, 0.35 * Y))  # cap the base (uncalibrated) estimate
    roll *= firmness  # then apply turf firmness, which may push past that cap
    return max(0.0, roll)


def wind_component_along_shot(wind_speed_mph: float, direction_from_deg: float) -> float:
    """Convert a WindField-style direction into this module's W.

    ``direction_from_deg`` is meteorological (0 = blowing INTO the player's
    face, a headwind), matching ``WindField.direction_from_deg``. This
    module wants the opposite sign convention (0 = tailwind, positive =
    helping), so the two are 180 degrees apart: W = -speed * cos(direction).
    """
    return -wind_speed_mph * math.cos(math.radians(direction_from_deg))
