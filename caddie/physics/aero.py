"""Golf ball aerodynamics: drag and Magnus lift coefficients.

Why this file is parameterised rather than hard-coded
-----------------------------------------------------
The literature fits (Bearman & Harvey 1976 wind-tunnel data; the widely
quoted Smits & Smith 1994 fits) disagree with each other by 10-15% and were
measured on balls whose dimple patterns are two ball generations old. A 10%
error in C_D is ~12 yards on a driver, which is the difference between the
fairway and the pine straw.

So the model here has a documented functional *form* taken from the
literature, with free coefficients that are **fitted to measured launch
condition -> carry distance data** (see ``calibrate.py``). The defaults below
are the result of that fit. Re-run the calibration against your own launch
monitor data to personalise it.

Non-dimensional groups
----------------------
Spin ratio (a.k.a. spin factor)::

    S = omega * r / V

Reynolds number::

    Re = V * D / nu

For a driver, V ~ 75 m/s and S ~ 0.08; for a wedge, V ~ 30 m/s and S ~ 0.35.
Lift coefficient rises steeply with S then saturates near S ~ 0.3, which is
why a wedge does not fly to the moon.
"""
from __future__ import annotations

from dataclasses import dataclass

from .constants import BALL_DIAMETER_MIN_M


@dataclass(frozen=True)
class AeroModel:
    """Drag/lift coefficient model.

    Functional form
    ---------------
    ``C_D = cd0 + cd_s * S + cd_s2 * S^2`` with a weak Reynolds correction.
    ``C_L = cl_s * S + cl_s2 * S^2``, clipped at ``cl_max``.

    The quadratic lift term is negative, producing the saturation seen in
    every published dataset. The form matches Smits & Smith (1994); the
    coefficient values are ours, from calibration.
    """

    # Drag. Fitted by scripts/calibrate.py against the Trackman tour table;
    # worst-case carry error 4.2 yd across driver..pitching wedge, versus
    # 13.9 yd for the published Smits & Smith coefficients.
    cd0: float = 0.1989
    cd_s: float = 0.6450
    cd_s2: float = -0.6285
    # Weak Re dependence: C_D scales by (1 + cd_re * ln(Re / re_ref)).
    # Golf balls are dimpled precisely to push the drag crisis below the
    # speeds of play, so above Re ~ 7e4 this term is small. Default 0 keeps
    # the model honest -- enable only if you have data resolving it.
    cd_re: float = 0.0
    re_ref: float = 1.5e5

    # Lift
    cl_s: float = 1.9569
    cl_s2: float = -1.0042
    cl_max: float = 0.34
    # KNOWN LIMITATION: the fit reproduces carry to ~4 yd but runs 5-11 yd
    # high on APEX for the low-spin long clubs, i.e. it slightly over-lifts.
    # Carry is right because the extra height is paid back as extra drag.
    # Consequence: descent angle and therefore predicted stopping power carry
    # more uncertainty than carry distance does. Feeding a real launch-monitor
    # session (with apex) into calibrate.py is the fix.

    # Guard rails. Outside this spin-ratio band the fits are extrapolation;
    # we clamp S rather than let a wild value produce a wild trajectory.
    s_min: float = 0.0
    s_max: float = 0.55

    def spin_ratio(self, speed: float, spin_rads: float, radius: float) -> float:
        if speed <= 1e-6:
            return 0.0
        s = spin_rads * radius / speed
        return min(max(s, self.s_min), self.s_max)

    def drag_coefficient(self, spin_ratio: float, reynolds: float) -> float:
        s = spin_ratio
        cd = self.cd0 + self.cd_s * s + self.cd_s2 * s * s
        if self.cd_re != 0.0 and reynolds > 1e3:
            import math

            cd *= 1.0 + self.cd_re * math.log(reynolds / self.re_ref)
        # A golf ball's drag coefficient never leaves this range in play.
        return min(max(cd, 0.15), 0.60)

    def lift_coefficient(self, spin_ratio: float) -> float:
        s = spin_ratio
        cl = self.cl_s * s + self.cl_s2 * s * s
        return min(max(cl, 0.0), self.cl_max)

    def reynolds(self, speed: float, kinematic_viscosity: float) -> float:
        return speed * BALL_DIAMETER_MIN_M / kinematic_viscosity


# ---------------------------------------------------------------------------
# Reference models, for comparison and for anyone who wants the textbook
# values instead of our calibration.
# ---------------------------------------------------------------------------

SMITS_SMITH_1994 = AeroModel(
    cd0=0.24, cd_s=0.18, cd_s2=0.0,
    cl_s=1.99, cl_s2=-3.25, cl_max=0.45,
)
"""The commonly cited Smits & Smith (1994) fits:
``C_D = 0.24 + 0.18 S``, ``C_L = 1.99 S - 3.25 S^2``.
Stated validity roughly 0.02 < S < 0.35, 0.4e5 < Re < 2.5e5.
Treat as a cross-check, not ground truth -- see module docstring."""


# ---------------------------------------------------------------------------
# An independent C_L(S) dataset, and what it says about our fit.
# ---------------------------------------------------------------------------
EMPIRICAL_CL_TABLE: tuple[tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.04, 0.10),
    (0.10, 0.16),
    (0.20, 0.23),
    (0.40, 0.33),
)
"""Tabulated lift coefficient against spin ratio, linearly interpolated.

Source: the ``cagrell/golfmodel`` implementation of MacDonald & Hanzely,
"The physics of the drive in golf", Am. J. Phys. 59(3) 1991, which carries this
as its empirical C_L relation. Independent of the Trackman carry table this
project calibrates against, which is what makes it useful: it constrains the
lift coefficient DIRECTLY, where our calibration only ever sees C_L through its
effect on carry, apex and descent angle.

WHAT IT REVEALS -- our fit over-lifts through the middle of the bag:

    S      table   ours    Smits & Smith     club
    0.08   0.140   0.150   0.138             Driver
    0.14   0.188   0.254   0.215             5-wood
    0.20   0.230   0.340   0.268             5-iron
    0.30   0.280   0.340   0.304             7-iron
    0.46   0.330   0.340   0.228             PW

Across 0.04 < S < 0.46 our C_L runs +0.047 high on average and +0.113 high at
worst, the worst of it in the 0.14-0.30 band. Smits & Smith tracks the table
almost exactly on average (mean -0.000) but collapses at high S.

This is the mechanism behind the apex limitation noted above, and it is a
DEGENERACY, not a coincidence. Fitting six coefficients to carry/apex/descent
lets the optimiser trade lift against drag: raise both together and carry is
preserved while the ball flies higher. Refitting against all 12 clubs makes it
plain -- worst carry error 4.75 yd, but apex then runs +8 to +10.6 yd high on
the woods and long irons, exactly where this table says C_L is most inflated.
Smits & Smith has the lift roughly right and the carry wrong, so ITS error is
in the drag; ours has the carry right and the lift wrong.

THE INDICATED FIX, not yet done: constrain C_L to this table (or fit only a
small correction to it) and let the calibration move the drag coefficients
alone. That is a better-posed problem than the current one -- fewer free
parameters against the same 36 measurements, and the one quantity the Trackman
table cannot pin down is supplied from outside instead of inferred."""


def tabulated_lift_coefficient(spin_ratio: float) -> float:
    """C_L from :data:`EMPIRICAL_CL_TABLE`, linearly interpolated and flat outside."""
    xs = [s for s, _ in EMPIRICAL_CL_TABLE]
    ys = [cl for _, cl in EMPIRICAL_CL_TABLE]
    if spin_ratio <= xs[0]:
        return ys[0]
    if spin_ratio >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(EMPIRICAL_CL_TABLE, EMPIRICAL_CL_TABLE[1:]):
        if spin_ratio <= x1:
            return y0 + (y1 - y0) * (spin_ratio - x0) / (x1 - x0)
    return ys[-1]
