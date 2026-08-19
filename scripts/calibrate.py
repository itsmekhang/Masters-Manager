"""Fit the aerodynamic model to measured launch-condition -> outcome data.

Why fit instead of using published coefficients
-----------------------------------------------
Published golf-ball drag/lift fits disagree by 10-15%, which is ~12 yards on
a driver. But we have a strong constraint the coefficient papers do not: a
table of (ball speed, launch angle, spin) -> (carry, apex, descent angle) for
the full bag, spanning S = 0.05 to 0.35. Six free coefficients against several
dozen measurements is a well-posed (and heavily overdetermined) inverse
problem, and solving it forces the aerodynamics to reproduce shots that
actually happened.

Run:  python scripts/calibrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caddie.physics.aero import SMITS_SMITH_1994, AeroModel
from caddie.physics.atmosphere import Atmosphere
from caddie.physics.ball import PGA_TOUR_AVERAGES, Ball
from caddie.physics.trajectory import FlightIntegrator

# Trackman tour averages are gathered at tournament venues in benign
# conditions; near-sea-level standard-ish air, calm, is the right assumption.
CALIB_ATM = Atmosphere.from_conditions(temp_c=21.0, altitude_m=0.0, relative_humidity=0.5)

# Residual scaling: how much error we consider "one unit" per observable.
# Carry is weighted hardest because it is what a caddie actually needs.
TOL_CARRY_YD = 1.5
TOL_APEX_YD = 6.0
TOL_DESCENT_DEG = 3.0


def simulate(params: np.ndarray, dt: float = 0.002):
    cd0, cd_s, cd_s2, cl_s, cl_s2, tau = params
    aero = AeroModel(cd0=cd0, cd_s=cd_s, cd_s2=cd_s2, cl_s=cl_s, cl_s2=cl_s2)
    ball = Ball(spin_decay_tau=tau)
    integ = FlightIntegrator(atmosphere=CALIB_ATM, aero=aero, ball=ball, dt=dt)
    return [integ.integrate(club.launch()) for club in PGA_TOUR_AVERAGES]


def residuals(params: np.ndarray) -> np.ndarray:
    out = []
    # Coarser dt during fitting; verified against dt=0.001 at the end.
    for club, traj in zip(PGA_TOUR_AVERAGES, simulate(params, dt=0.004)):
        out.append((traj.carry_yards - club.published_carry_yd) / TOL_CARRY_YD)
        if club.published_apex_yd is not None:
            out.append((traj.apex_yards - club.published_apex_yd) / TOL_APEX_YD)
        if club.published_descent_deg is not None:
            out.append(
                (traj.descent_angle_deg - club.published_descent_deg) / TOL_DESCENT_DEG
            )
    return np.array(out)


def report(title: str, aero: AeroModel, ball: Ball) -> float:
    integ = FlightIntegrator(atmosphere=CALIB_ATM, aero=aero, ball=ball, dt=0.001)
    print()
    print(title)
    print("-" * len(title))
    print(
        f"{'club':<8} {'carry':>16} {'apex':>14} {'descent':>14}   "
        f"{'S_launch':>8} {'C_D':>5} {'C_L':>5}"
    )
    worst = 0.0
    for club in PGA_TOUR_AVERAGES:
        launch = club.launch()
        traj = integ.integrate(launch)
        d_carry = traj.carry_yards - club.published_carry_yd
        worst = max(worst, abs(d_carry))

        def cmp(sim: float, pub: float | None) -> str:
            """Format 'sim vs published (delta)', or note an unconstrained row."""
            if pub is None:
                return f"{sim:5.1f} vs    -- (  --- )"
            return f"{sim:5.1f} vs {pub:4.1f} ({sim - pub:+5.1f})"

        s0 = aero.spin_ratio(launch.speed, abs(launch.spin_vector()[1]), Ball().radius)
        re0 = aero.reynolds(launch.speed, CALIB_ATM.kinematic_viscosity)
        print(
            f"{club.name:<8} "
            f"{traj.carry_yards:6.1f} vs {club.published_carry_yd:5.1f} ({d_carry:+5.1f}) "
            f"{cmp(traj.apex_yards, club.published_apex_yd)} "
            f"{cmp(traj.descent_angle_deg, club.published_descent_deg)}   "
            f"{s0:8.3f} {aero.drag_coefficient(s0, re0):5.3f} "
            f"{aero.lift_coefficient(s0):5.3f}"
        )
    print(f"worst carry error: {worst:.2f} yd")
    return worst


def main() -> None:
    print("Calibration air:", CALIB_ATM.describe())

    # Baseline: the textbook model, unfitted.
    report("BEFORE -- Smits & Smith (1994) published fits", SMITS_SMITH_1994, Ball())

    x0 = np.array([0.24, 0.18, 0.0, 1.99, -3.25, 24.0])
    # Spin decay is bounded to the physically observed 3-7 %/s band
    # (tau = 14-33 s). Left unbounded the optimiser drives tau to ~60 s and
    # buys carry accuracy with an unphysical spin history -- a good fit for
    # the wrong reason, which would then mispredict any shot whose flight
    # time differs from the calibration set.
    bounds = (
        np.array([0.15, 0.00, -1.0, 1.0, -6.0, 14.0]),
        np.array([0.32, 1.00, 0.5, 3.0, -1.0, 33.0]),
    )

    print()
    print(
        f"fitting {len(x0)} coefficients to {len(residuals(x0))} measurements "
        f"across {len(PGA_TOUR_AVERAGES)} clubs ..."
    )
    sol = least_squares(residuals, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, verbose=0)

    cd0, cd_s, cd_s2, cl_s, cl_s2, tau = sol.x
    fitted_aero = AeroModel(cd0=cd0, cd_s=cd_s, cd_s2=cd_s2, cl_s=cl_s, cl_s2=cl_s2)
    fitted_ball = Ball(spin_decay_tau=tau)

    worst = report("AFTER -- fitted", fitted_aero, fitted_ball)

    print()
    print("fitted coefficients (paste into caddie/physics/aero.py + ball.py):")
    print(f"    cd0   = {cd0:.4f}")
    print(f"    cd_s  = {cd_s:.4f}")
    print(f"    cd_s2 = {cd_s2:.4f}")
    print(f"    cl_s  = {cl_s:.4f}")
    print(f"    cl_s2 = {cl_s2:.4f}")
    print(f"    spin_decay_tau = {tau:.2f}  s  "
          f"({100.0 / tau:.1f} %/s initial decay)")
    print(f"    cost = {sol.cost:.4f}, worst carry error = {worst:.2f} yd")


if __name__ == "__main__":
    main()
