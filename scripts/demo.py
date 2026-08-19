"""End-to-end demo: real Augusta geometry + real elevation + real physics.

Run:  python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caddie.course import load_augusta, terrain_from_hole
from caddie.physics import (
    AUGUSTA_APRIL_TYPICAL,
    PGA_TOUR_AVERAGES,
    FlightIntegrator,
    WindField,
)
from caddie.physics.wind import AUGUSTA_SHELTERED_CORRIDOR
from caddie.shot import ShotShape, plan_shot

YARD_M = 0.9144


def club(name: str):
    """Look a club up in the tour-average table.

    Read from the table rather than hardcoded here, so the demo cannot drift
    out of sync with the calibration set -- an earlier version of this script
    called its shot a 9-iron while using the 8-iron's ball speed.
    """
    return next(c for c in PGA_TOUR_AVERAGES if c.name == name)


def main() -> None:
    course = load_augusta()
    print(course.summary())

    print()
    print("=" * 78)
    print("Air:", AUGUSTA_APRIL_TYPICAL.describe())
    integ = FlightIntegrator(atmosphere=AUGUSTA_APRIL_TYPICAL, dt=0.002)

    # ---------------------------------------------------------------
    # Hole 12, Golden Bell. 155 yards, slightly downhill, over Rae's
    # Creek, and the most wind-confounded tee shot in golf.
    # ---------------------------------------------------------------
    h = course.hole(12)
    frame = h.frame_from(h.tee, h.pin)
    terrain = terrain_from_hole(h, frame)
    px, py, pz = frame.to_local(h.pin)

    print()
    print("=" * 78)
    print(f"HOLE {h.number} -- {h.name}, par {h.par}")
    print("=" * 78)
    print(f"  card yardage        {h.card_yardage} yd")
    print(f"  measured tee->pin   {h.straight_yards:.1f} yd")
    print(f"  elevation change    {h.elevation_change_m:+.1f} m "
          f"({h.elevation_change_m / 0.3048:+.1f} ft)")
    print(f"  naive 'plays like'  {h.tee.plays_like_yards_to(h.pin):.1f} yd")
    print(f"  pin in shot frame   x={px:.1f} m, y={py:+.2f} m, z={pz:+.1f} m")

    # A stock 8-iron-ish flight. We search launch conditions for the shot
    # that finishes pin-high, which is what a caddie actually needs.
    nine = club("9-iron")
    launch = nine.launch()

    print()
    print(f"  Stock {nine.name} ({launch.describe()}) in varying wind:")
    print(f"  {'wind':<22} {'carry':>8} {'offline':>9} {'apex':>7} {'descent':>8} "
          f"{'vs pin':>8}")
    print("  " + "-" * 68)
    winds = [
        ("calm", None),
        ("10 mph in", WindField.from_mph(10, 0, **AUGUSTA_SHELTERED_CORRIDOR)),
        ("10 mph down", WindField.from_mph(10, 180, **AUGUSTA_SHELTERED_CORRIDOR)),
        ("10 mph from right", WindField.from_mph(10, 90, **AUGUSTA_SHELTERED_CORRIDOR)),
        ("10 mph from left", WindField.from_mph(10, 270, **AUGUSTA_SHELTERED_CORRIDOR)),
    ]
    for label, wind in winds:
        t = integ.integrate(launch, wind=wind, terrain=terrain)
        vs_pin = t.carry_yards - px / YARD_M
        print(f"  {label:<22} {t.carry_yards:7.1f}y {t.offline_yards:+8.1f}y "
              f"{t.apex_yards:6.1f}y {t.descent_angle_deg:7.1f}d {vs_pin:+7.1f}y")

    print()
    print("  Note the asymmetry: the headwind costs more than the tailwind")
    print("  gives back, and the sheltered profile means the ball only meets")
    print("  the true wind near its apex, above the pines.")

    # ---------------------------------------------------------------
    # Hole 10 -- the 100 ft drop. Shows why terrain must be integrated,
    # not approximated.
    # ---------------------------------------------------------------
    h10 = course.hole(10)
    frame10 = h10.tee_frame()
    terrain10 = terrain_from_hole(h10, frame10)

    print()
    print("=" * 78)
    print(f"HOLE {h10.number} -- {h10.name}: why terrain must be integrated")
    print("=" * 78)
    print(f"  elevation change  {h10.elevation_change_m:+.1f} m "
          f"({h10.elevation_change_m / 0.3048:+.1f} ft) -- the big drop")

    driver = club("Driver").launch()
    flat = integ.integrate(driver)
    real = integ.integrate(driver, terrain=terrain10)
    print(f"  driver carry on flat ground     {flat.carry_yards:6.1f} yd")
    print(f"  driver carry on real terrain    {real.carry_yards:6.1f} yd  "
          f"({real.carry_yards - flat.carry_yards:+.1f} yd)")
    print(f"  naive elevation rule of thumb   "
          f"{flat.carry_yards - h10.elevation_change_m / YARD_M:6.1f} yd")
    print()
    print("  The rule of thumb and the integration disagree because a ball")
    print("  falling into a valley also spends longer in the air, keeps")
    print("  decelerating, and lands shallower. Only one of the two knows that.")

    # ---------------------------------------------------------------
    # Hole 13 -- Azalea. An 83 degree dogleg left: the shot is a draw
    # around the corner, which is what the shaping layer is for.
    # ---------------------------------------------------------------
    h13 = course.hole(13)
    frame13 = h13.tee_frame()
    terrain13 = terrain_from_hole(h13, frame13)

    print()
    print("=" * 78)
    print(f"HOLE {h13.number} -- {h13.name}: shaping a tee shot")
    print("=" * 78)
    print(f"  dogleg            {h13.dogleg_deg:.1f}deg left -- the most severe on "
          f"the course")
    print("  Driver, drawn around the corner. The solver picks the spin axis")
    print("  for the requested curve AND the start line that brings it back")
    print("  to the target line.")
    print()
    print(f"  {'shot':<18} {'aim':>7} {'axis':>7} {'curve':>8} {'carry':>8} "
          f"{'cost':>7} {'finish':>8}")
    print("  " + "-" * 68)

    for curve in (0.0, -10.0, -20.0, -30.0):
        shape = ShotShape(curve_yards=curve, hand="RH")
        plan = plan_shot(integ, club("Driver"), shape, terrain=terrain13)
        print(
            f"  {shape.describe():<18} {plan.aim_deg:+6.1f}d {plan.spin_axis_deg:+6.1f}d "
            f"{plan.curve_achieved_yards:+7.1f}y {plan.trajectory.carry_yards:7.1f}y "
            f"{plan.shaping_cost_yards:+6.1f}y {plan.finish_offline_yards:+7.1f}y"
        )

    print()
    print("  The cost column is not a penalty applied on top -- it is what")
    print("  happens when lift tilts out of the vertical by cos(axis), so a")
    print("  bigger curve buys shape with carry.")

    # A knockdown into the wind, on 12, where holding the ball matters.
    print()
    print("  Same 9-iron on 12 into 15 mph, stock vs knockdown:")
    print(f"  {'flight':<14} {'carry':>8} {'apex':>7} {'descent':>9} {'vs pin':>8}")
    print("  " + "-" * 52)
    headwind = WindField.from_mph(15, 0, **AUGUSTA_SHELTERED_CORRIDOR)
    for height in ("stock", "knockdown"):
        plan = plan_shot(
            integ, nine, ShotShape(height=height), wind=headwind, terrain=terrain
        )
        t = plan.trajectory
        print(
            f"  {height:<14} {t.carry_yards:7.1f}y {t.apex_yards:6.1f}y "
            f"{t.descent_angle_deg:8.1f}d {t.carry_yards - px / YARD_M:+7.1f}y"
        )
    print()
    print("  NOTE: the knockdown deltas are hand-set, not measured -- see")
    print("  FLIGHT_HEIGHTS in caddie/shot.py. The signs are right; the")
    print("  magnitudes want a launch monitor session.")


if __name__ == "__main__":
    main()
