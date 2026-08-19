"""Shot-shaping tests.

Same standard as the physics suite: every test asserts something that a
plausible-but-wrong implementation would get wrong. Several of these would fail
if the sidespin sign were flipped, if shaping were modelled by adding sidespin
instead of tilting the axis, or if "curve" were measured from the target line
instead of the start line.
"""
from __future__ import annotations

import math

import pytest

from caddie.physics import Atmosphere, FlightIntegrator, LaunchConditions, WindField
from caddie.physics.ball import PGA_TOUR_AVERAGES
from caddie.shot import (
    FLIGHT_HEIGHTS,
    DPlaneModel,
    ShotShape,
    bend_yards,
    calibrate_face_to_path,
    plan_shot,
    solve_aim,
    solve_spin_axis,
    with_spin_axis,
)


@pytest.fixture(scope="module")
def integ():
    return FlightIntegrator(
        atmosphere=Atmosphere.from_conditions(temp_c=21.0, altitude_m=0.0),
        dt=0.002,
    )


def club(name: str):
    return next(c for c in PGA_TOUR_AVERAGES if c.name == name)


SEVEN_IRON = LaunchConditions(120.0, 16.3, 7097)


# --- Conventions -----------------------------------------------------------

def test_positive_spin_axis_curves_right(integ):
    """The sign that, if flipped, aims every shaped shot at the wrong side."""
    right = integ.integrate(with_spin_axis(SEVEN_IRON, 15.0))
    left = integ.integrate(with_spin_axis(SEVEN_IRON, -15.0))
    assert right.offline_yards > 5.0
    assert left.offline_yards < -5.0
    assert right.offline_yards == pytest.approx(-left.offline_yards, abs=0.5)


def test_tilting_the_axis_preserves_total_spin():
    """A shaped ball spins the same amount, just about a different axis.

    If shaping added sidespin on top of unchanged backspin, total spin would
    rise, C_L would rise, and the shaped shot would fly FURTHER than the
    straight one -- backwards from reality. This is the guard against that.
    """
    straight = SEVEN_IRON
    shaped = with_spin_axis(SEVEN_IRON, 20.0)
    assert shaped.total_spin_rpm == pytest.approx(straight.total_spin_rpm, rel=1e-9)


def test_bend_is_measured_from_the_start_line(integ):
    """A pushed straight shot lands offline but has not bent at all."""
    pushed = LaunchConditions(120.0, 16.3, 7097, azimuth_deg=4.0)
    traj = integ.integrate(pushed)
    assert traj.offline_yards > 8.0          # it finishes well right...
    assert abs(bend_yards(traj, 4.0)) < 1.0  # ...but it never curved


# --- The cost of shaping ---------------------------------------------------

def test_shaping_costs_carry(integ):
    """Tilting lift out of the vertical must shorten the shot.

    Not an applied penalty -- a consequence of the Magnus force rotating. If
    this fails, the lift decomposition is wrong.
    """
    straight = integ.integrate(SEVEN_IRON)
    shaped = integ.integrate(with_spin_axis(SEVEN_IRON, 25.0))
    assert shaped.carry_yards < straight.carry_yards - 1.0


def test_shaping_lowers_apex(integ):
    """cos(tilt) on the vertical Magnus term means a shaped ball flies lower."""
    straight = integ.integrate(SEVEN_IRON)
    shaped = integ.integrate(with_spin_axis(SEVEN_IRON, 25.0))
    assert shaped.apex_yards < straight.apex_yards


def test_more_tilt_curves_more_monotonically(integ):
    """Monotonicity is what makes the bisection solvers valid."""
    bends = [
        bend_yards(integ.integrate(with_spin_axis(SEVEN_IRON, t)), 0.0)
        for t in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0)
    ]
    assert bends == sorted(bends)
    assert bends[0] == pytest.approx(0.0, abs=0.2)


# --- Solvers ---------------------------------------------------------------

@pytest.mark.parametrize("curve", [-15.0, -8.0, -3.0, 3.0, 8.0, 15.0])
def test_solve_spin_axis_hits_the_requested_curve(integ, curve):
    axis = solve_spin_axis(integ, SEVEN_IRON, curve)
    traj = integ.integrate(with_spin_axis(SEVEN_IRON, axis))
    assert bend_yards(traj, 0.0) == pytest.approx(curve, abs=0.3)


def test_solve_spin_axis_sign_matches_requested_side(integ):
    assert solve_spin_axis(integ, SEVEN_IRON, 10.0) > 0
    assert solve_spin_axis(integ, SEVEN_IRON, -10.0) < 0


def test_solve_spin_axis_refuses_the_impossible(integ):
    """A pitching wedge cannot be bent 80 yards. Say so, don't extrapolate."""
    with pytest.raises(ValueError, match="cannot bend"):
        solve_spin_axis(integ, club("PW").launch(), 80.0)


def test_solve_aim_puts_a_curving_ball_on_the_target_line(integ):
    shaped = with_spin_axis(SEVEN_IRON, 20.0)
    unaimed = integ.integrate(shaped)
    assert unaimed.offline_yards > 5.0  # drifts right of target if aimed straight

    aim = solve_aim(integ, shaped)
    assert aim < 0.0  # a fade must be started LEFT to finish on line
    aimed = integ.integrate(
        LaunchConditions(120.0, 16.3, 7097, azimuth_deg=aim, spin_axis_deg=20.0)
    )
    assert aimed.offline_yards == pytest.approx(0.0, abs=0.3)


# --- plan_shot: curve and aim together -------------------------------------

@pytest.mark.parametrize("curve", [-10.0, -5.0, 5.0, 10.0])
def test_plan_shot_curves_as_asked_and_finishes_on_line(integ, curve):
    plan = plan_shot(integ, club("7-iron"), ShotShape(curve_yards=curve))
    assert plan.converged
    assert plan.curve_achieved_yards == pytest.approx(curve, abs=0.4)
    assert plan.finish_offline_yards == pytest.approx(0.0, abs=0.4)


def test_plan_shot_reports_a_negative_shaping_cost(integ):
    plan = plan_shot(integ, club("7-iron"), ShotShape(curve_yards=12.0))
    assert plan.shaping_cost_yards < 0.0


def test_draw_and_fade_are_mirror_images_in_still_air(integ):
    """No physical asymmetry between the two sides without wind or handedness."""
    fade = plan_shot(integ, club("7-iron"), ShotShape(curve_yards=8.0))
    draw = plan_shot(integ, club("7-iron"), ShotShape(curve_yards=-8.0))
    assert fade.spin_axis_deg == pytest.approx(-draw.spin_axis_deg, abs=0.3)
    assert fade.aim_deg == pytest.approx(-draw.aim_deg, abs=0.2)
    assert fade.trajectory.carry_yards == pytest.approx(
        draw.trajectory.carry_yards, abs=0.3
    )


def test_holding_a_ball_straight_against_a_crosswind_needs_tilt(integ):
    """Zero *observed* curve in a crosswind is a real shot, and it isn't zero tilt.

    Curve is defined against the start line including wind drift, so asking for
    a straight ball in a left-to-right wind should make the solver spin the ball
    the other way to hold it.
    """
    wind = WindField.from_mph(15.0, 270.0)  # from the player's left, pushes right
    plan = plan_shot(integ, club("7-iron"), ShotShape(curve_yards=0.0), wind=wind)
    assert plan.spin_axis_deg < -1.0  # must curve it left to cancel the drift
    assert plan.curve_achieved_yards == pytest.approx(0.0, abs=0.4)


# --- Layer 2: flighting ----------------------------------------------------

def test_knockdown_flies_lower_and_shorter(integ):
    stock = plan_shot(integ, club("7-iron"), ShotShape(height="stock"))
    down = plan_shot(integ, club("7-iron"), ShotShape(height="knockdown"))
    assert down.trajectory.apex_yards < stock.trajectory.apex_yards
    assert down.trajectory.carry_yards < stock.trajectory.carry_yards


def test_flighted_flies_higher_and_lands_steeper(integ):
    stock = plan_shot(integ, club("7-iron"), ShotShape(height="stock"))
    up = plan_shot(integ, club("7-iron"), ShotShape(height="flighted"))
    assert up.trajectory.apex_yards > stock.trajectory.apex_yards
    assert up.trajectory.descent_angle_deg > stock.trajectory.descent_angle_deg


def test_stock_height_is_the_identity():
    launch = SEVEN_IRON
    assert FLIGHT_HEIGHTS["stock"].apply(launch) == launch


def test_unknown_height_is_rejected():
    with pytest.raises(ValueError, match="unknown height"):
        ShotShape(height="banana")


# --- Naming / handedness ---------------------------------------------------

def test_handedness_only_renames_never_reshapes():
    rh = ShotShape(curve_yards=-8.0, hand="RH")
    lh = ShotShape(curve_yards=-8.0, hand="LH")
    assert rh.shape_name == "draw"
    assert lh.shape_name == "fade"
    assert rh.curve_yards == lh.curve_yards  # the physics is identical


def test_small_curve_reads_as_straight():
    assert ShotShape(curve_yards=0.4).shape_name == "straight"


# --- Layer 3: D-plane ------------------------------------------------------

def test_closed_face_relative_to_path_draws_the_ball():
    """Face left of path is a draw. The relationship the whole layer rests on."""
    model = DPlaneModel()
    assert model.spin_axis_deg(face_deg=1.0, path_deg=4.0) < 0.0
    assert model.spin_axis_deg(face_deg=4.0, path_deg=1.0) > 0.0


def test_start_line_is_dominated_by_the_face():
    model = DPlaneModel(face_weight=0.8)
    start = model.start_line_deg(face_deg=5.0, path_deg=0.0)
    assert start == pytest.approx(4.0)
    assert start > model.start_line_deg(face_deg=0.0, path_deg=5.0)


def test_dplane_inverse_round_trips():
    model = DPlaneModel()
    face, path = model.face_path_for(spin_axis_deg=-12.0, start_line_deg=3.0)
    assert model.spin_axis_deg(face, path) == pytest.approx(-12.0, abs=1e-9)
    assert model.start_line_deg(face, path) == pytest.approx(3.0, abs=1e-9)


def test_dplane_launch_from_swing_produces_the_expected_flight(integ):
    """A face closed to the path should actually draw when flown."""
    model = DPlaneModel()
    launch = model.launch_from_swing(SEVEN_IRON, face_deg=1.0, path_deg=4.0)
    traj = integ.integrate(launch)
    assert bend_yards(traj, launch.azimuth_deg) < -2.0


def test_calibrate_face_to_path_recovers_a_known_constant(integ):
    """Round trip: build an observation from a known constant, measure it back.

    This is the function that turns Layer 3's one unjustified number into a
    measurement, so it had better be self-consistent.
    """
    truth = 2.5
    face_to_path = 3.0
    axis = truth * face_to_path
    observed_curve = bend_yards(
        integ.integrate(with_spin_axis(SEVEN_IRON, axis)), 0.0
    )

    measured = calibrate_face_to_path(
        integ, SEVEN_IRON, face_to_path_deg=face_to_path,
        observed_curve_yards=observed_curve,
    )
    assert measured == pytest.approx(truth, rel=0.02)


def test_calibrate_face_to_path_rejects_zero_face_to_path(integ):
    with pytest.raises(ValueError, match="non-zero"):
        calibrate_face_to_path(
            integ, SEVEN_IRON, face_to_path_deg=0.0, observed_curve_yards=10.0
        )
