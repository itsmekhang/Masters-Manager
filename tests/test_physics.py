"""Physics tests.

These are not just smoke tests -- each asserts a behaviour that would be
wrong in a naive implementation, so the suite actually protects the model.
"""
from __future__ import annotations

import math

import pytest

from caddie.physics import (
    AUGUSTA_APRIL_TYPICAL,
    PGA_TOUR_AVERAGES,
    Atmosphere,
    Ball,
    FlightIntegrator,
    LaunchConditions,
    WindField,
)
from caddie.physics.constants import MPH_MS


@pytest.fixture(scope="module")
def integ():
    return FlightIntegrator(
        atmosphere=Atmosphere.from_conditions(temp_c=21.0, altitude_m=0.0),
        dt=0.002,
    )


# --- Atmosphere -----------------------------------------------------------

def test_isa_sea_level_density():
    atm = Atmosphere(temp_c=15.0, pressure_pa=101325.0, relative_humidity=0.0)
    assert atm.density == pytest.approx(1.225, abs=0.002)


def test_humid_air_is_less_dense():
    """Counterintuitive but correct: water vapour is lighter than air."""
    dry = Atmosphere(temp_c=25.0, relative_humidity=0.0)
    humid = Atmosphere(temp_c=25.0, relative_humidity=1.0)
    assert humid.density < dry.density


def test_cold_air_is_denser():
    assert (
        Atmosphere(temp_c=5.0).density > Atmosphere(temp_c=30.0).density
    )


def test_altitude_reduces_density():
    sea = Atmosphere.from_conditions(temp_c=15.0, altitude_m=0.0)
    denver = Atmosphere.from_conditions(temp_c=15.0, altitude_m=1609.0)
    assert denver.density < sea.density
    # Denver is famously ~17-18% thinner air at standard temperature.
    assert 0.80 < denver.density / sea.density < 0.86


# --- Calibration ----------------------------------------------------------

@pytest.mark.parametrize("club", PGA_TOUR_AVERAGES, ids=lambda c: c.name)
def test_carry_matches_tour_reference(integ, club):
    """The fitted model must reproduce every calibration club to within 6 yd."""
    traj = integ.integrate(club.launch())
    assert traj.carry_yards == pytest.approx(club.published_carry_yd, abs=6.0)


def test_monotonic_carry_across_the_bag(integ):
    carries = [integ.integrate(c.launch()).carry_yards for c in PGA_TOUR_AVERAGES]
    assert carries == sorted(carries, reverse=True), carries


# --- Force model sanity ---------------------------------------------------

def test_backspin_extends_carry(integ):
    """Lift is real: strip the spin and the ball falls out of the sky."""
    spun = integ.integrate(LaunchConditions(167.0, 10.9, 2686))
    dead = integ.integrate(LaunchConditions(167.0, 10.9, 0.0))
    assert spun.carry_yards > dead.carry_yards + 40


@pytest.mark.parametrize("azimuth", [-8.0, -3.0, 0.0, 3.0, 8.0])
def test_pure_backspin_never_curves_whatever_the_start_line(integ, azimuth):
    """The spin axis is referenced to the flight direction, not the target line.

    Regression test. Before the axis was rotated by the launch azimuth, a shot
    started off-line with "pure backspin" had an axis that was not square to its
    own flight, so the Magnus force acted in the wrong vertical plane and the
    ground track bent ~1.5 yd on a 7-iron. It also made carry depend on the
    start line, which is plainly wrong -- a push does not shorten a shot.
    """
    traj = integ.integrate(LaunchConditions(120.0, 16.3, 7097, azimuth_deg=azimuth))
    dx = traj.positions[-1, 0] - traj.positions[0, 0]
    dy = traj.positions[-1, 1] - traj.positions[0, 1]
    bend = dy - dx * math.tan(math.radians(azimuth))
    assert bend == pytest.approx(0.0, abs=0.02)

    straight = integ.integrate(LaunchConditions(120.0, 16.3, 7097))
    assert traj.carry_yards == pytest.approx(straight.carry_yards, abs=0.05)


def test_spin_magnitude_is_invariant_to_start_line():
    """Rotating the axis to follow the launch direction must not create spin."""
    on_line = LaunchConditions(120.0, 16.3, 7097)
    pushed = LaunchConditions(120.0, 16.3, 7097, azimuth_deg=7.0)
    assert pushed.total_spin_rpm == pytest.approx(on_line.total_spin_rpm, rel=1e-12)


def test_excess_spin_costs_distance(integ):
    """Past the optimum, more spin means more drag and a steeper, shorter shot."""
    normal = integ.integrate(LaunchConditions(167.0, 10.9, 2686))
    spinny = integ.integrate(LaunchConditions(167.0, 10.9, 5000))
    assert spinny.carry_yards < normal.carry_yards
    assert spinny.descent_angle_deg > normal.descent_angle_deg


def test_sidespin_curves_the_correct_way(integ):
    """Positive sidespin must move the ball RIGHT, and symmetrically."""
    right = integ.integrate(LaunchConditions(167.0, 10.9, 2686, sidespin_rpm=800))
    left = integ.integrate(LaunchConditions(167.0, 10.9, 2686, sidespin_rpm=-800))
    straight = integ.integrate(LaunchConditions(167.0, 10.9, 2686))
    assert right.offline_yards > straight.offline_yards + 5
    assert left.offline_yards < straight.offline_yards - 5
    assert right.offline_yards == pytest.approx(-left.offline_yards, abs=1.0)


def test_thin_air_flies_further(integ):
    launch = LaunchConditions(167.0, 10.9, 2686)
    sea = FlightIntegrator(
        atmosphere=Atmosphere.from_conditions(15.0, altitude_m=0.0), dt=0.002
    ).integrate(launch)
    mile_high = FlightIntegrator(
        atmosphere=Atmosphere.from_conditions(15.0, altitude_m=1609.0), dt=0.002
    ).integrate(launch)
    assert mile_high.carry_yards > sea.carry_yards + 10


# --- Wind -----------------------------------------------------------------

def test_headwind_shortens_tailwind_lengthens(integ):
    launch = LaunchConditions(120.0, 16.3, 7097)
    calm = integ.integrate(launch).carry_yards
    into = integ.integrate(launch, WindField.from_mph(15, 0)).carry_yards
    down = integ.integrate(launch, WindField.from_mph(15, 180)).carry_yards
    assert into < calm < down


def test_headwind_hurts_more_than_tailwind_helps(integ):
    """A real and much-quoted asymmetry: a headwind adds drag AND lift, which
    balloons the shot, so 15 into costs more than 15 downwind gives back."""
    launch = LaunchConditions(120.0, 16.3, 7097)
    calm = integ.integrate(launch).carry_yards
    into = integ.integrate(launch, WindField.from_mph(15, 0)).carry_yards
    down = integ.integrate(launch, WindField.from_mph(15, 180)).carry_yards
    assert (calm - into) > (down - calm)


def test_crosswind_pushes_downwind(integ):
    """Wind FROM the left (270 deg) must push the ball to the right."""
    launch = LaunchConditions(120.0, 16.3, 7097)
    from_left = integ.integrate(launch, WindField.from_mph(15, 270))
    assert from_left.offline_yards > 5


def test_wind_profile_increases_with_height():
    w = WindField(speed_ref=10.0, roughness_length=0.5)
    assert w.speed_at(2.0) < w.speed_at(10.0) < w.speed_at(40.0)


def test_wind_vanishes_at_the_ground():
    w = WindField(speed_ref=10.0, roughness_length=0.5)
    assert w.speed_at(0.0) == 0.0


def test_shelter_reduces_low_level_wind():
    exposed = WindField(speed_ref=10.0, shelter_factor=1.0)
    sheltered = WindField(speed_ref=10.0, shelter_factor=0.4)
    assert sheltered.speed_at(3.0) < exposed.speed_at(3.0)
    # Above the treeline the two must converge -- shelter is a canopy effect.
    assert sheltered.speed_at(60.0) == pytest.approx(exposed.speed_at(60.0), rel=1e-6)


# --- Terrain --------------------------------------------------------------

def test_downhill_lie_carries_further(integ):
    """Elevated tee: ground drops away downrange, so the ball flies longer.

    Note the terrain must be at or below the launch point at x=0 -- a uniform
    offset would place the tee underground, which the integrator rejects.
    """
    launch = LaunchConditions(120.0, 16.3, 7097)
    flat = integ.integrate(launch).carry_yards
    downhill = integ.integrate(launch, terrain=lambda x, y: -0.12 * x).carry_yards
    assert downhill > flat + 15


def test_uphill_target_carries_less(integ):
    launch = LaunchConditions(120.0, 16.3, 7097)
    flat = integ.integrate(launch).carry_yards
    uphill = integ.integrate(launch, terrain=lambda x, y: 0.10 * x).carry_yards
    assert uphill < flat - 10


def test_launch_below_terrain_is_rejected(integ):
    with pytest.raises(ValueError):
        integ.integrate(LaunchConditions(120.0, 16.3, 7097), terrain=lambda x, y: 5.0)


# --- Numerics -------------------------------------------------------------

def test_integration_is_step_size_converged():
    """dt=0.002 must be indistinguishable from a far finer step."""
    launch = LaunchConditions(167.0, 10.9, 2686)
    coarse = FlightIntegrator(dt=0.002).integrate(launch).carry_yards
    fine = FlightIntegrator(dt=0.0002).integrate(launch).carry_yards
    assert coarse == pytest.approx(fine, abs=0.1)


def test_integration_is_deterministic():
    launch = LaunchConditions(167.0, 10.9, 2686)
    a = FlightIntegrator(dt=0.002).integrate(launch).carry_yards
    b = FlightIntegrator(dt=0.002).integrate(launch).carry_yards
    assert a == b


def test_ball_lands_on_sloping_terrain_accurately(integ):
    """The ground-crossing bisection must land the ball ON the surface, even
    when the surface is moving under it during the final step."""
    def slope(x, y):
        return -0.1 * x

    traj = integ.integrate(LaunchConditions(120.0, 16.3, 7097), terrain=slope)
    x, y, z = traj.impact
    assert z == pytest.approx(slope(x, y), abs=1e-3)


def test_spin_decays_in_flight(integ):
    traj = integ.integrate(LaunchConditions(120.0, 16.3, 7097))
    assert traj.impact_spin_rpm < 7097
    assert traj.impact_spin_rpm > 4000  # but not absurdly


# --- Launch conditions ----------------------------------------------------

def test_spin_axis_matches_equivalent_sidespin():
    """The two ways of specifying curve must agree in total spin."""
    axis = LaunchConditions(167.0, 10.9, 3000, spin_axis_deg=15.0)
    assert axis.total_spin_rpm == pytest.approx(3000, rel=1e-9)


def test_azimuth_starts_the_ball_offline(integ):
    pushed = integ.integrate(LaunchConditions(167.0, 10.9, 2686, azimuth_deg=3.0))
    assert pushed.offline_yards > 10
