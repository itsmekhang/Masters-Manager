"""Course model tests, including validation of the ingested Augusta data.

The geometry comes from a single third-party source, so these tests are the
guard rail: they check it against independently sourced facts (official
yardages, and Augusta topography that is common knowledge).
"""
from __future__ import annotations

import math

import pytest

from caddie.course import GeoPoint, ShotFrame, load_augusta, terrain_from_hole

OFFICIAL_YARDAGE = [445, 585, 350, 240, 495, 180, 450, 570, 460,
                    495, 520, 155, 545, 440, 550, 170, 440, 465]
OFFICIAL_PAR = [4, 5, 4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4]


@pytest.fixture(scope="module")
def augusta():
    return load_augusta()


# --- Scorecard ------------------------------------------------------------

def test_par_and_total(augusta):
    assert augusta.par == 72
    assert augusta.card_yardage_total == 7555
    assert len(augusta) == 18


def test_nines_add_up(augusta):
    front = sum(h.card_yardage for h in augusta.holes[:9])
    back = sum(h.card_yardage for h in augusta.holes[9:])
    assert (front, back) == (3775, 3780)


@pytest.mark.parametrize("i", range(18))
def test_par_and_yardage_per_hole(augusta, i):
    h = augusta.hole(i + 1)
    assert h.par == OFFICIAL_PAR[i]
    assert h.card_yardage == OFFICIAL_YARDAGE[i]


def test_hole_names(augusta):
    assert augusta.hole(12).name == "Golden Bell"
    assert augusta.hole(13).name == "Azalea"
    assert augusta.hole(10).name == "Camellia"


# --- Geometry validation --------------------------------------------------

@pytest.mark.parametrize("i", range(18))
def test_route_length_reproduces_card_yardage(augusta, i):
    """Independent check on the geometry: the tee -> aim -> pin route must
    reproduce the official card yardage. 20 yd tolerance because card yardage
    follows the architect's playing line, not our polyline."""
    h = augusta.hole(i + 1)
    assert h.route_yards == pytest.approx(h.card_yardage, abs=20.0)


def test_route_beats_straight_line_overall(augusta):
    """Routing through aim points must explain the card better than a straight
    tee-to-pin line. If this fails, the aim points are not real."""
    route_err = sum(abs(h.route_yards - h.card_yardage) for h in augusta)
    straight_err = sum(abs(h.straight_yards - h.card_yardage) for h in augusta)
    assert route_err < straight_err / 1.5


def test_hole_13_is_a_severe_dogleg(augusta):
    """Azalea bends hard left; its straight-line distance is far short of the
    card. This is the single strongest signal that the aim points are genuine."""
    h13 = augusta.hole(13)
    assert h13.card_yardage - h13.straight_yards > 60
    assert h13.dogleg_deg > 20


def test_par_threes_have_no_aim_points(augusta):
    for n in (4, 6, 12, 16):
        assert augusta.hole(n).aim_points == []


def test_par_fives_have_two_aim_points(augusta):
    for n in (2, 8, 13, 15):
        assert len(augusta.hole(n).aim_points) == 2


def test_every_hole_has_par_minus_one_route_points(augusta):
    for h in augusta:
        assert len(h.route_points) == h.par - 1


def test_all_points_lie_on_the_property(augusta):
    """Sanity-check the coordinates actually sit at Augusta National."""
    for h in augusta:
        for p in h.route_points:
            assert 33.490 < p.lat < 33.512, f"hole {h.number} lat {p.lat}"
            assert -82.030 < p.lon < -82.012, f"hole {h.number} lon {p.lon}"


# --- Elevation ------------------------------------------------------------

def test_elevation_present_everywhere(augusta):
    for h in augusta:
        for p in h.route_points:
            assert p.elevation_m is not None


def test_hole_10_is_the_big_drop(augusta):
    """Camellia falls ~100 ft from tee to green -- the largest drop on the
    course. Verified against USGS 3DEP, not asserted from folklore."""
    dz_ft = augusta.hole(10).elevation_change_m / 0.3048
    assert dz_ft < -80

    drops = {h.number: h.elevation_change_m for h in augusta}
    assert min(drops, key=drops.get) == 10


def test_hole_18_climbs(augusta):
    assert augusta.hole(18).elevation_change_m / 0.3048 > 40


def test_amen_corner_is_the_low_point(augusta):
    """Holes 11-13 sit at the bottom of the property."""
    lows = [augusta.hole(n).pin.elevation_m for n in (11, 12, 13)]
    others = [
        h.pin.elevation_m for h in augusta if h.number not in (11, 12, 13)
    ]
    assert max(lows) < sorted(others)[2]


def test_total_relief_matches_reported_figure(augusta):
    """Augusta is reported to have ~175 ft of elevation change."""
    lo, hi = augusta.elevation_range_m()
    relief_ft = (hi - lo) / 0.3048
    assert 120 < relief_ft < 220


# --- Frames ---------------------------------------------------------------

def test_shot_frame_roundtrip(augusta):
    h = augusta.hole(1)
    frame = h.tee_frame()
    back = frame.to_geo(*frame.to_local(h.pin))
    assert back.lat == pytest.approx(h.pin.lat, abs=1e-9)
    assert back.lon == pytest.approx(h.pin.lon, abs=1e-9)


def test_pin_is_downrange_in_tee_frame(augusta):
    """Aiming at the pin must put it on the +x axis with ~zero lateral offset."""
    h = augusta.hole(12)  # par 3, aimed straight at the pin
    frame = h.frame_from(h.tee, h.pin)
    x, y, z = frame.to_local(h.pin)
    assert x > 0
    assert abs(y) < 0.5
    assert x == pytest.approx(h.straight_yards * 0.9144, rel=1e-3)


def test_frame_elevation_matches_hole(augusta):
    h = augusta.hole(10)
    x, y, z = h.frame_from(h.tee, h.pin).to_local(h.pin)
    assert z == pytest.approx(h.elevation_change_m, abs=1e-6)


def test_terrain_callback_interpolates_between_known_points(augusta):
    h = augusta.hole(10)
    frame = h.frame_from(h.tee, h.pin)
    terrain = terrain_from_hole(h, frame)
    # At the tee and pin it must return the measured elevations (relative).
    assert terrain(0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    px, py, pz = frame.to_local(h.pin)
    assert terrain(px, py) == pytest.approx(pz, abs=1e-6)


# --- Provenance -----------------------------------------------------------

def test_sources_and_gaps_are_recorded(augusta):
    """The model must carry its own provenance and admit what it lacks."""
    assert "geometry" in augusta.sources
    assert "elevation" in augusta.sources
    assert augusta.known_gaps
