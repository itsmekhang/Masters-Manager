"""Hole-map georeference tests.

The georeference is hand-picked data, so these tests guard the things that go
wrong with hand-picked data: a missing hole, a transposed coordinate, a mirrored
transform, or a pick so far off that the implied scale is nonsense.
"""
from __future__ import annotations

import pytest
from PIL import Image

from caddie.course import load_augusta
from caddie.course.maps import (
    MapProjection,
    enu_offset,
    load_hole_info,
    load_hole_maps,
)


@pytest.fixture(scope="module")
def course():
    return load_augusta()


@pytest.fixture(scope="module")
def maps():
    return load_hole_maps()


@pytest.fixture(scope="module")
def info():
    return load_hole_info()


def projection(hole, hole_map):
    with Image.open(hole_map.image_path) as im:
        w, h = im.size
    return MapProjection(hole_map, w, h, hole.tee, hole.pin), w, h


# --- Coverage --------------------------------------------------------------

def test_every_hole_has_a_map(maps, course):
    missing = [h.number for h in course.holes if h.number not in maps]
    assert not missing, f"holes without a georeference: {missing}"


def test_every_map_image_exists(maps):
    missing = [n for n, hm in maps.items() if not hm.exists()]
    assert not missing, f"holes whose image is missing: {missing}"


def test_normalised_coordinates_are_in_range(maps):
    for n, hm in maps.items():
        for name, (u, v) in (("tee", hm.tee_uv), ("pin", hm.pin_uv)):
            assert 0.0 <= u <= 1.0, f"hole {n} {name} u out of range"
            assert 0.0 <= v <= 1.0, f"hole {n} {name} v out of range"


# --- The transform ---------------------------------------------------------

@pytest.mark.parametrize("number", range(1, 19))
def test_reference_points_map_to_their_picked_pixels(course, maps, number):
    """The fit is exact at both reference points, by construction."""
    hole = course.hole(number)
    hm = maps[number]
    proj, w, h = projection(hole, hm)

    tx, ty = proj.geo_to_pixels(hole.tee)
    assert (tx, ty) == pytest.approx((hm.tee_uv[0] * w, hm.tee_uv[1] * h), abs=1e-6)

    px, py = proj.geo_to_pixels(hole.pin)
    assert (px, py) == pytest.approx((hm.pin_uv[0] * w, hm.pin_uv[1] * h), abs=1e-6)


@pytest.mark.parametrize("number", range(1, 19))
def test_transform_is_orientation_reversing(course, maps, number):
    """ENU is y-up, images are y-down, so the map must include a reflection.

    Fit a plain rotation instead and the whole hole comes out mirrored -- which
    on a symmetrical hole looks almost right and is completely wrong. The
    determinant is the cheap invariant that catches it.
    """
    hole = course.hole(number)
    proj, _, _ = projection(hole, maps[number])

    # Columns of the linear part: images of the unit east and north vectors.
    e_x, e_y = proj.to_pixels(1.0, 0.0)
    n_x, n_y = proj.to_pixels(0.0, 1.0)
    o_x, o_y = proj.to_pixels(0.0, 0.0)
    det = (e_x - o_x) * (n_y - o_y) - (n_x - o_x) * (e_y - o_y)
    assert det < 0.0, f"hole {number}: transform is not orientation-reversing"


@pytest.mark.parametrize("number", range(1, 19))
def test_scale_is_plausible(course, maps, number):
    """A hole corridor is a few hundred metres by one or two hundred."""
    hole = course.hole(number)
    proj, _, _ = projection(hole, maps[number])
    fw, fh = proj.extent_metres()
    assert 100.0 < fw < 700.0, f"hole {number}: implausible map width {fw:.0f} m"
    assert 40.0 < fh < 400.0, f"hole {number}: implausible map height {fh:.0f} m"


@pytest.mark.parametrize("number", range(1, 19))
def test_route_stays_within_the_image(course, maps, number):
    """Tee, aim points and pin should all be on the artwork.

    This uses geometry the two reference points did NOT determine -- the aim
    points are free -- so it is a real check on the pick rather than a tautology.
    """
    hole = course.hole(number)
    hm = maps[number]
    proj, w, h = projection(hole, hm)
    for point in hole.route_points:
        x, y = proj.geo_to_pixels(point)
        assert -0.05 * w <= x <= 1.05 * w, f"hole {number}: route x off image"
        assert -0.05 * h <= y <= 1.05 * h, f"hole {number}: route y off image"


def test_longer_holes_get_a_coarser_scale(course, maps):
    """Sanity check across holes: metres-per-pixel should track hole length.

    The illustrations were drawn independently, so this correlation was not
    imposed by the picks. If a pick were badly wrong its hole would fall off
    the trend, so a monotone-ish relationship is evidence the set is coherent.
    """
    pairs = []
    for hole in course.holes:
        proj, _, _ = projection(hole, maps[hole.number])
        pairs.append((hole.card_yardage, proj.metres_per_pixel))

    par3s = [mpp for yd, mpp in pairs if yd <= 250]
    longs = [mpp for yd, mpp in pairs if yd >= 500]
    assert max(par3s) < min(longs), (
        "short holes should all be drawn at a finer scale than long ones: "
        f"par-3 max {max(par3s):.4f} vs long-hole min {min(longs):.4f}"
    )


def test_enu_offset_is_zero_at_the_origin(course):
    tee = course.hole(1).tee
    assert enu_offset(tee, tee) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_enu_offset_signs(course):
    """East is +x, north is +y. A sign slip here mirrors every overlay."""
    from caddie.course.model import GeoPoint

    tee = course.hole(1).tee
    east = GeoPoint(tee.lat, tee.lon + 0.001)
    north = GeoPoint(tee.lat + 0.001, tee.lon)
    assert enu_offset(tee, east)[0] > 0
    assert enu_offset(tee, east)[1] == pytest.approx(0.0, abs=1e-6)
    assert enu_offset(tee, north)[1] > 0
    assert enu_offset(tee, north)[0] == pytest.approx(0.0, abs=1e-6)


# --- hole info.txt ---------------------------------------------------------

def test_hole_info_covers_all_eighteen(info):
    assert sorted(info) == list(range(1, 19))


def test_hole_info_par_agrees_with_the_course_model(course, info):
    for hole in course.holes:
        assert info[hole.number]["par"] == hole.par, f"par mismatch on {hole.number}"


def test_hole_info_yardage_agrees_except_on_seventeen(course, info):
    """Documents a real source disagreement rather than papering over it.

    Every hole matches the PGA Tour card the course model was built from except
    the 17th, where this file says 450 and the model says 440. Left as a failing
    expectation would hide it; asserted explicitly, it stays visible and this
    test breaks if either source is corrected.
    """
    mismatches = {
        h.number: (h.card_yardage, info[h.number]["yards"])
        for h in course.holes
        if info[h.number]["yards"] != h.card_yardage
    }
    assert mismatches == {17: (440, 450)}


def test_scoring_average_is_consistent_with_par(info):
    """Par 3s should average near 3, par 5s below 5. Catches a shifted table."""
    for hole in info.values():
        avg, par = hole["historicalAverage"], hole["par"]
        if par == 3:
            assert 2.9 < avg < 3.5
        elif par == 4:
            assert 3.9 < avg < 4.5
        else:
            assert 4.5 < avg < 5.1, f"par 5 hole {hole['hole']} averages {avg}"


def test_difficulty_ranks_are_a_permutation(info):
    ranks = sorted(h["historicalRank"] for h in info.values())
    assert ranks == list(range(1, 19))


def test_hardest_hole_is_eleven(info):
    """A named fact worth pinning: White Dogwood ranks 1st, Camellia 2nd."""
    by_rank = {h["historicalRank"]: h["hole"] for h in info.values()}
    assert by_rank[1] == 11
    assert by_rank[2] == 10
