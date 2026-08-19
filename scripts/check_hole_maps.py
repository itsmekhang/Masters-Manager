"""Audit the hole-map georeferences.

The two reference pixels per hole are placed by eye, so they need checking. The
transform is *fitted* to those two points, which means the tee and pin land
correctly by construction -- their agreement proves nothing. Two things do:

1. Scale plausibility. Metres-per-pixel implies an image footprint in metres.
   A golf hole corridor is a few hundred metres long and one to two hundred
   wide; anything far outside that means a mis-pick.
2. The route. Drawing tee -> aim points -> pin on the artwork uses geometry the
   two reference points did NOT determine, so if the aim points track the
   fairway the georeference is good, and if they wander into the trees it isn't.

Run:  python scripts/check_hole_maps.py
Writes a contact sheet to artifacts/hole_maps_check.png and prints the table.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from caddie.course import load_augusta
from caddie.course.maps import MapProjection, load_hole_info, load_hole_maps

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "hole_maps_check.png"


def main() -> None:
    course = load_augusta()
    maps = load_hole_maps()
    info = load_hole_info()

    print(f"{'hole':>4} {'m/px':>7} {'footprint m':>14} {'tee-pin m':>10} "
          f"{'route in frame':>15} {'card':>6} {'info':>6} {'avg':>6} {'rank':>5}")
    print("-" * 88)

    fig, axes = plt.subplots(6, 3, figsize=(16, 20), facecolor="#fcfcfb")
    problems: list[str] = []

    for ax, hole in zip(axes.ravel(), course.holes):
        hm = maps.get(hole.number)
        if hm is None or not hm.exists():
            ax.set_axis_off()
            ax.set_title(f"#{hole.number} — no map", fontsize=9)
            continue

        im = Image.open(hm.image_path)
        W, H = im.size
        proj = MapProjection(hm, W, H, hole.tee, hole.pin)

        ax.imshow(np.asarray(im))
        ax.set_axis_off()

        # The route: geometry the two reference points did not fix.
        pts = [proj.geo_to_pixels(p) for p in hole.route_points]
        px, py = zip(*pts)
        ax.plot(px, py, "-", color="#2a78d6", lw=2.0, zorder=3)
        ax.plot(px[1:-1], py[1:-1], "o", ms=7, color="#eb6834",
                mec="white", mew=1.5, zorder=4)
        ax.plot([px[0]], [py[0]], "s", ms=9, color="white",
                mec="#0b0b0b", mew=1.5, zorder=5)
        ax.plot([px[-1]], [py[-1]], "*", ms=15, color="white",
                mec="#0b0b0b", mew=1.5, zorder=5)

        inside = all(-0.05 * W <= x <= 1.05 * W and -0.05 * H <= y <= 1.05 * H
                     for x, y in pts)
        mpp = proj.metres_per_pixel
        fw, fh = proj.extent_metres()
        card = hole.card_yardage
        inf = info.get(hole.number, {})
        info_yards = inf.get("yards")

        if not inside:
            problems.append(f"hole {hole.number}: route leaves the image")
        if not (100.0 <= fw <= 700.0 and 40.0 <= fh <= 400.0):
            problems.append(
                f"hole {hole.number}: implausible footprint {fw:.0f}x{fh:.0f} m"
            )
        if info_yards is not None and info_yards != card:
            problems.append(
                f"hole {hole.number}: card yardage {card} but hole info says "
                f"{info_yards}"
            )

        print(
            f"{hole.number:>4} {mpp:7.4f} {f'{fw:.0f} x {fh:.0f}':>14} "
            f"{hole.tee.distance_to(hole.pin):10.1f} {str(inside):>15} "
            f"{card:>6} {str(info_yards):>6} "
            f"{inf.get('historicalAverage', float('nan')):6.3f} "
            f"{inf.get('historicalRank', 0):>5}"
        )

        ax.set_title(
            f"#{hole.number} {hole.name} · par {hole.par} · {card} yd · "
            f"{mpp:.3f} m/px",
            fontsize=9, color="#0b0b0b",
        )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=62, facecolor="#fcfcfb")
    print()
    print(f"contact sheet -> {OUT}")

    if problems:
        print()
        print("FLAGGED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("no automatic checks flagged")


if __name__ == "__main__":
    main()
