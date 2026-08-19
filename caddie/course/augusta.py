"""Load the Augusta National model from the ingested data file."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .model import Course, GeoPoint, Hole

DATA_PATH = Path(__file__).parent / "data" / "augusta_national.json"


def _point(d: dict) -> GeoPoint:
    return GeoPoint(lat=d["lat"], lon=d["lon"], elevation_m=d.get("elevation_m"))


@lru_cache(maxsize=1)
def load_augusta(path: Path | None = None) -> Course:
    """Load Augusta National.

    Regenerate the underlying data with ``python scripts/ingest_augusta.py``.
    """
    p = path or DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found -- run scripts/ingest_augusta.py to build it"
        )
    doc = json.loads(p.read_text(encoding="utf-8"))

    holes = [
        Hole(
            number=h["number"],
            name=h["name"],
            par=h["par"],
            card_yardage=h["card_yardage"],
            tee=_point(h["tee"]),
            pin=_point(h["pin"]),
            aim_points=[_point(a) for a in h["aim_points"]],
        )
        for h in doc["holes"]
    ]

    return Course(
        name=doc["course"],
        holes=holes,
        par=doc["par"],
        card_yardage_total=doc["card_yardage_total"],
        sources=doc.get("sources", {}),
        known_gaps=doc.get("known_gaps", []),
    )


AMEN_CORNER = (11, 12, 13)
"""Holes 11, 12 and 13 -- the low point of the property and the reason wind
modelling at Augusta is genuinely hard."""
