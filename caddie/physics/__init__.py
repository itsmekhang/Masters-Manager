"""Ball flight physics."""
from .aero import SMITS_SMITH_1994, AeroModel
from .atmosphere import AUGUSTA_APRIL_TYPICAL, Atmosphere
from .ball import (
    BAG_LIMIT,
    EXTRA_VARIANTS,
    PGA_TOUR_AVERAGES,
    WEDGES,
    Ball,
    ClubReference,
    LaunchConditions,
)
from .trajectory import FlightIntegrator, Trajectory, flat_terrain
from .wind import CALM, WindField

__all__ = [
    "AeroModel",
    "SMITS_SMITH_1994",
    "Atmosphere",
    "AUGUSTA_APRIL_TYPICAL",
    "Ball",
    "ClubReference",
    "LaunchConditions",
    "PGA_TOUR_AVERAGES",
    "WEDGES",
    "EXTRA_VARIANTS",
    "BAG_LIMIT",
    "FlightIntegrator",
    "Trajectory",
    "flat_terrain",
    "WindField",
    "CALM",
]
