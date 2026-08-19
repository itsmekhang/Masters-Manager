"""Wind field models.

Wind is the single largest controllable uncertainty in a shot recommendation,
and Augusta is the hardest place on earth to model it: the course is cut
through mature loblolly pines 100+ feet tall, so the ball spends the first
half of its flight in a sheltered, sheared boundary layer and the apex well
above the treeline in near-free-stream flow. That vertical shear is why
players describe the wind at Amen Corner as "swirling" -- it genuinely is
different at ball height than at apex height.

We therefore model wind as a height-dependent vector field, not a constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WindField:
    """Log-law wind profile with optional per-hole shelter and veer.

    Attributes
    ----------
    speed_ref:
        Wind speed, m/s, at ``height_ref``.
    direction_from_deg:
        METEOROLOGICAL convention -- the direction the wind blows FROM, in
        degrees clockwise from the shot's target line. So 0 = straight into
        the player's face (headwind), 180 = directly downwind, 90 = blowing
        from the player's right.
    height_ref:
        Reference height for ``speed_ref``, m. Weather stations use 10 m.
    roughness_length:
        Aerodynamic roughness z0, m. Governs how fast wind builds with
        height. Open grass ~0.03; scattered trees ~0.25; mature closed
        forest canopy ~1.0. Augusta's tree-lined corridors behave like
        0.5-1.0 near the ground.
    displacement_height:
        Zero-plane displacement d, m. For flow over a canopy the profile is
        effectively lifted by ~0.7x canopy height. Set to 0 for open holes.
    shelter_factor:
        Multiplier 0..1 applied below the treeline to represent a corridor
        shielded from the prevailing wind. 1.0 = fully exposed.
    veer_deg_per_100m:
        Directional change with height (Ekman veer plus canopy channelling).
        Small in absolute terms but it is the mechanism behind a shot that
        starts riding a crosswind and lands in a different one.
    """

    speed_ref: float = 0.0
    direction_from_deg: float = 0.0
    height_ref: float = 10.0
    roughness_length: float = 0.25
    displacement_height: float = 0.0
    shelter_factor: float = 1.0
    veer_deg_per_100m: float = 0.0

    # Treeline height, m. Below this the shelter factor applies.
    treeline_height: float = 27.0

    def __post_init__(self) -> None:
        if self.roughness_length <= 0:
            raise ValueError("roughness_length must be > 0")
        if not 0.0 <= self.shelter_factor <= 1.0:
            raise ValueError("shelter_factor must be in [0, 1]")

    def speed_at(self, height: float) -> float:
        """Wind speed at a height above ground, m/s (log law)."""
        z0 = self.roughness_length
        z = max(height - self.displacement_height, 0.0)
        z_ref = max(self.height_ref - self.displacement_height, z0 * 1.01)

        if z <= z0:
            return 0.0  # no-slip: wind vanishes at the roughness scale

        profile = math.log(z / z0) / math.log(z_ref / z0)
        speed = self.speed_ref * profile

        # Shelter ramps smoothly out as the ball climbs past the pines,
        # rather than switching discontinuously (which would put a kink in
        # the trajectory and upset the integrator).
        if self.shelter_factor < 1.0:
            t = min(max(height / max(self.treeline_height, 1e-6), 0.0), 1.0)
            blend = t * t * (3.0 - 2.0 * t)  # smoothstep
            speed *= self.shelter_factor + (1.0 - self.shelter_factor) * blend

        return speed

    def direction_at(self, height: float) -> float:
        return self.direction_from_deg + self.veer_deg_per_100m * (height / 100.0)

    def vector_at(self, height: float) -> tuple[float, float, float]:
        """Wind velocity vector in the shot frame (x downrange, y right, z up).

        A wind blowing FROM straight ahead (0 deg) must produce a velocity
        pointing back at the player, i.e. -x. Hence the negative signs.
        """
        speed = self.speed_at(height)
        if speed == 0.0:
            return (0.0, 0.0, 0.0)
        phi = math.radians(self.direction_at(height))
        return (-speed * math.cos(phi), -speed * math.sin(phi), 0.0)

    @classmethod
    def from_mph(
        cls,
        speed_mph: float,
        direction_from_deg: float = 0.0,
        **kwargs,
    ) -> "WindField":
        from .constants import MPH_MS

        return cls(
            speed_ref=speed_mph * MPH_MS,
            direction_from_deg=direction_from_deg,
            **kwargs,
        )

    def describe(self) -> str:
        from .constants import MPH_MS

        return (
            f"{self.speed_ref / MPH_MS:.0f} mph from {self.direction_from_deg:.0f}deg "
            f"@ {self.height_ref:.0f} m, z0={self.roughness_length:.2f} m, "
            f"shelter={self.shelter_factor:.2f}"
        )


CALM = WindField()

AUGUSTA_SHELTERED_CORRIDOR = dict(roughness_length=0.8, shelter_factor=0.45, treeline_height=30.0)
"""Kwargs for a tight tree-lined hole (e.g. 11, 13 tee shot). The ball only
meets the true wind above the pines."""

AUGUSTA_OPEN = dict(roughness_length=0.15, shelter_factor=0.9, treeline_height=25.0)
"""Kwargs for the exposed parts of the property (e.g. 2nd fairway, 15/16)."""
