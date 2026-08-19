"""Air state: density and viscosity from the weather you can actually read
off a forecast (temperature, pressure, humidity) plus altitude.

Density is what turns "170 yards" into "163 yards" on a cold morning, so it
gets a proper moist-air treatment rather than a fudge factor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (
    G,
    LAPSE_RATE,
    MU_REF,
    P0_PA,
    R_DRY_AIR,
    R_WATER_VAPOUR,
    SUTHERLAND_S,
    T0_K,
    T_REF_VISC,
)


def saturation_vapour_pressure(temp_c: float) -> float:
    """Saturation vapour pressure of water in Pa.

    Buck (1981) equation, accurate to ~0.05% over -30..+50 degC, which is
    far better than we need but costs nothing.
    """
    return 611.21 * math.exp(
        (18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c))
    )


def isa_pressure(altitude_m: float) -> float:
    """ICAO Standard Atmosphere pressure at geopotential altitude (Pa)."""
    exponent = G / (R_DRY_AIR * LAPSE_RATE)
    return P0_PA * (1.0 - LAPSE_RATE * altitude_m / T0_K) ** exponent


@dataclass(frozen=True)
class Atmosphere:
    """Immutable air state at the point of play.

    Attributes
    ----------
    temp_c:
        Ambient dry-bulb temperature, degrees Celsius.
    pressure_pa:
        ABSOLUTE station pressure. Note that weather reports and aviation
        METARs give *sea-level-corrected* pressure (altimeter setting); use
        :meth:`from_conditions` with ``altimeter_hpa`` to convert.
    relative_humidity:
        0..1. Moist air is *less* dense than dry air at the same pressure
        (water, 18 g/mol, is lighter than air's 29 g/mol), so high humidity
        makes the ball fly marginally farther -- the opposite of the
        clubhouse folklore.
    altitude_m:
        Elevation above mean sea level. Only used for bookkeeping and for
        pressure conversion; density is computed from pressure directly.
    """

    temp_c: float = 15.0
    pressure_pa: float = P0_PA
    relative_humidity: float = 0.5
    altitude_m: float = 0.0

    @classmethod
    def from_conditions(
        cls,
        temp_c: float,
        altitude_m: float = 0.0,
        relative_humidity: float = 0.5,
        altimeter_hpa: float | None = None,
        station_pressure_hpa: float | None = None,
    ) -> "Atmosphere":
        """Build an Atmosphere from the numbers a golfer can look up.

        Provide at most one pressure input:
          * ``station_pressure_hpa`` -- true local absolute pressure.
          * ``altimeter_hpa``        -- sea-level-reduced pressure, i.e. the
            number in a weather app or METAR ``Q1013``.
          * neither                  -- ISA pressure for ``altitude_m``.
        """
        if station_pressure_hpa is not None and altimeter_hpa is not None:
            raise ValueError("give station_pressure_hpa or altimeter_hpa, not both")

        if station_pressure_hpa is not None:
            pressure = station_pressure_hpa * 100.0
        elif altimeter_hpa is not None:
            # Reduce the sea-level value back down to the station using the
            # same ISA relation the altimeter setting is defined with.
            pressure = altimeter_hpa * 100.0 * (isa_pressure(altitude_m) / P0_PA)
        else:
            pressure = isa_pressure(altitude_m)

        return cls(
            temp_c=temp_c,
            pressure_pa=pressure,
            relative_humidity=relative_humidity,
            altitude_m=altitude_m,
        )

    @property
    def temp_k(self) -> float:
        return self.temp_c + 273.15

    @property
    def vapour_pressure_pa(self) -> float:
        return self.relative_humidity * saturation_vapour_pressure(self.temp_c)

    @property
    def density(self) -> float:
        """Moist-air density, kg/m^3.

        Dalton's law: treat the air as dry air at partial pressure (p - e)
        plus water vapour at partial pressure e.
        """
        e = self.vapour_pressure_pa
        p_dry = self.pressure_pa - e
        return p_dry / (R_DRY_AIR * self.temp_k) + e / (R_WATER_VAPOUR * self.temp_k)

    @property
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity, Pa*s, via Sutherland's law."""
        t = self.temp_k
        return (
            MU_REF
            * (t / T_REF_VISC) ** 1.5
            * (T_REF_VISC + SUTHERLAND_S)
            / (t + SUTHERLAND_S)
        )

    @property
    def kinematic_viscosity(self) -> float:
        return self.dynamic_viscosity / self.density

    @property
    def density_ratio(self) -> float:
        """Density relative to ISA sea level. Handy sanity number: carry
        scales roughly as ``density_ratio ** -0.5`` for a full shot."""
        from .constants import RHO0

        return self.density / RHO0

    def describe(self) -> str:
        return (
            f"{self.temp_c:.0f}degC, {self.pressure_pa / 100:.0f} hPa station, "
            f"{self.relative_humidity * 100:.0f}% RH, {self.altitude_m:.0f} m MSL "
            f"-> rho={self.density:.4f} kg/m^3 ({self.density_ratio * 100:.1f}% of ISA)"
        )


# Convenience: a typical second-week-of-April Augusta afternoon.
# Augusta, GA sits ~40 m (130 ft) above sea level; the course itself spans
# roughly 175 ft of relief, handled per-hole in the course model.
AUGUSTA_APRIL_TYPICAL = Atmosphere.from_conditions(
    temp_c=22.0, altitude_m=45.0, relative_humidity=0.55
)
