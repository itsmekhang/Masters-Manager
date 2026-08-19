"""Six-degree-of-freedom-lite ball flight integration.

Equations of motion
-------------------
The ball is a sphere with backspin, so it feels three forces:

    m dv/dt = -m g z_hat  +  F_drag  +  F_magnus

with the aerodynamic forces evaluated on the velocity of the ball *relative
to the air*, ``u = v - v_wind`` -- this is the whole reason wind matters:

    F_drag   = -1/2 rho A C_D |u| u
    F_magnus = +1/2 rho A C_L |u|^2 * (omega x u) / |omega x u|

The Magnus direction ``omega_hat x u_hat`` is what makes backspin produce
lift and sidespin produce curve, from a single expression. Spin magnitude
decays exponentially in flight; the spin *axis* is held fixed, which matches
launch-monitor observation closely over a single shot.

Integration is fixed-step RK4. At the speeds and timescales of a golf shot
(flight ~4-7 s, forces smooth) RK4 at dt = 2 ms is accurate to well under a
tenth of a yard while staying fast enough to run tens of thousands of
Monte-Carlo trajectories for a shot recommendation, and -- unlike an
adaptive solver -- it is bit-for-bit reproducible, which matters when you
are diffing two candidate clubs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .aero import AeroModel
from .atmosphere import Atmosphere
from .ball import Ball, LaunchConditions
from .constants import G, YARD_M
from .wind import CALM, WindField

# A terrain callback maps (x, y) -> ground height z. Default: flat.
TerrainFn = Callable[[float, float], float]


def flat_terrain(x: float, y: float) -> float:
    return 0.0


@dataclass
class Trajectory:
    """Result of a flight integration. Distances in metres internally."""

    times: np.ndarray          # (n,)
    positions: np.ndarray      # (n, 3) x downrange, y right, z up
    velocities: np.ndarray     # (n, 3)
    spins_rpm: np.ndarray      # (n,) total spin magnitude
    launch: LaunchConditions
    landed: bool = True
    # Diagnostics
    aero_samples: list[tuple[float, float, float]] = field(default_factory=list)

    # --- Primary outputs ---------------------------------------------------
    @property
    def impact(self) -> np.ndarray:
        """Landing position (x, y, z)."""
        return self.positions[-1]

    @property
    def carry(self) -> float:
        """Carry distance along the ground from launch to first impact, m.

        Measured as straight-line horizontal distance, matching launch
        monitor convention (which reports carry along the target line as
        'carry' and lateral miss separately).
        """
        dx = self.positions[-1, 0] - self.positions[0, 0]
        dy = self.positions[-1, 1] - self.positions[0, 1]
        return math.hypot(dx, dy)

    @property
    def carry_yards(self) -> float:
        return self.carry / YARD_M

    @property
    def downrange(self) -> float:
        """Distance along the target line only (x), m."""
        return self.positions[-1, 0] - self.positions[0, 0]

    @property
    def downrange_yards(self) -> float:
        return self.downrange / YARD_M

    @property
    def offline(self) -> float:
        """Lateral deviation at impact, m. Positive = right of target."""
        return self.positions[-1, 1] - self.positions[0, 1]

    @property
    def offline_yards(self) -> float:
        return self.offline / YARD_M

    @property
    def apex(self) -> float:
        """Peak height above launch, m."""
        return float(self.positions[:, 2].max() - self.positions[0, 2])

    @property
    def apex_yards(self) -> float:
        return self.apex / YARD_M

    @property
    def flight_time(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def descent_angle_deg(self) -> float:
        """Angle below horizontal at impact. Drives how much the ball stops."""
        vx, vy, vz = self.velocities[-1]
        horizontal = math.hypot(vx, vy)
        if horizontal < 1e-9:
            return 90.0
        return math.degrees(math.atan2(-vz, horizontal))

    @property
    def impact_speed(self) -> float:
        return float(np.linalg.norm(self.velocities[-1]))

    @property
    def impact_spin_rpm(self) -> float:
        return float(self.spins_rpm[-1])

    def summary(self) -> str:
        return (
            f"carry {self.carry_yards:6.1f} yd | offline {self.offline_yards:+6.1f} yd | "
            f"apex {self.apex_yards:5.1f} yd | descent {self.descent_angle_deg:4.1f}deg | "
            f"time {self.flight_time:4.2f} s | impact {self.impact_speed:5.1f} m/s, "
            f"{self.impact_spin_rpm:5.0f} rpm"
        )


class FlightIntegrator:
    """Integrates ball flight under gravity, drag, Magnus lift and wind."""

    def __init__(
        self,
        atmosphere: Atmosphere | None = None,
        aero: AeroModel | None = None,
        ball: Ball | None = None,
        dt: float = 0.002,
        max_time: float = 20.0,
    ) -> None:
        self.atmosphere = atmosphere or Atmosphere()
        self.aero = aero or AeroModel()
        self.ball = ball or Ball()
        self.dt = dt
        self.max_time = max_time

    # --- Force model -------------------------------------------------------
    def acceleration(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        spin_vec: np.ndarray,
        wind: WindField,
    ) -> np.ndarray:
        rho = self.atmosphere.density
        nu = self.atmosphere.kinematic_viscosity
        area = self.ball.area
        mass = self.ball.mass
        radius = self.ball.radius

        # Air-relative velocity is what the ball's boundary layer sees.
        w = np.array(wind.vector_at(float(position[2])))
        u = velocity - w
        u_mag = float(np.linalg.norm(u))

        accel = np.array([0.0, 0.0, -G])
        if u_mag < 1e-6:
            return accel

        spin_mag = float(np.linalg.norm(spin_vec))
        s = self.aero.spin_ratio(u_mag, spin_mag, radius)
        re = self.aero.reynolds(u_mag, nu)
        cd = self.aero.drag_coefficient(s, re)
        cl = self.aero.lift_coefficient(s)

        q = 0.5 * rho * area  # dynamic-pressure prefactor without the V^2

        # Drag opposes air-relative motion.
        accel += (-q * cd * u_mag / mass) * u

        # Magnus force along omega x u.
        if spin_mag > 1e-9:
            lift_dir = np.cross(spin_vec, u)
            lift_norm = float(np.linalg.norm(lift_dir))
            if lift_norm > 1e-9:
                accel += (q * cl * u_mag**2 / mass / lift_norm) * lift_dir

        return accel

    # --- Integration -------------------------------------------------------
    def integrate(
        self,
        launch: LaunchConditions,
        wind: WindField | None = None,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        terrain: TerrainFn = flat_terrain,
        record_every: int = 5,
    ) -> Trajectory:
        """Fly the ball until it meets the ground.

        Parameters
        ----------
        origin:
            Launch point. Set ``z`` to the tee height above the local datum.
        terrain:
            Callable ``(x, y) -> ground height``. This is how elevated tees
            and downhill greens change carry: the ball simply flies until it
            hits the surface, which is the physically correct treatment
            rather than a "plays like" yardage adjustment.
        record_every:
            Store every Nth step. The integration always uses ``dt``.
        """
        wind = wind or CALM

        pos = np.array(origin, dtype=float)
        vel = np.array(launch.velocity_vector(), dtype=float)
        spin0 = np.array(launch.spin_vector(), dtype=float)
        tau = self.ball.spin_decay_tau

        # Guard: launching from below ground is a caller error worth catching.
        # Tolerance is 1 mm, not 1e-6 m: an origin round-tripped through
        # lat/lon <-> local-metres conversion (e.g. a previous shot's landing
        # spot fed back in as the next shot's origin) picks up floating-point
        # residue at the micrometre scale, which used to be enough to trip a
        # 1e-6 m guard on a shot that is, physically, sitting exactly on the
        # ground.
        if pos[2] < terrain(pos[0], pos[1]) - 1e-3:
            raise ValueError("launch origin is below the terrain surface")

        times = [0.0]
        positions = [pos.copy()]
        velocities = [vel.copy()]
        spins = [float(np.linalg.norm(spin0))]

        def spin_at(t: float) -> np.ndarray:
            return spin0 * math.exp(-t / tau)

        def deriv(t: float, state: np.ndarray) -> np.ndarray:
            p, v = state[:3], state[3:]
            a = self.acceleration(p, v, spin_at(t), wind)
            return np.concatenate((v, a))

        state = np.concatenate((pos, vel))
        t = 0.0
        dt = self.dt
        step = 0
        landed = False

        while t < self.max_time:
            prev_state = state.copy()
            prev_t = t

            # Classic RK4.
            k1 = deriv(t, state)
            k2 = deriv(t + dt / 2, state + dt / 2 * k1)
            k3 = deriv(t + dt / 2, state + dt / 2 * k2)
            k4 = deriv(t + dt, state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            step += 1

            ground_now = terrain(float(state[0]), float(state[1]))
            # Only test for landing on the way down, so a shot launched from
            # a tee below the fairway grade is not "landed" at t=0.
            if state[2] <= ground_now and state[5] < 0.0:
                frac = _ground_crossing_fraction(prev_state, state, terrain)
                state = prev_state + frac * (state - prev_state)
                t = prev_t + frac * dt
                landed = True

            if landed or step % record_every == 0:
                times.append(t)
                positions.append(state[:3].copy())
                velocities.append(state[3:].copy())
                spins.append(float(np.linalg.norm(spin_at(t))))

            if landed:
                break

        from .constants import RPM_RADS

        return Trajectory(
            times=np.array(times),
            positions=np.array(positions),
            velocities=np.array(velocities),
            spins_rpm=np.array(spins) / RPM_RADS,
            launch=launch,
            landed=landed,
        )


def _ground_crossing_fraction(
    before: np.ndarray, after: np.ndarray, terrain: TerrainFn, iterations: int = 24
) -> float:
    """Bisect for the fraction of the last step at which z == terrain(x, y).

    Linear interpolation of the height *excess* is enough for flat ground,
    but on a sloping green the terrain moves under the ball during the step,
    so we bisect the signed clearance instead. 24 iterations resolves the
    crossing to ~1e-7 of a 2 ms step: sub-millimetre.
    """

    def clearance(frac: float) -> float:
        s = before + frac * (after - before)
        return float(s[2] - terrain(float(s[0]), float(s[1])))

    lo, hi = 0.0, 1.0
    c_lo = clearance(lo)
    if c_lo <= 0.0:
        return 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if clearance(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
