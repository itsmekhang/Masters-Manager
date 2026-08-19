"""Physical constants and USGA/R&A ball specifications.

All values SI unless the name says otherwise. Sources noted inline so the
numbers can be audited rather than trusted.
"""

# --- Fundamental ---------------------------------------------------------
G = 9.80665                 # m/s^2, standard gravity (ISO 80000-3)
R_DRY_AIR = 287.0528        # J/(kg*K), specific gas constant, dry air
R_WATER_VAPOUR = 461.495    # J/(kg*K), specific gas constant, water vapour

# --- ISA sea-level reference (ICAO Standard Atmosphere) ------------------
T0_K = 288.15               # K   (15 degC)
P0_PA = 101325.0            # Pa
LAPSE_RATE = 0.0065         # K/m, troposphere temperature lapse
RHO0 = 1.225                # kg/m^3, ISA sea-level density

# Sutherland's law constants for air viscosity
MU_REF = 1.716e-5           # Pa*s at T_REF
T_REF_VISC = 273.15         # K
SUTHERLAND_S = 110.4        # K

# --- Golf ball, conforming limits (USGA/R&A Rules of Golf, Equipment) ----
# The rules set a MAXIMUM mass and a MINIMUM diameter. Tour balls sit
# essentially at both limits, so these double as realistic values.
BALL_MASS_MAX_KG = 0.04593      # 1.620 oz
BALL_DIAMETER_MIN_M = 0.042672  # 1.680 in
BALL_RADIUS_M = BALL_DIAMETER_MIN_M / 2.0

# Moment of inertia. A uniform sphere gives (2/5) m r^2 = 8.36e-6 kg*m^2.
# Real multi-layer balls are slightly denser toward the mantle; measured
# values cluster around 8.4-8.8e-6. We default to the uniform-sphere value
# and expose it as a parameter (see ball.py) because spin decay is the only
# thing sensitive to it.
BALL_INERTIA_UNIFORM = 0.4 * BALL_MASS_MAX_KG * BALL_RADIUS_M**2

# --- Unit conversions ----------------------------------------------------
YARD_M = 0.9144
FOOT_M = 0.3048
INCH_M = 0.0254
MPH_MS = 0.44704
KNOT_MS = 0.514444
RPM_RADS = 2.0 * 3.141592653589793 / 60.0
