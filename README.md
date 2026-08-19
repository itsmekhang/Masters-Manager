# AI Caddie — Physics + Augusta National model

Foundation layer for a shot-recommendation engine: a validated ball-flight
physics model and a validated geometric model of Augusta National.

Two pieces are done. The recommender itself is not built yet — see
[Next](#next-the-recommender).

```bash
pip install -r requirements.txt
python -m pytest tests -q               # 215 tests
python scripts/demo.py                  # end-to-end on holes 12, 10 and 13
python scripts/check_hole_maps.py       # audit the hole-map georeferences
streamlit run scripts/explorer.py       # interactive: hole x club x shape x wind x air
```

`scripts/explorer.py` is a front end for the two layers below — pick a hole, a
shot origin, a club, wind and air, and see the integrated flight against real
terrain in side and plan view, with a calm-air overlay to isolate what the wind
did. It reports the shot you asked for; it does **not** choose a club, for the
reasons in [Next](#next-the-recommender).

## 1. Physics

`caddie/physics/` integrates the real equations of motion rather than applying
"plays like" adjustments:

```
m dv/dt = -m g ẑ  -  ½ρA·C_D|u|u  +  ½ρA·C_L|u|²·(ω×u)/|ω×u|
```

where `u = v - v_wind` is the air-relative velocity — the reason wind matters
at all. Fixed-step RK4 at dt = 2 ms, which is step-size converged to 0.1 yd
and bit-for-bit reproducible (important when diffing two candidate clubs, and
required for Monte-Carlo dispersion later).

| module | what it does |
|---|---|
| `atmosphere.py` | Moist-air density from temp/pressure/humidity/altitude; Buck vapour pressure, Sutherland viscosity |
| `aero.py` | `C_D`, `C_L` as functions of spin ratio `S = ωr/V` and Reynolds number |
| `wind.py` | Log-law wind profile with canopy shelter — wind at ball height ≠ wind at apex |
| `trajectory.py` | RK4 integrator, arbitrary terrain callback, bisected ground crossing |
| `ball.py` | Ball spec, launch conditions, spin-axis handling |

### Calibration — the aerodynamics are fitted, not assumed

Published golf-ball drag/lift fits disagree by 10–15%, which is ~12 yards on a
driver. So `aero.py` takes its functional *form* from the literature
(Smits & Smith 1994) and fits six free coefficients to measured
launch-condition → outcome data (`scripts/calibrate.py`, 6 parameters against
15 measurements):

Evaluated against the full 12-club tour table (36 measurements):

| | worst carry error |
|---|---|
| Smits & Smith 1994 published coefficients | 12.7 yd |
| **fitted (shipped)** | **within 6 yd on every club** |
| refit on all 36 measurements | 4.8 yd carry, but apex +8 to +11 yd on the woods |

Spin decay is bounded to the physically observed 3–7 %/s. Left free, the
optimiser drives it to ~60 s (1.7 %/s) and buys carry accuracy with an
unphysical spin history — a good fit for the wrong reason.

### Known limitation: the lift coefficient is too high mid-bag

The fit reproduces carry well but runs high on *apex*. The cause is now
identified rather than guessed, by checking C_L against an independent
tabulated dataset (`aero.EMPIRICAL_CL_TABLE`, from MacDonald & Hanzely 1991 via
`cagrell/golfmodel`) — a source that constrains lift *directly*, where our
calibration only ever sees it through carry and apex:

| spin ratio | table | ours | Smits & Smith | club |
|---|---|---|---|---|
| 0.08 | 0.140 | 0.150 | 0.138 | Driver |
| 0.14 | 0.188 | **0.254** | 0.215 | 5-wood |
| 0.20 | 0.230 | **0.340** | 0.268 | 5-iron |
| 0.30 | 0.280 | 0.340 | 0.304 | 7-iron |
| 0.46 | 0.330 | 0.340 | 0.228 | PW |

Our C_L runs +0.047 high on average and +0.113 at worst, concentrated in
S = 0.14–0.30. This is a **degeneracy, not a coincidence**: fitting six
coefficients to carry/apex/descent lets the optimiser raise lift and drag
together, preserving carry while flying the ball higher. Smits & Smith has the
lift about right and the carry wrong, so *its* error is in the drag; ours has
the carry right and the lift wrong.

Consequences: descent angle (and so predicted stopping power) is less certain
than carry, and predicted *curvature* inherits the same error, since sideways
force is `C_L·sin(tilt)` — worst through the middle of the bag, close for
driver and wedges.

**The indicated fix, not yet done:** constrain C_L to the table and calibrate
the drag alone. Fewer free parameters against the same 36 measurements, with
the one quantity the Trackman table cannot pin down supplied from outside
instead of inferred.

**Recalibrate to yourself.** Replace `PGA_TOUR_AVERAGES` in `ball.py` with your
own launch monitor session and re-run `scripts/calibrate.py`. This matters more
than any published average.

## 2. Augusta National

`caddie/course/`, data in `caddie/course/data/augusta_national.json`, rebuilt by
`scripts/ingest_augusta.py`.

| datum | source | confidence |
|---|---|---|
| Par, card yardage | PGA Tour API (`pga_course_stats("R2025014")`) | authoritative; matches official Masters scorecard exactly |
| Tee / aim point / pin lat-lon, all 18 holes | provisualizer.com `3dlink.php?id=1` | single third-party source, **independently validated below** |
| Point elevations | USGS 3DEP via `epqs.nationalmap.gov` | authoritative bare-earth DEM (~1–3 m vertical) |

### Validation

The geometry comes from one third-party source, so it is checked against
independent facts:

**Yardage.** Route length through the aim points reproduces the official card
to **5.7 yd mean error** (max 15.8). Straight tee-to-pin is much worse
(12.4 mean, 88 yd max on hole 13) — correct, because 13 doglegs hard left. The
aim points carry real information.

**Topography.** USGS elevation independently reproduces Augusta's known relief
without being told any of it:

| hole | measured | known for |
|---|---|---|
| 10 Camellia | **−101 ft** | biggest drop on the course |
| 2 Pink Dogwood | −89 ft | plunging tee shot |
| 11 White Dogwood | −63 ft | downhill into Amen Corner |
| 8 / 18 | **+62 / +60 ft** | the two climbs |

Amen Corner (11/12/13) sits at the property's low point; the 1st is highest.
Total relief **164 ft**, against Augusta's reported ~175 ft.

Hole 13 also comes out as by far the most severe dogleg (83°), as it should.

### Known gaps — read before trusting this

- **No green contours, bunker polygons, fairway boundaries or tree positions.**
  Only point geometry exists. `terrain_from_hole()` inverse-distance-interpolates
  between known points: it captures the 100 ft drop on 10, but it will not tell
  you which way a putt breaks.
- **No shot-tracking data for Augusta exists in the PGA Tour API.** ShotLink is
  not deployed at the Masters (the club runs the event, not the Tour).
  `pga_shot_details("R2025014", ...)` returns empty for every player and round,
  while the same call at TPC Sawgrass returns 67 shots × 35 columns with
  coordinates. Dispersion and strokes-gained baselines must be built from other
  venues and transferred.
- **The pin is one representative point**, not the four daily Masters placements.
- Wind shelter factors per hole are hand-set, not measured.

## 3. Shot shaping

`caddie/shot.py` is the layer above the launch-monitor parameter set: it turns an
*intent* — "a 20-yard draw that finishes on the target line" — into the six
numbers, and it is explicit about which of its three layers you can trust.

**Layer 1, curvature and aim — validated physics, nothing invented.** Curving a
ball is a tilted spin axis, which the integrator already does exactly. Because
`spin_ratio` uses spin *magnitude*, tilting leaves C_D and C_L alone and only
rotates the Magnus force: vertical lift scales as cos(tilt), lateral as
sin(tilt). So the carry cost of shaping is not a fudge factor — it falls out of
the calibrated equations. Two bisection solvers invert the model: one for the
spin axis that produces a requested curve, one for the start line that brings it
back to the target. They alternate, because curve and aim are weakly coupled.

Hole 13, Azalea — an 83° dogleg left, so the shot is a draw around the corner:

| shot | aim | spin axis | curve | carry | cost | finish |
|---|---|---|---|---|---|---|
| straight | −0.0° | +0.0° | +0.0 yd | 277.0 | +0.0 | −0.0 |
| 10 yd draw | +2.1° | −3.8° | −10.0 yd | 276.8 | −0.2 | −0.0 |
| 20 yd draw | +4.2° | −7.6° | −20.1 yd | 276.2 | −0.8 | −0.0 |
| 30 yd draw | +6.2° | −11.4° | −30.0 yd | 275.1 | −1.9 | −0.0 |

Curve is measured from the *start line*, so it includes wind drift — which means
asking for zero curve in a crosswind asks the solver to hold the ball against
the wind, and it does.

**Layer 2, trajectory height — hand-set, not measured.** `FLIGHT_HEIGHTS` gives
knockdown / three-quarter / stock / flighted as deltas on speed, launch and
spin. The signs are uncontroversial; the magnitudes are estimates and are not in
any tour-average table. Treat them like the per-hole wind shelter factors. That
said, the *mechanism* is real: into 15 mph on 12 the knockdown carries slightly
**further** than the stock shot despite less speed and spin, because it doesn't
balloon (apex 25.3 yd vs 35.1). Nothing encodes that — it comes out of the
integration.

**Layer 3, face and path (the D-plane) — unvalidated.** `DPlaneModel` maps club
face and club path onto a start line and a spin axis. The form is standard
ball-flight law; both constants are hand-set, and one of them
(`axis_per_face_to_path`) has no good published value at all. So there is
`calibrate_face_to_path()`: give it one shot you actually hit — "3° of
face-to-path curved it 12 yards" — and it inverts the *validated* Layer 1
physics to measure the constant. Any launch monitor reporting face and path
gives you both inputs.

### A bug this surfaced

Writing the tests turned up a real defect in `LaunchConditions.spin_vector()`:
the spin axis was referenced to the **target line** rather than the flight
direction. A ball started 4° off line with "pure backspin" therefore had an axis
not square to its own flight, so the Magnus force acted in the wrong vertical
plane — the ground track bent ~1.5 yd on a 7-iron, and carry depended on the
start line, which is plainly wrong: a push does not shorten a shot. Launch
monitors define spin axis relative to flight, so the vector is now rotated by
the launch azimuth. Zero effect at azimuth 0 — every straight shot and the whole
calibration set — and it makes aim and curve decouple cleanly, which is what the
solvers rely on.

## 4. Hole maps

`hole diagrams/` (project root) holds an illustrated plan-view map and a green closeup
for all 18 holes, plus `hole info.txt`. `caddie/course/maps.py` georeferences the
maps so a trajectory can be drawn on the artwork.

**The transform.** Two reference points per hole — the back tee and the centre of
the green, located by eye — pin down scale, rotation and translation exactly,
which is all a plan view needs. It is a similarity **with a reflection**, because
local ENU is y-up and images are y-down:

```
px = A*E + B*N + c1        A = s·cos(theta)
py = B*E - A*N + c2        B = s·sin(theta)      det = -s^2
```

Fit a plain rotation instead and every hole comes out mirrored — which on a
symmetrical hole looks almost right. `test_maps.py` asserts the determinant is
negative on all 18 for exactly that reason.

**Checking hand-picked data.** The fit is exact at both reference points, so
their agreement proves nothing. Three things do, and `scripts/check_hole_maps.py`
reports all of them:

- **The route.** Drawing tee → aim points → pin uses geometry the two reference
  points did *not* determine. On all 18 the aim points land in the fairway, 13's
  route traces the dogleg, and 16's crosses the pond.
- **Implied scale.** Every hole's footprint comes out 240–650 m long by
  110–255 m wide.
- **Cross-hole coherence.** Metres-per-pixel tracks hole length monotonically —
  the par 3s (0.058–0.071) all finer than the long par 5s (0.140–0.148). The
  illustrations were drawn independently, so nothing in the picks imposed that;
  a badly wrong pick would fall off the trend.

**Accuracy, honestly.** These are marketing illustrations, not survey plans, with
reference pixels placed by eye. Good enough to see which side of a bunker a ball
finishes; useless for anything finer. If the overlay and the numbers disagree,
the drawing is wrong.

### hole info.txt

Par and yardage (which match the model, with one exception below) plus something
genuinely new: **historical scoring average and difficulty rank**. Nothing else
in this project knows which holes actually play hard — only how long they are and
how much they climb. 11 White Dogwood ranks hardest, 10 Camellia second, and both
sit in Amen Corner's approach; 2 Pink Dogwood is easiest relative to par.

**One source disagreement:** hole 17 is 440 yd on the PGA Tour card the course
model is built from, and 450 yd in this file. The model still uses 440. The
mismatch is asserted explicitly in `test_maps.py` so it stays visible and the
test breaks if either source is corrected.

## Next: the recommender

The two hard prerequisites are done and validated. What a shot recommendation
still needs:

1. **Player dispersion model** — a shot is a distribution, not a point. Build
   from launch monitor data, or transfer ShotLink dispersion from other venues.
2. **Bounce and roll** — carry is solved, total distance is not. Needs turf
   restitution and a Stimpmeter-calibrated roll model (Augusta runs ~13).
3. **Expected-strokes surface** — the objective function. Requires knowing what
   lies where, i.e. the green/bunker/water polygons noted as missing above.
4. **Monte-Carlo optimiser** over club and aim point, maximising expected
   strokes gained. The integrator is already deterministic and fast enough.

Step 3 is the real blocker, and it is a data problem rather than a modelling
one. Options: trace polygons from the Masters course map or satellite imagery,
or use the Cesium terrain/imagery path that provisualizer itself uses.
