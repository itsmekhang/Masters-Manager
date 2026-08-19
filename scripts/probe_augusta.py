"""Can we get real Augusta National data? Probe course stats + shot details."""
import pandas as pd
import pgatourpy as p

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

MASTERS_2025 = "R2025014"


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


section("pga_course_stats(R2025014)  <- hole-by-hole Augusta?")
try:
    cs = p.pga_course_stats(MASTERS_2025)
    print("shape:", cs.shape)
    print("columns:", list(cs.columns))
    print(cs.to_string())
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")

section("pga_tournament_overview(R2025014)")
try:
    ov = p.pga_tournament_overview(MASTERS_2025)
    for k, v in ov.items():
        s = repr(v)
        print(f"  {k}: {s[:200]}")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")

section("pga_field(R2025014) -- need a player_id")
player_id = None
try:
    fld = p.pga_field(MASTERS_2025)
    print("shape:", fld.shape, "columns:", list(fld.columns))
    print(fld.head(5).to_string())
    for col in ("player_id", "id", "playerId"):
        if col in fld.columns:
            player_id = str(fld.iloc[0][col])
            break
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")

# Rory McIlroy won the 2025 Masters; 28237 is his PGA Tour player id.
for candidate in [player_id, "28237"]:
    if not candidate:
        continue
    section(f"pga_shot_details({MASTERS_2025}, player_id={candidate}, round=4)")
    try:
        sd = p.pga_shot_details(MASTERS_2025, candidate, 4)
        print("shape:", sd.shape)
        print("columns:", list(sd.columns))
        print(sd.head(25).to_string())
        break
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")

section("pga_leaderboard_holes(R2025014, round=4) -- per-hole yardage/par")
try:
    lh = p.pga_leaderboard_holes(MASTERS_2025, round=4)
    print("shape:", lh.shape, "columns:", list(lh.columns))
    print(lh.head(20).to_string())
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
