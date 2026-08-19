"""Is shot_details empty for Augusta specifically, or broken everywhere?

Augusta National does not host ShotLink (the Masters is run by the club, not
the PGA Tour, and uses its own tracking). If a normal Tour event returns
coordinates and the Masters does not, that confirms a data gap we must design
around rather than a library bug.
"""
import pandas as pd
import pgatourpy as p

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 80)

RORY = "28237"
CASES = [
    ("R2025011", "THE PLAYERS (TPC Sawgrass) - ShotLink venue"),
    ("R2025014", "Masters (Augusta National)"),
]

for tid, label in CASES:
    print("=" * 78)
    print(f"{label}   [{tid}]")
    print("=" * 78)
    for rnd in (1, 4):
        try:
            sd = p.pga_shot_details(tid, RORY, rnd)
            print(f"  round {rnd}: shape={sd.shape}")
            if not sd.empty:
                print(f"  columns: {list(sd.columns)}")
                print(sd.head(8).to_string())
                break
        except Exception as exc:
            print(f"  round {rnd}: FAILED {type(exc).__name__}: {exc}")
    print()

print("=" * 78)
print("pga_coverage(R2025014) -- what tracking exists at Augusta?")
print("=" * 78)
try:
    cov = p.pga_coverage("R2025014")
    print("shape:", cov.shape, "columns:", list(cov.columns))
    print(cov.to_string())
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
