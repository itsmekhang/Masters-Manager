"""Probe pgatourPY to find out what is actually reachable for Augusta National.

Question we need answered before relying on it:
  1. What functions exist?
  2. Is the Masters in the schedule (it is not a PGA Tour-operated event)?
  3. If so, does shot-level data with coordinates come back?
"""
import inspect
import pgatourpy as p

print("=" * 70)
print("EXPORTS")
print("=" * 70)
names = [n for n in dir(p) if not n.startswith("_")]
for n in names:
    obj = getattr(p, n)
    if callable(obj):
        try:
            sig = str(inspect.signature(obj))
        except (TypeError, ValueError):
            sig = "(?)"
        print(f"  {n}{sig}")
    else:
        print(f"  {n}  [{type(obj).__name__}]")

print()
print("=" * 70)
print("SCHEDULE 2025 -- looking for Masters")
print("=" * 70)
try:
    sched = p.pga_schedule(2025)
    print(type(sched))
    if hasattr(sched, "columns"):
        print("columns:", list(sched.columns))
        print(sched.head(30).to_string())
    else:
        print(repr(sched)[:3000])
except Exception as exc:
    print(f"pga_schedule FAILED: {type(exc).__name__}: {exc}")
