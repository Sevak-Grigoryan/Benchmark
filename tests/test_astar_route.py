"""Offline test of the opt-in A* transit planner (simulation/astar_route.py, BENCH_ROUTE_PLANNER=astar), no simulator needed.

1. astar_route self-test (a row of trees across the way: the route must go around or over it, every leg clear).
2. For scenes easy_v2_<first>..<last> at a low preferred height (default 10 m, where trees are in the way): plan with A*,
   fly the whole mission with the offline pilot and report the transit legs, the detour against the straight line and the
   closest approach to any obstacle (must stay >= benchmark_tasks.ROUTE_MARGIN_M; route_clear already rejects less).
3. With the planner unset, plan() must not contain any A* leg (the default straight route is unchanged).
    python3 tests/test_astar_route.py [first last [agl]]      -> one line per scene + PASS / FAIL; exit 1 on a failure
"""
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import offline_preview as op  # noqa: E402  (sets up every import path)

bt, pilot = op.bt, op.pilot
import astar_route  # noqa: E402


def clearance(m, labels, obst):
    """Closest horizontal approach to an obstacle (below its top + margin) while flying the beats labelled `labels`."""
    fr, _ = pilot.simulate_offline(m["beats"], op.ground, m["start_xy"], math.radians(m["yaw_deg"]))
    O = np.array(obst, dtype=float)
    worst = math.inf
    for f in fr[::3]:
        if f["beat"] not in labels:
            continue
        d = np.hypot(O[:, 0] - f["x"], O[:, 1] - f["y"]) - O[:, 2]
        below = f["z"] < O[:, 3] + bt.ROUTE_MARGIN_M
        if below.any():
            worst = min(worst, float(d[below].min()))
    return worst


def main():
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (101, 112)
    agl = sys.argv[3] if len(sys.argv) > 3 else "10"
    astar_route._selftest()
    bad = []
    os.environ.pop("BENCH_ROUTE_PLANNER", None)
    lay = op.easy_v2.layout(first, op.nv, op.fs)
    m = bt.plan(first, SimpleNamespace(layout=lay, ground=op.ground, bridge=None))
    if "route_planner" in m["bench"] or any("(leg " in b.get("label", "") for b in m["beats"]):
        bad.append((first, "default planner produced A* legs"))
    os.environ["BENCH_ROUTE_PLANNER"], os.environ["BENCH_ROUTE_AGL"] = "astar", agl
    for idx in range(first, last + 1):
        lay = op.easy_v2.layout(idx, op.nv, op.fs)
        try:
            m = bt.plan(idx, SimpleNamespace(layout=lay, ground=op.ground, bridge=None))
        except Exception as e:  # noqa: BLE001
            bad.append((idx, f"no plan: {e}"))
            continue
        i0, i1 = m["bench"]["episode_beats"]
        ep = m["beats"][i0:i1 + 1]
        legs = ep[:m["bench"]["route_planner"]["transit_legs"]]            # the A* legs, then the original last leg
        pts = [m["start_xy"]] + [b["xy"] for b in legs]
        length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        straight = math.dist(pts[0], pts[-1])
        skip_xy = m["bench"]["target"]["xy"]
        obst = [o for o in bt._obstacles(lay, set()) if math.hypot(o[0] - skip_xy[0], o[1] - skip_xy[1]) > 0.5]
        cam = op.episode_camera(m)
        c = clearance(m, {b["label"] for b in legs}, obst)
        if not cam["ok"]:
            bad.append((idx, "camera not ahead"))
        if c < bt.ROUTE_MARGIN_M - 1e-6:
            bad.append((idx, f"transit passes {c:.1f} m from an obstacle"))
        print(f"{idx} {m['bench']['task_type']:8s} legs {m['bench']['route_planner']['transit_legs']} "
              f"length {length:6.1f} m (straight {straight:6.1f}, +{100 * (length / max(straight, 1e-6) - 1):4.1f} %) "
              f"min clearance {c:5.1f} m  camera_ahead {cam['ok']}")
    print("PASS" if not bad else f"FAIL {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
