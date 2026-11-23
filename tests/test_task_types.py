"""Offline test of the four task types' success rules (inspect / orbit / flyover / land), no simulator needed.

For each type it builds a small synthetic flight that does the task, a goal.json success block made by the publisher's own
task_spec(), and runs the evaluator on it and on the type's self-test edits (evaluation/evaluator_selftest.type_cases).
    python3 tests/test_task_types.py        -> one PASS / FAIL line per case
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("evaluation", "publishing")]
import camera_geometry as geo  # noqa: E402
from evaluate import evaluate  # noqa: E402
from evaluator_selftest import type_cases  # noqa: E402
from publish_episode import task_spec  # noqa: E402

K = {"fx_px": 1466.0, "fy_px": 1466.0, "cx_px": 640.0, "cy_px": 360.0}
SIZE = [1280, 720]
FPS = 30.0


def frame(i, x, y, z, look=None, pitch=None, yaw=None, landed=False):
    """One telemetry record; the camera looks at `look` (x, y, z) or uses yaw / pitch."""
    if look is not None:
        dx, dy, dz = look[0] - x, look[1] - y, look[2] - z
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(-dz, math.hypot(dx, dy)))
    return {"frame": i, "t": round(i / FPS, 4), "pose": {"x": x, "y": y, "z": z, "yaw_deg": yaw, "roll_deg": 0.0, "pitch_deg": 0.0},
            "camera": {"x": x, "y": y, "z": z, "yaw_deg": yaw, "pitch_deg": pitch}, "agl_m": z,
            "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
            "px4": {"armed": not landed, "landed_state": "ON_GROUND" if landed else "IN_AIR", "system_status": "ACTIVE"}}


def path(a, b, n):
    return [tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3)) for i in range(n + 1)]


def episode(typ):
    """-> (task, goal, telemetry, actions) of a synthetic flight that does the task."""
    box = {"center": [0.0, 0.0, 4.0], "half_extents": [5.0, 3.5, 4.0], "yaw_deg": 30.0}
    c = box["center"]
    top = c[2] + box["half_extents"][2]
    target = {"name": "house_1", "kind": "house", "description": "the test house", "xy": c[:2], "point_xyz": c,
              "box": box, "top_z": top}
    params, D = {}, None
    if typ == "inspect":
        D = 3.0
        lo, hi = 0.0, 60.0
        for _ in range(60):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if geo.distance_to_box((m, 0.0, 6.0), box) < D else (lo, m)
        pts = path((80.0, 0.0, 25.0), (hi, 0.0, 6.0), 300)
        tele = [frame(i, *p, look=c) for i, p in enumerate(pts)]
    elif typ == "orbit":
        R = 20.0
        params = {"radius_m": R, "direction": "counter-clockwise"}
        pts = path((80.0, 0.0, 25.0), (R, 0.0, 25.0), 150)
        pts += [(R * math.cos(a), R * math.sin(a), 25.0) for a in [2 * math.pi * 1.05 * i / 400 for i in range(1, 401)]]
        tele = [frame(i, *p, look=c) for i, p in enumerate(pts)]
    elif typ == "flyover":
        z = top + 16.0
        off = (z - c[2]) * math.tan(math.radians(10.0))
        pts = path((80.0, 0.0, 25.0), (off, 0.0, z), 300)
        tele = [frame(i, *p, look=c) for i, p in enumerate(pts)]
    else:
        D = 7.0
        params = {"standoff_m": D, "landing_radius_m": 25.0}
        lo, hi = 0.0, 60.0
        for _ in range(60):                       # spot at ground level whose box distance is D
            m = (lo + hi) / 2
            lo, hi = (m, hi) if geo.distance_to_box((m, 0.0, 0.23), box) < D else (lo, m)
        pts = path((80.0, 0.0, 20.0), (hi, 0.0, 20.0), 300) + path((hi, 0.0, 20.0), (hi, 0.0, 0.23), 200)[1:]
        tele = [frame(i, *p, pitch=5.0, yaw=180.0) for i, p in enumerate(pts)]
        tele += [frame(len(tele) + i, hi, 0.0, 0.23, pitch=5.0, yaw=180.0, landed=True) for i in range(30)]
    bench = {"params": params, "tolerance_m": 5.0}
    T = task_spec(typ, bench, D, 5.0, 120.0, True)
    goal = {"episode_id": f"test/{typ}", "target": target,
            "success": dict(T["success"], time_limit_s=120.0, px4_failure_states=["CRITICAL", "EMERGENCY", "FLIGHT_TERMINATION"]),
            "reference": {"path_m": round(geo.path_length(tele), 1)}}
    task = {"observation": {"rgb": {"camera_intrinsics": K, "size": SIZE}}}
    n = len(tele)
    acts = [{"index": 0, "frame": 0, "t": 0.0, "sticks": [0, 0, 0, 0]},
            {"index": 1, "frame": n, "t": round(n / FPS, 4), "stop": True}]
    return task, goal, tele, acts


COLLISION = {"monitor_verified": True, "first_prohibited": None, "invalid": None, "physics": {"hz": 240}, "contacts": []}


def main():
    bad = 0
    for typ in ("inspect", "orbit", "flyover", "land"):
        task, goal, tele, acts = episode(typ)
        cases = [(f"{typ}: synthetic flight that does the task", tele, acts, goal, True)]
        cases += [(f"{typ}: {name}", *rest) for name, *rest in type_cases(typ, task, goal, tele, acts)]
        for name, t, ac, g, expect in cases:
            r = evaluate(task, g, t, ac, COLLISION)
            ok = r["success"] == expect
            bad += not ok
            print(f'{"PASS" if ok else "FAIL"}  {name:70s} SR={r["success"]!s:5} NE={r["navigation_error_m"]}  ({r["episode_end"]})')
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
