"""
SELF-TEST of the success checker on one published episode: the expert flight must succeed, and edited copies of it (made in
memory only -- nothing on disk is changed) must give the expected verdict. These are checker tests, NOT agent flights.

    python evaluation/evaluator_selftest.py <episode dir> [-o results/checker_selftest.json]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, split_telemetry, task_kind  # noqa: E402


def shift_end(tele, goal, metres):
    """Move the last frame `metres` toward (+) or away from (-) the target, along the line the goal measures: the 3D line
    to target.point_xyz for 3D goals (the height above ground changes with it), else horizontally to target.xy."""
    t = copy.deepcopy(tele)
    r = t[-1]
    three_d = goal["success"].get("distance", "").startswith("3D")
    tx, ty, tz = goal["target"]["point_xyz"] if three_d else (*goal["target"]["xy"], r["pose"]["z"])
    dx, dy, dz = tx - r["pose"]["x"], ty - r["pose"]["y"], tz - r["pose"]["z"]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    for key in ("pose", "camera"):
        r[key]["x"] += metres * dx / d
        r[key]["y"] += metres * dy / d
        r[key]["z"] += metres * dz / d
    r["agl_m"] += metres * dz / d
    return t


def turn_end(tele, deg):
    t = copy.deepcopy(tele)
    for key in ("pose", "camera"):
        t[-1][key]["yaw_deg"] += deg
    return t


def stop_at(tele, acts, frac):
    n = max(2, int(len(tele) * frac))
    a = [x for x in copy.deepcopy(acts) if not x.get("stop") and x["frame"] < n]
    return tele[:n], a + [{"index": len(a), "frame": n, "t": tele[n - 1]["t"], "stop": True}]


def edit_frame(tele, i, **kw):
    t = copy.deepcopy(tele)
    for path, v in kw.items():
        a, b = path.split("__")
        t[i][a][b] = v
    return t


def strip_status(tele):
    t = copy.deepcopy(tele)
    for r in t:
        r["px4"].pop("system_status", None)
    return t


def with_contact(col, tele, frac, verdict="prohibited"):
    """A copy of the collision record with one contact of `verdict` at fraction `frac` of the episode (in memory)."""
    c = copy.deepcopy(col)
    t = tele[-1]["t"] * frac
    rec = {"t": round(t, 6), "episode_step": int(round(t * c["physics"]["hz"])), "step": None, "phase": "episode",
           "verdict": verdict, "drone_part": "rotor", "drone_collider": "/World/x500/body/collision_proxies/rotor_disc_0",
           "other_collider": "/World/CollisionProxies/(edited copy)", "total_impulse_ns": 0.01, "points": []}
    if verdict == "prohibited":
        c["first_prohibited"] = rec
    c["contacts"] = c.get("contacts", []) + [rec]
    return c


def move_all(tele, fn):
    """A copy of the flight with every frame's pose and camera moved by fn(x, y) -> (x, y)."""
    t = copy.deepcopy(tele)
    for r in t:
        for key in ("pose", "camera"):
            r[key]["x"], r[key]["y"] = fn(r[key]["x"], r[key]["y"])
    return t


def set_last(tele, n_frames, **kw):
    """A copy with the last n_frames frames edited: key 'a__b' sets r[a][b]."""
    t = copy.deepcopy(tele)
    for r in t[-n_frames:]:
        for path, v in kw.items():
            a, b = path.split("__")
            r[a][b] = v
    return t


def type_cases(kind, task, goal, tele, acts):
    """Cases that check the task part of the rule, per task type (every one an edited copy of the expert flight)."""
    s = goal["success"]
    if kind == "inspect":
        tol = s["standoff_tolerance_m"]
        out = [(f"stops {tol * 0.6:.0f} m further out (inside the +-{tol:.0f} m band)", shift_end(tele, goal, -0.6 * tol), acts, goal, True),
               ("stops half-way to the target", *stop_at(tele, acts, 0.5), goal, False),
               (f"stops {tol + 10:.0f} m too far from the target", shift_end(tele, goal, -(tol + 10)), acts, goal, False),
               ("stops at the right distance but turned 120 deg away", turn_end(tele, 120.0), acts, goal, False)]
        if not s.get("distance", "").startswith("3D to the target box"):
            out.append((f"stops {tol + 10:.0f} m too close to the target", shift_end(tele, goal, tol + 10), acts, goal, False))
        return out
    pt = goal["target"]["point_xyz"]
    if kind == "orbit":
        flip = "clockwise" if s["direction"] == "counter-clockwise" else "counter-clockwise"
        wide = move_all(tele, lambda x, y: (pt[0] + 2.0 * (x - pt[0]), pt[1] + 2.0 * (y - pt[1])))
        away = copy.deepcopy(tele)
        for r in away:
            r["camera"]["yaw_deg"] += 180.0
        out = [("stops after half of the flight (circle not complete)", *stop_at(tele, acts, 0.5), goal, False),
               (f"circles {s['direction']} but the task asked {flip}", tele, acts,
                {**goal, "success": {**s, "direction": flip}}, False),
               ("circles at twice the radius (outside the band)", wide, acts, goal, False)]
        if s.get("min_in_view_fraction", 0.0) > 0:      # goals that require the camera on the target while circling
            out.append(("circles with the camera turned away from the target", away, acts, goal, False))
        else:                                           # camera free (forward along the circle): turning it changes nothing
            out.append(("circles with the camera turned away (camera not part of this rule)", away, acts, goal, True))
        return out
    if kind == "flyover":
        low = set_last(tele, 1, pose__z=goal["target"]["top_z"] - 1.0)
        return [("stops half-way to the target", *stop_at(tele, acts, 0.5), goal, False),
                ("stops 10 m beside the target", shift_end(tele, {**goal, "success": {}}, -10.0), acts, goal, False),
                ("stops below the target's top", low, acts, goal, False),
                ("stops above it but with the camera level (pitch 0)", set_last(tele, 1, camera__pitch_deg=0.0), acts, goal, False)]
    if kind == "land" and "standoff_m" in s:
        c, D, tol = goal["target"]["box"]["center"], s["standoff_m"], s["standoff_tolerance_m"]
        e = tele[-1]["pose"]
        h = math.hypot(e["x"] - c[0], e["y"] - c[1]) or 1.0
        ux, uy = (e["x"] - c[0]) / h, (e["y"] - c[1]) / h
        out = lambda k: move_all(tele, lambda x, y: (x + k * ux, y + k * uy))
        return [("stops half-way to the target (in the air)", *stop_at(tele, acts, 0.5), goal, False),
                ("says STOP still in the air at the landing spot", set_last(tele, 1, px4__landed_state="IN_AIR", px4__armed=True),
                 acts, goal, False),
                (f"lands {0.6 * tol:.0f} m further from the target (inside the +-{tol:.0f} m band)", out(0.6 * tol), acts, goal, True),
                (f"lands {tol + 10:.0f} m too far from the target", out(tol + 10.0), acts, goal, False)]
    if kind == "land":
        pad = s["pad_xy"]
        return [("stops half-way to the landing pad (in the air)", *stop_at(tele, acts, 0.5), goal, False),
                ("says STOP still in the air above the spot", set_last(tele, 1, px4__landed_state="IN_AIR", px4__armed=True),
                 acts, goal, False),
                ("lands 3 m from the spot (inside the 5 m radius)", move_all(tele, lambda x, y: (x + 3.0, y)), acts, goal, True),
                ("lands 8 m from the spot", move_all(tele, lambda x, y: (x + 8.0, y)), acts, goal, False)]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    task = json.loads((a.episode / "task.json").read_text())
    goal = json.loads((a.episode / "goal.json").read_text())
    tele, col = split_telemetry(json.loads((a.episode / "reference/telemetry.json").read_text()))
    acts = json.loads((a.episode / "reference/action.json").read_text())["actions"]
    if col is None:
        ref = a.episode / "reference"
        col = json.loads(((ref / "collisions.json") if (ref / "collisions.json").exists() else (ref / "collision.json")).read_text())
    mid = len(tele) // 2
    no_stop = [x for x in acts if not x.get("stop")]
    low = copy.deepcopy(tele)
    low[mid]["agl_m"] = 0.5
    unverified = dict(copy.deepcopy(col), monitor_verified=False)
    cases = [
        ("expert flight, unchanged", tele, acts, goal, True),
        ("rotor touches a tree half-way (prohibited contact)", tele, acts, goal, False, with_contact(col, tele, 0.5)),
        ("brief prohibited contact 0.2 s before STOP", tele, acts, goal, False,
         with_contact(col, tele, 1.0 - 0.2 / tele[-1]["t"])),
        ("contact only inside the contact offset (proximity, no touch)", tele, acts, goal, True,
         with_contact(col, tele, 0.5, "proximity")),
        ("no collision-monitoring data", tele, acts, goal, False, None),
        ("collision monitor not verified", tele, acts, goal, False, unverified),
        ("never says STOP (flies until the time limit)", tele, no_stop, goal, False),
        ("too slow: time limit cut to 20 s", tele, acts,
         {**goal, "success": {**goal["success"], "time_limit_s": 20.0}}, False),
    ]
    cases += type_cases(task_kind(goal), task, goal, tele, acts)
    if "px4_failure_states" in goal["success"]:
        # v0.3b: no tilt / height limits of the benchmark; PX4's own failure report fails the flight
        cases += [
            ("75 deg tilt, PX4 reports nothing (agent's own choice)", edit_frame(tele, mid, pose__roll_deg=75.0), acts, goal, True),
            ("flies 0.5 m above the ground without touching it", low, acts, goal, True),
            ("PX4 reports CRITICAL (failsafe) in the middle of the flight", edit_frame(tele, mid, px4__system_status="CRITICAL"),
             acts, goal, False),
            ("PX4 terminates the flight (FLIGHT_TERMINATION)", edit_frame(tele, mid, px4__system_status="FLIGHT_TERMINATION"),
             acts, goal, False),
            ("no PX4 status in the telemetry (invalid)", strip_status(tele), acts, goal, False),
        ]
    else:
        cases += [
            ("crash: 75 deg tilt in the middle of the flight", edit_frame(tele, mid, pose__roll_deg=75.0), acts, goal, False),
            ("touches the ground in the middle of the flight", low, acts, goal, False),
        ]
    rows = []
    for name, t, ac, g, expect, *c in cases:
        r = evaluate(task, g, t, ac, c[0] if c else col)
        ok = r["success"] == expect
        rows.append({"case": name, "expected_success": expect, "success": r["success"], "oracle_success": r["oracle_success"],
                     "valid": r["valid"], "navigation_error_m": r["navigation_error_m"], "spl": r["spl"],
                     "episode_end": r["episode_end"], "pass": ok})
        ne = "  n/a" if r["navigation_error_m"] is None else f'{r["navigation_error_m"]:5.1f}'
        print(f'{"PASS" if ok else "FAIL"}  {name:62s} SR={r["success"]!s:5} OSR={r["oracle_success"]!s:5} '
              f'NE={ne} m SPL={r["spl"]:.2f}  ({r["episode_end"]})')
    # score without the collision rule (record-only flights: contact recorded, flight continued; stopped flights: unknown)
    stopped_col = with_contact(col, tele, 0.5)
    stopped_col["termination"] = {"reason": "prohibited contact (edited copy)"}
    nc_cases = [
        ("record-only: contact half-way, then STOP in the goal state", with_contact(col, tele, 0.5), tele, acts, False, True),
        ("record-only: contact half-way, then STOP far from the target", with_contact(col, tele, 0.5),
         *stop_at(tele, acts, 0.6), False, False),
        ("stopped at the contact: score without the rule is unknown", stopped_col, tele, acts, False, None),
        ("no contact: both scores agree", col, tele, acts, True, True),
    ]
    for name, c, t, ac, exp_sr, exp_nc in nc_cases:
        r = evaluate(task, goal, t, ac, c)
        ok = r["success"] == exp_sr and r["success_without_collision_rule"] == exp_nc
        rows.append({"case": name, "expected_success": exp_sr, "success": r["success"],
                     "expected_success_without_collision_rule": exp_nc,
                     "success_without_collision_rule": r["success_without_collision_rule"], "valid": r["valid"],
                     "episode_end": r["episode_end"], "pass": ok})
        print(f'{"PASS" if ok else "FAIL"}  {name:62s} SR={r["success"]!s:5} SR-NC={r["success_without_collision_rule"]!s:5}')
    c = with_contact(col, tele, 0.5)
    c["first_prohibited"]["drone_velocity_mps"] = [3.0, 4.0, 0.0]
    r = evaluate(task, goal, tele, acts, c)
    hz = col["physics"]["hz"]
    ok = r["impact_speed_mps"] == 5.0 and r["peak_contact_force_n"] == round(0.01 * hz, 3)
    rows.append({"case": "impact metrics: |v| = 5 m/s, force = impulse x physics rate", "impact_speed_mps": r["impact_speed_mps"],
                 "peak_contact_force_n": r["peak_contact_force_n"], "pass": ok})
    print(f'{"PASS" if ok else "FAIL"}  {"impact metrics: |v| = 5 m/s, force = impulse x physics rate":62s} '
          f'speed={r["impact_speed_mps"]} force={r["peak_contact_force_n"]} N')
    if a.out:
        a.out.write_text(json.dumps({"note": "checker self-test: edited copies of the expert flight, in memory; not agent flights",
                                     "all_pass": all(x["pass"] for x in rows), "cases": rows}, indent=1) + "\n")
    sys.exit(0 if all(x["pass"] for x in rows) else 1)


if __name__ == "__main__":
    main()
