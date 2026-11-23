"""
SCORE one flight of a collision-aware benchmark episode (the expert's or any agent's), using only goal.json + the flight.

    python evaluate.py <episode dir> --telemetry flown_telemetry.json --actions flown_action.json \
                       --collision flown_collision.json [-o result.json]

The flight is in the reference format: telemetry = one record per frame (t, pose, agl_m, camera, px4), actions = one entry
per change, collision = the PhysX contact monitor's episode record (reference/collision.json format, times from the episode
start); the episode ended by the agent's STOP if the last action is {"stop": true}, by the simulator's collision termination,
or by the time limit.

Collision rule (collision-v2: a contact counts only when PhysX applied a contact impulse): the first prohibited
UAV-environment contact ends and FAILS the episode. A flight without collision-monitoring data, or whose monitor was not
verified, is INVALID: it is never scored as collision-free.

The collision record: --collision <file>; without it, collisions.json (v0.3) or collision.json (v0.2) next to the telemetry.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_geometry as geo  # noqa: E402


def split_telemetry(obj):
    """(frames, embedded collision record or None) for both telemetry layouts."""
    if isinstance(obj, dict):
        return obj["frames"], obj.get("collision")
    return obj, None


def target_distance(rec, goal):
    """Drone-to-target distance used by the success rule. Goals from 2026-11-19 on ("distance" starts with "3D"): the
    straight-line 3D distance from the drone position to target.point_xyz. Older goals: the horizontal (x, y) distance to
    target.xy, so already-published episodes keep their meaning."""
    tg, p = goal["target"], rec["pose"]
    if goal["success"].get("distance", "").startswith("3D to the target box"):
        return geo.distance_to_box((p["x"], p["y"], p["z"]), tg["box"])
    if goal["success"].get("distance", "").startswith("3D"):
        return math.dist((p["x"], p["y"], p["z"]), tg["point_xyz"])
    return math.hypot(p["x"] - tg["xy"][0], p["y"] - tg["xy"][1])


def goal_state(rec, goal, K, size):
    """(in distance band, target in frame, distance to the target) for one frame."""
    s, tg = goal["success"], goal["target"]
    dh = target_distance(rec, goal)
    band = abs(dh - s["standoff_m"]) <= s["standoff_tolerance_m"]
    view = geo.in_frame(geo.project(rec["camera"], tg["point_xyz"], K), size, s["frame_margin"])
    return band, view, dh


def task_kind(goal):
    return goal["success"].get("task_type", "inspect")


def _in_view(rec, point, K, size, margin):
    return geo.in_frame(geo.project(rec["camera"], point, K), size, margin)


def _landed(rec):
    px4 = rec.get("px4") or {}
    return px4.get("landed_state") == "ON_GROUND" or px4.get("armed") is False


def task_outcome(tele, goal, K, size):
    """The task part of the success rule, per task type, on the flight `tele` (already cut at STOP / contact / time limit):
    goal_met (the task's own condition at STOP; orbit: over the flight), oracle (met at SOME frame), error_m (NE) and the
    numbers behind them."""
    s, tg = goal["success"], goal["target"]
    typ, last = task_kind(goal), tele[-1]
    pt = tg.get("point_xyz")
    if typ == "inspect":
        states = [goal_state(r, goal, K, size) for r in tele]
        band, view, d = states[-1]
        return {"goal_met": band and view, "oracle": any(b and v for b, v, _ in states),
                "error_m": abs(d - s["standoff_m"]),
                "details": {"distance_to_target_at_end_m": round(d, 2), "in_distance_band_at_end": band,
                            "target_in_frame_at_end": view,
                            "target_in_frame_fraction": round(sum(v for _, v, _ in states) / len(states), 3),
                            "closest_to_ring_m": round(min(abs(dd - s["standoff_m"]) for _, _, dd in states), 2)}}
    if typ == "orbit":
        R, tol = s["radius_m"], s["radius_tolerance_m"]
        band = (R - tol, R + tol)
        sign = 1.0 if s["direction"] == "counter-clockwise" else -1.0
        swept = sign * geo.swept_angle_deg(tele, pt[:2], band)
        inband = [r for r in tele if band[0] <= math.hypot(r["pose"]["x"] - pt[0], r["pose"]["y"] - pt[1]) <= band[1]]
        frac = sum(_in_view(r, pt, K, size, 0.0) for r in inband) / len(inband) if inband else 0.0
        met = swept >= s["min_swept_deg"] and frac >= s.get("min_in_view_fraction", 0.0)
        r_end = math.hypot(last["pose"]["x"] - pt[0], last["pose"]["y"] - pt[1])
        return {"goal_met": met, "oracle": met, "error_m": abs(r_end - R),
                "details": {"swept_deg_in_band": round(swept, 1), "in_view_fraction_in_band": round(frac, 3),
                            "radius_at_end_m": round(r_end, 2)}}
    if typ == "flyover":
        def state(r):
            h = math.hypot(r["pose"]["x"] - pt[0], r["pose"]["y"] - pt[1])
            above = r["pose"]["z"] >= tg["top_z"] + s["min_height_above_top_m"]
            view = _in_view(r, pt, K, size, s["frame_margin"])
            return h <= s["max_horizontal_m"] and above and view, h, above, view
        st = [state(r) for r in tele]
        ok, h, above, view = st[-1]
        return {"goal_met": ok, "oracle": any(x[0] for x in st), "error_m": h,
                "details": {"horizontal_distance_at_end_m": round(h, 2), "above_target_at_end": above,
                            "target_in_frame_at_end": view, "height_above_top_at_end_m": round(last["pose"]["z"] - tg["top_z"], 2)}}
    if typ == "land" and "standoff_m" in s:
        D, tol = s["standoff_m"], s["standoff_tolerance_m"]
        def state(r):
            d = geo.distance_to_box((r["pose"]["x"], r["pose"]["y"], r["pose"]["z"]), tg["box"])
            return _landed(r) and abs(d - D) <= tol, d
        st = [state(r) for r in tele]
        ok, d = st[-1]
        return {"goal_met": ok, "oracle": any(x[0] for x in st), "error_m": abs(d - D),
                "details": {"landed_at_end": _landed(last), "distance_to_target_at_end_m": round(d, 2),
                            "closest_to_ring_m": round(min(abs(x[1] - D) for x in st), 2)}}
    if typ == "land":
        pad = s["pad_xy"]
        def state(r):
            h = math.hypot(r["pose"]["x"] - pad[0], r["pose"]["y"] - pad[1])
            return _landed(r) and h <= s["max_horizontal_m"], h
        st = [state(r) for r in tele]
        ok, h = st[-1]
        return {"goal_met": ok, "oracle": any(x[0] for x in st), "error_m": h,
                "details": {"landed_at_end": _landed(last), "horizontal_distance_to_spot_at_end_m": round(h, 2)}}
    raise ValueError(f"unknown task type {typ}")


def flight_failure(goal, tele):
    """(t, reason) of the first flight failure, or (None, None); ("missing", why) when it cannot be checked.
    v0.3b goals: PX4's own report (HEARTBEAT system_status in px4_failure_states). Older goals: their tilt / height limits."""
    s = goal["success"]
    if "px4_failure_states" in s:
        if any("system_status" not in (r.get("px4") or {}) for r in tele):
            return "missing", "no PX4 system_status in the telemetry"
        r = next((r for r in tele if r["px4"]["system_status"] in s["px4_failure_states"]), None)
        return (None, None) if r is None else (r["t"], f"PX4 reported {r['px4']['system_status']}")
    for r in tele:
        if abs(r["pose"]["roll_deg"]) > s["max_tilt_deg"] or abs(r["pose"]["pitch_deg"]) > s["max_tilt_deg"]:
            return r["t"], f"tilted beyond {s['max_tilt_deg']:.0f} deg (crash)"
        if r["agl_m"] < s["min_agl_m"]:
            return r["t"], f"below {s['min_agl_m']:.0f} m above the ground"
    return None, None


def clearance_stats(tele):
    """Smallest PhysX clearance over the frames that have one (None if no frame has it)."""
    out = {}
    for key, col in (("nearest_m", "nearest_collider"), ("nearest_non_terrain_m", "nearest_non_terrain_collider")):
        rows = [(r["clearance"][key], r["t"], r["clearance"][col]) for r in tele
                if r.get("clearance") and r["clearance"].get(key) is not None]
        out[key] = min(rows) if rows else None
    return out


def success_without_collision_rule(goal, K, size, tele, stopped, hit, collision, success):
    """Same success rule with contacts ignored, on the WHOLE flight. Needs a record-only flight (contacts recorded, flight not
    stopped): if the flight was terminated at the contact, what would have followed is unknown -> None. Upper bound: damage
    after a contact is not simulated."""
    if hit is None:
        return success
    if collision.get("termination"):
        return None
    fail_t, _ = flight_failure(goal, tele)
    if fail_t == "missing":
        return None
    return bool(stopped and fail_t is None and task_outcome(tele, goal, K, size)["goal_met"])


def impact_speed(hit, tele):
    """|v| of the drone at the first prohibited contact: the PhysX step's velocity if recorded, else the last frame before it."""
    v = hit.get("drone_velocity_mps")
    if v is None:
        prev = [r for r in tele if r["t"] <= hit["t"] + 1e-6] or tele[:1]
        v = [prev[-1]["velocity"][k] for k in ("vx", "vy", "vz")]
    return round(math.sqrt(sum(x * x for x in v)), 3)


def task_shortfall(typ, s, d):
    """Why a STOP was not in the goal state, in words."""
    if typ == "inspect":
        if not d["in_distance_band_at_end"]:
            return f"{d['distance_to_target_at_end_m']:.1f} m from the target (needs {s['standoff_m']:.0f} +- {s['standoff_tolerance_m']:.0f} m)"
        return "the target is not in the frame"
    if typ == "orbit":
        if d["swept_deg_in_band"] < s["min_swept_deg"]:
            return (f"only {d['swept_deg_in_band']:.0f} deg {s['direction']} around the target within {s['radius_m']:.0f} +- "
                    f"{s['radius_tolerance_m']:.0f} m (needs {s['min_swept_deg']:.0f})")
        return f"the camera was on the target only {d['in_view_fraction_in_band']:.0%} of the orbit (needs {s['min_in_view_fraction']:.0%})"
    if typ == "flyover":
        if d["horizontal_distance_at_end_m"] > s["max_horizontal_m"]:
            return f"{d['horizontal_distance_at_end_m']:.1f} m beside the target (needs <= {s['max_horizontal_m']:.0f} m)"
        if not d["above_target_at_end"]:
            return f"not above the target ({d['height_above_top_at_end_m']:.1f} m over its top, needs >= {s['min_height_above_top_m']:.0f})"
        return "the target is not in the frame"
    if not d["landed_at_end"]:
        return "not landed"
    if "standoff_m" in s:
        return (f"landed {d['distance_to_target_at_end_m']:.1f} m from the target "
                f"(needs {s['standoff_m']:.0f} +- {s['standoff_tolerance_m']:.0f} m)")
    return f"landed {d['horizontal_distance_to_spot_at_end_m']:.1f} m from the spot (needs <= {s['max_horizontal_m']:.0f} m)"


def invalid(goal, why):
    return {"episode_id": goal["episode_id"], "valid": False, "success": False, "oracle_success": False, "spl": 0.0,
            "navigation_error_m": None, "episode_end": f"INVALID: {why}", "collision_monitoring": why}


def evaluate(task, goal, tele, actions, collision):
    s, tg = goal["success"], goal["target"]
    if collision is None:
        return invalid(goal, "no collision-monitoring data -- not scored as a collision-free flight")
    if not collision.get("monitor_verified"):
        return invalid(goal, "collision monitor not verified (no contact reports seen during initialization)")
    if collision.get("invalid"):
        return invalid(goal, f"episode invalid: {collision['invalid'].get('verdict')}")
    hit = collision.get("first_prohibited")
    hit_t = hit["t"] if hit else None
    K, size = task["observation"]["rgb"]["camera_intrinsics"], task["observation"]["rgb"]["size"]
    t_limit = s["time_limit_s"]
    stopped = bool(actions) and bool(actions[-1].get("stop"))
    stop_t = actions[-1]["t"] if stopped else None
    tele = [r for r in tele if r["t"] <= t_limit + 1e-6] or tele[:1]
    if stopped and stop_t > t_limit + 1e-6:
        stopped = False                                   # said STOP only after the time limit
    full_tele, full_stopped = tele, stopped               # whole flight, for the score without the collision rule
    if hit_t is not None and hit_t > t_limit + 1e-6:
        hit, hit_t = None, None                           # a contact after the time limit is outside the episode
    if hit_t is not None:
        if stopped and stop_t < hit_t - 1e-9:
            hit, hit_t = None, None                       # STOP came first: the episode was over
        else:
            stopped = False
            tele = [r for r in tele if r["t"] <= hit_t + 1e-6] or tele[:1]
    last = tele[-1]

    fail_t, fail_why = flight_failure(goal, tele)
    if fail_t == "missing":
        return invalid(goal, fail_why)
    out = task_outcome(tele, goal, K, size)
    typ = task_kind(goal)
    safe = fail_t is None and hit is None
    success = stopped and safe and out["goal_met"]
    L, L_ref = geo.path_length(tele), goal["reference"]["path_m"]
    cl = clearance_stats(tele)
    if hit is not None:
        end = (f"COLLISION at t = {hit_t:.4f} s (physics step {hit['episode_step']}): {hit['drone_part']} "
               f"{hit['drone_collider'].rsplit('/', 1)[-1]} touched {hit['other_collider']}")
    elif not stopped:
        end = "time limit reached without STOP" if last["t"] >= t_limit - 1.0 else "flight ended without STOP"
    elif fail_t is not None:
        end = f"STOP, but {fail_why} at t = {fail_t:.1f} s"
    elif out["goal_met"]:
        end = "STOP in the goal state"
    else:
        end = "STOP, but " + task_shortfall(typ, s, out["details"])
    return {
        "episode_id": goal["episode_id"],
        "valid": True,
        "success": bool(success),
        "task_type": typ,
        "oracle_success": out["oracle"] and fail_t is None,
        "navigation_error_m": round(out["error_m"], 2),
        "spl": round(float(success) * L_ref / max(L, L_ref, 1e-6), 4),
        "episode_end": end,
        "stopped": stopped,
        **out["details"],
        "heading_error_at_end_deg": round(geo.bearing_error_deg(last["pose"], tg["point_xyz"]), 1),
        "flight_failure": None if fail_t is None else {"t": round(fail_t, 3), "reason": fail_why},
        "max_tilt_deg": round(max(math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(r["pose"]["roll_deg"]))
                                                                        * math.cos(math.radians(r["pose"]["pitch_deg"]))))))
                                  for r in tele), 2),
        "collided": hit is not None,
        "collision": hit and {k: hit[k] for k in ("t", "episode_step", "drone_part", "drone_collider", "other_collider",
                                                  "total_impulse_ns")},
        "time_to_first_collision_s": None if hit is None else round(hit_t, 4),
        "collision_part": None if hit is None else hit["drone_part"],
        "collision_impulse_ns": None if hit is None else hit.get("total_impulse_ns"),
        "peak_contact_force_n": None if hit is None or hit.get("total_impulse_ns") is None
        else round(hit["total_impulse_ns"] * collision["physics"]["hz"], 3),
        "impact_speed_mps": None if hit is None else impact_speed(hit, tele),
        "min_clearance_m": None if cl["nearest_m"] is None else round(cl["nearest_m"][0], 3),
        "min_clearance_non_terrain_m": None if cl["nearest_non_terrain_m"] is None else round(cl["nearest_non_terrain_m"][0], 3),
        "min_clearance_non_terrain": cl["nearest_non_terrain_m"] and {"t": cl["nearest_non_terrain_m"][1],
                                                                     "collider": cl["nearest_non_terrain_m"][2]},
        "success_without_collision_rule": success_without_collision_rule(goal, K, size, full_tele, full_stopped, hit,
                                                                          collision, bool(success)),
        "path_length_m": round(L, 1),
        "reference_path_m": L_ref,
        "duration_s": round(last["t"], 2),
        "time_limit_s": t_limit,
        "end_position_xyz": [round(v, 2) for v in geo.xyz(last)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path)
    ap.add_argument("--telemetry", required=True, type=Path)
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("--collision", type=Path, help="v0.2 layout: the flight's collision.json (v0.3: inside telemetry.json; "
                                                  "no record at all -> the flight is INVALID)")
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    task = json.loads((a.episode / "task.json").read_text())
    goal = json.loads((a.episode / "goal.json").read_text())
    frames, col = split_telemetry(json.loads(a.telemetry.read_text()))
    if a.collision is None:
        a.collision = next((p for p in (a.telemetry.parent / "collisions.json", a.telemetry.parent / "collision.json")
                            if p.exists()), None)
    if a.collision is not None:
        col = json.loads(a.collision.read_text()) if a.collision.exists() else None
    res = evaluate(task, goal, frames, json.loads(a.actions.read_text())["actions"], col)
    txt = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
