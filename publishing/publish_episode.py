"""
PUBLISH one collision-aware benchmark episode from its flown + rendered work dir (login node, ffmpeg only).

    python publishing/publish_episode.py --work $BENCH_RUNS/vlncol_forest_073 \
                                         --dest $BENCH_DATA/Easy/Forest/Simulation_N

dest/
  task.json          WHAT THE AGENT GETS: scene, start pose, instruction (+ paraphrases), observation / action interface
  goal.json          HIDDEN, evaluator only: target, stand-off distance + tolerance, success rule, thresholds, reference stats
  reference/         the expert solution of the episode (one valid way to succeed, NOT the definition of success)
    trajectory.mp4   onboard camera, episode frames only (no take-off / landing)
    action.json      sticks + gimbal command, one entry per change, ends with STOP
    telemetry.json   every frame: pose, attitude quaternion, physics step, velocity, camera pose + extrinsics, PX4 state
    collisions.json  the PhysX contact monitor's record of the episode (collision-v2): monitor verified, contacts, coverage,
                     termination
  results/           filled by evaluate.py / evaluator_selftest.py

Refuses to publish if: dest is not empty, the video frame count differs from the telemetry, a paraphrase is too close to the
canonical instruction (similarity >= 0.8), a paraphrase / the short instruction drops the stand-off distance, the flight has no
verified collision monitoring, or the expert touched anything during the episode.
"""
import argparse
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # Benchmark_codebase
sys.path[:0] = [str(HERE), *(str(ROOT / d) for d in ("collision", "evaluation", "simulation"))]
import episode_collisions as collision_record  # noqa: E402
import camera_geometry as geo  # noqa: E402
from episode_frames import episode_window  # noqa: E402

_CONDA_BIN = Path(os.environ.get("CONDA_ROOT", "/mnt/weka/shared-cache/miniforge3")) / "bin"
FFMPEG = shutil.which("ffmpeg") or str(_CONDA_BIN / "ffmpeg")
FFPROBE = shutil.which("ffprobe") or str(_CONDA_BIN / "ffprobe")
BENCH_VERSION = "vln-x500-bench v0.4 (2026-11-19; four Easy task types: inspect / orbit / flyover / land; distances to the target box; 5 m tolerance; egocentric instructions; collision-v2: a contact counts only when PhysX applies a contact impulse; PX4 failure report; PhysX clearance + impact metrics)"
FPS = 30
GIMBAL_STEP_DEG = 5.0
TIME_LIMIT_FACTOR = 2.0
STANDOFF_TOL = (10.0, 0.30)     # |3D distance - D| <= max(10 m, 30 % of D)
COMPASS_WORDS = re.compile(r"\b(north|south|east|west)(ern|wards?|-?(east|west)(ern)?)?\b", re.I)
SIDE_LIMITS_DEG = (30.0, 150.0)  # |bearing from the start heading| <= 30: ahead; >= 150: behind; else left / right
FRAME_MARGIN = 0.10              # target point inside the image with 10 % of width / height clear at every edge
# v0.3b (2026-11-14): no tilt / height limits of ours. A crash is what PX4 itself reports: its HEARTBEAT system_status
# (MAVLink MAV_STATE) leaves normal operation when its failsafe / Failure Detector acts or it terminates the flight.
PX4_FAILURE_STATES = ("CRITICAL", "EMERGENCY", "FLIGHT_TERMINATION")
SIMILARITY_MAX = 0.8


def nframes(p):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def stitch(work, f0, f1, out):
    slices = []
    for p in sorted((work / "slices").glob("slice_*.json")):
        info = json.loads(p.read_text())
        if f0 <= info["start"] < f1:
            slices.append((info, p.with_suffix(".mp4")))
    pos = f0
    for info, mp4 in slices:
        if info["start"] != pos or not mp4.exists() or nframes(mp4) != info["frames"]:
            sys.exit(f"slice coverage broken at frame {pos} ({mp4.name})")
        pos = info["end"]
    if pos != f1:
        sys.exit(f"slices cover [{f0},{pos}) but the episode is [{f0},{f1})")
    lst = work / "slices" / "concat_episode.txt"
    lst.write_text("".join(f"file '{mp4.resolve()}'\n" for _, mp4 in slices))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    got = nframes(out)
    if got != f1 - f0:
        sys.exit(f"stitched video has {got} frames, the episode has {f1 - f0}")


def episode_actions(tele, flight_actions, f0, f1):
    """Sticks the expert held + the gimbal pitch command (5 deg steps), one entry per change, then STOP."""
    out = []
    for f in tele[f0:f1]:
        a = flight_actions[f["action_index"]]
        g = GIMBAL_STEP_DEG * round(f["camera"]["pitch_deg"] / GIMBAL_STEP_DEG)
        cmd = (a["x1"], a["y1"], a["x2"], a["y2"], g)
        if not out or tuple(out[-1][k] for k in ("x1", "y1", "x2", "y2", "gimbal_pitch_deg")) != cmd:
            if out:
                out[-1]["hold_frames"] = f["frame"] - f0 - out[-1]["frame"]
                out[-1]["hold_s"] = round(out[-1]["hold_frames"] / FPS, 4)
            out.append({"index": len(out), "frame": f["frame"] - f0, "t": round((f["frame"] - f0) / FPS, 4),
                        "x1": cmd[0], "y1": cmd[1], "x2": cmd[2], "y2": cmd[3], "gimbal_pitch_deg": g})
    out[-1]["hold_frames"] = f1 - f0 - out[-1]["frame"]
    out[-1]["hold_s"] = round(out[-1]["hold_frames"] / FPS, 4)
    out.append({"index": len(out), "frame": f1 - f0, "t": round((f1 - f0) / FPS, 4), "stop": True})
    return out


def episode_telemetry(tele, acts, f0, f1):
    out, ai = [], 0
    for f in tele[f0:f1]:
        i = f["frame"] - f0
        while ai + 1 < len(acts) and not acts[ai + 1].get("stop") and acts[ai + 1]["frame"] <= i:
            ai += 1
        out.append({"frame": i, "t": round(i / FPS, 4), "action_index": ai, "physics_step": f["physics_step"] - tele[f0]["physics_step"],
                    "pose": f["pose"], "attitude_quat_xyzw": f["attitude_quat_xyzw"], "velocity": f["velocity"],
                    "agl_m": f["agl_m"], "camera": f["camera"], "camera_extrinsics": geo.extrinsics(f["camera"]),
                    "px4": f["px4"], "clearance": f.get("clearance")})
    return out


def target_side(start_xy, start_yaw_deg, target_xy):
    """Where the target is as seen from the drone at the start pose: ahead / left / right / behind (+ the signed bearing,
    + = left, counter-clockwise like the ENU yaw)."""
    b = math.degrees(math.atan2(target_xy[1] - start_xy[1], target_xy[0] - start_xy[0]))
    rel = (b - start_yaw_deg + 180.0) % 360.0 - 180.0
    if abs(rel) <= SIDE_LIMITS_DEG[0]:
        return "ahead", rel
    if abs(rel) >= SIDE_LIMITS_DEG[1]:
        return "behind", rel
    return ("left" if rel > 0 else "right"), rel


def check_instructions(canon, para, short, D, side):
    """The agent does not know the compass: no north / south / east / west anywhere, and every text must name the
    target's side as seen from the start heading."""
    for text in [canon, *para, short]:
        if COMPASS_WORDS.search(text):
            sys.exit(f"compass direction in an instruction (use left / right / ahead / behind): {text}")
        if not re.search(rf"\b{side}\b", text, re.I):
            sys.exit(f"instruction does not say the target is {side} of the start heading: {text}")
    num = None if D is None else str(int(round(D)))
    for p in para:
        s = difflib.SequenceMatcher(None, canon.lower(), p.lower()).ratio()
        if s >= SIMILARITY_MAX:
            sys.exit(f"paraphrase too close to the canonical instruction ({s:.2f}): {p}")
        if num and num not in p:
            sys.exit(f"paraphrase drops the {num} m stand-off: {p}")
    if num and num not in short:
        sys.exit(f"short instruction drops the {num} m stand-off: {short}")
    return [round(difflib.SequenceMatcher(None, canon.lower(), p.lower()).ratio(), 3) for p in para]


COMMON_RULE = (", and during the whole episode PX4 never reported a failure (its HEARTBEAT system_status stayed out of "
               + "/".join(PX4_FAILURE_STATES) + ": PX4's own failsafe / Failure Detector / flight termination), and no part of "
               "the drone (rotors, airframe, landing gear) touched anything -- the PhysX contact monitor's first prohibited "
               "contact ends and fails the episode. A flight without verified collision monitoring, or without PX4 status, is "
               "invalid. No tilt, height or speed limit of the benchmark's own is applied: how the agent flies is its own decision")
NO_LANDING = ("the simulator ends the episode at the first contact of the drone with anything (this episode has no landing), "
              "right after that 240 Hz physics step")


def task_spec(typ, bench, D, tol, t_limit, new_goal):
    """Per task type: the task.json task_type text, the goal.json success block and rule text, the OSR / NE definitions and
    the collision note for the agent. new_goal: goals from 2026-11-19 on (distances to the target box, 5 m tolerance)."""
    P = bench.get("params", {})
    if typ == "inspect":
        dist = ("3D to the target box: shortest distance from the drone position to target.box (the target's oriented "
                "bounding box measured on the USD stage)" if new_goal else
                "3D (x, y, z): straight line from the drone position to target.point_xyz")
        where = "target.box surface" if new_goal else "target.point_xyz"
        return {"task_type": "inspect (stop at a distance from the target, facing it)",
                "success": {"task_type": "inspect", "standoff_m": D, "standoff_tolerance_m": tol, "distance": dist,
                            "frame_margin": FRAME_MARGIN,
                            "not_required": "the route, the speed, the height, how long it hovers"},
                "rule": lambda T: (f"the agent says STOP within {T:.0f} s, and at that moment its 3D distance to the {where} is "
                                   f"{D:.0f} +- {tol:.0f} m and the target point is inside the image with {FRAME_MARGIN:.0%} of "
                                   f"the frame clear at every edge"),
                "osr": "oracle success: at SOME frame the drone was in the distance band with the target in frame",
                "ne": f"navigation error at STOP = | 3D distance to the {where} - standoff_m |",
                "collision_text": NO_LANDING}
    if typ == "orbit":
        R, sense = P["radius_m"], P["direction"]
        return {"task_type": "orbit (circle the target once in the given direction, at a distance from its centre, then stop)",
                "success": {"task_type": "orbit", "radius_m": R, "radius_tolerance_m": tol, "direction": sense,
                            "direction_note": "seen from above; counter-clockwise = the drone's bearing from the target increases",
                            "radius_note": "horizontal distance from the drone to the target's centre (target.point_xyz)",
                            "min_swept_deg": 360.0, "min_in_view_fraction": 0.0,
                            "not_required": "the height, the speed, where the circle starts, where the agent stops, where the "
                                            "camera points (the expert flies with the camera ahead, along the circle)"},
                "rule": lambda T: (f"the agent says STOP within {T:.0f} s, and before it the drone went at least 360 deg "
                                   f"{sense} around the target's centre, counting only frames whose horizontal distance to it "
                                   f"is {R:.0f} +- {tol:.0f} m"),
                "osr": "oracle success: the orbit condition was met (same as the task part of SR, STOP not needed)",
                "ne": "navigation error at STOP = | horizontal distance to target.point_xyz - radius_m |",
                "collision_text": NO_LANDING}
    if typ == "flyover":
        return {"task_type": "flyover (stop right above the target, camera looking down at it)",
                "success": {"task_type": "flyover", "max_horizontal_m": tol, "min_height_above_top_m": 2.0,
                            "frame_margin": FRAME_MARGIN,
                            "not_required": "the route, the speed, the exact height above the target"},
                "rule": lambda T: (f"the agent says STOP within {T:.0f} s, and at that moment its horizontal distance to "
                                   f"target.point_xyz is <= {tol:.0f} m, it is at least 2 m above the target's top (target.top_z) "
                                   f"and the target point is inside the image with {FRAME_MARGIN:.0%} of the frame clear at every "
                                   f"edge"),
                "osr": "oracle success: at SOME frame the drone was above the target within the radius with the target in frame",
                "ne": "navigation error at STOP = horizontal distance to target.point_xyz",
                "collision_text": NO_LANDING}
    if typ == "land":
        return {"task_type": "land (land at a distance from the target structure)",
                "success": {"task_type": "land", "standoff_m": D, "standoff_tolerance_m": tol,
                            "distance": "3D to the target box: shortest distance from the drone position to target.box (the "
                                        "structure's oriented bounding box measured on the USD stage)",
                            "landed": "PX4 reports landed_state ON_GROUND (or disarmed) at STOP",
                            "not_required": "the side of the target it lands on, the route, the speed, the heading at touchdown"},
                "rule": lambda T: (f"the agent says STOP within {T:.0f} s, and at that moment PX4 reports the drone landed "
                                   f"(landed_state ON_GROUND or disarmed) with its 3D distance to the target.box surface "
                                   f"{D:.0f} +- {tol:.0f} m (any side)"),
                "osr": "oracle success: at SOME frame the drone was landed in the distance band",
                "ne": "navigation error at STOP = | 3D distance to the target.box surface - standoff_m |",
                "collision_text": (f"the simulator ends the episode at the first contact of the drone with anything, except the "
                                   f"landing gear touching the ground within {P['landing_radius_m']:.1f} m of the target's "
                                   f"centre (touching the target itself is a collision), right after that 240 Hz physics step")}
    raise ValueError(typ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--no-video", action="store_true", help="publish everything except reference/trajectory.mp4")
    a = ap.parse_args()
    work, dest = a.work, a.dest
    video = dest / "reference" / "trajectory.mp4"
    if (dest / "task.json").exists() and not video.exists():
        # published earlier with --no-video: only add the video, and check it matches the published telemetry
        f0, f1, _, _ = episode_window(work / "flight")
        n = len(json.loads((dest / "reference" / "telemetry.json").read_text()))
        if f1 - f0 != n:
            sys.exit(f"episode window has {f1 - f0} frames but the published telemetry has {n}")
        stitch(work, f0, f1, video)
        print(f"added {video} ({n} frames)")
        return
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {dest}")
    flight = work / "flight"
    if not (flight / "FLIGHT_OK").exists():
        sys.exit("flight not finished OK")
    meta = json.loads((flight / "meta.json").read_text())
    tele = json.loads((flight / "telemetry.json").read_text())
    fact = json.loads((flight / "action.json").read_text())["actions"]
    mid = meta["mission"]["id"]
    m = meta["mission"]
    bench, tgt = m["bench"], m["bench"]["target"]
    f0, f1, t0, t1 = episode_window(flight)
    fcol = json.loads((flight / "collision.json").read_text())
    ecol = collision_record.episode_record(fcol, tele[f0]["t"], tele[f1 - 1]["t"], tele[f0]["physics_step"])
    if not ecol["monitor_verified"]:
        sys.exit("collision monitor not verified for this flight")
    if ecol["first_prohibited"] or ecol["invalid"]:
        sys.exit(f"the expert touched something during the episode: {ecol['first_prohibited'] or ecol['invalid']}")
    bad = next((f for f in tele[f0:f1] if f["px4"].get("system_status") in PX4_FAILURE_STATES), None)
    if any("system_status" not in f["px4"] for f in tele[f0:f1]):
        sys.exit("telemetry has no PX4 system_status (flown before v0.3b): cannot check PX4 failures")
    if bad:
        sys.exit(f"PX4 reported {bad['px4']['system_status']} during the expert episode at t = {bad['t']}")

    ref = dest / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(exist_ok=True)
    if not a.no_video:
        stitch(work, f0, f1, ref / "trajectory.mp4")
    acts = episode_actions(tele, fact, f0, f1)
    et = episode_telemetry(tele, acts, f0, f1)
    (ref / "action.json").write_text(json.dumps({"actions": acts}, indent=1))
    (ref / "telemetry.json").write_text(json.dumps(et))
    (ref / "collisions.json").write_text(json.dumps(ecol, indent=1))

    typ = bench.get("task_type", "inspect")
    params = bench.get("params", {})
    D = bench.get("standoff_m", params.get("standoff_m"))
    pp = bench.get("instructions") or json.loads((HERE / "instructions.json").read_text()).get(mid)
    if not pp or "instruction" not in pp:
        sys.exit(f"no egocentric instruction for {mid} (mission bench.instructions or publishing/instructions.json)")
    K = meta["camera_intrinsics"]
    p0, last = et[0], et[-1]
    side, side_deg = target_side(geo.xyz(p0)[:2], p0["pose"]["yaw_deg"], tgt["xy"])
    num = {"inspect": D, "land": D, "orbit": params.get("radius_m")}.get(typ)
    sims = check_instructions(pp["instruction"], pp["paraphrases"], pp["short"], num, side)
    look = tgt.get("look_at_xyz") or tgt["point_xyz"]
    ref_len = geo.path_length(et)
    dur = round(len(et) / FPS, 3)
    t_limit = 10.0 * math.ceil(TIME_LIMIT_FACTOR * dur / 10.0)
    tol = bench.get("tolerance_m") or max(STANDOFF_TOL[0], STANDOFF_TOL[1] * D)
    ep_id = f"Easy/Forest/{dest.name}"
    sc = bench["scene"]
    view0 = geo.in_frame(geo.project(p0["camera"], look, K), meta["frame_size"], 0.0)
    T = task_spec(typ, bench, D, tol, t_limit, "tolerance_m" in bench)

    task = {
        "episode_id": ep_id,
        "benchmark_version": BENCH_VERSION,
        "difficulty": "Easy",
        "task_type": T["task_type"],
        "instruction": pp["instruction"],
        "instruction_paraphrases": pp["paraphrases"],
        "instruction_short": pp["short"],
        "scene": {"environment": "Forest", "scene_id": sc["scene_mission"], "source_simulation": sc["source_simulation"],
                  "rebuild": f"Benchmark_codebase Forest scene with the Easy v2 props layout of {sc['scene_mission']} "
                             f"(scene_loader.boot('Forest', ..., mission_id='{sc['scene_mission']}'))"},
        "start": {"xyz": [round(v, 3) for v in geo.xyz(p0)], "yaw_deg": round(p0["pose"]["yaw_deg"], 2),
                  "camera_pitch_deg": round(p0["camera"]["pitch_deg"], 1),
                  "state": "in the air, hovering, armed, PX4 Position mode",
                  "note": "the runner flies the take-off and the turn to this heading by itself and hands control to the agent here"},
        "time_limit_s": t_limit,
        "observation": {
            "rgb": {"size": meta["frame_size"], "fps": FPS, "camera_intrinsics": K,
                    "mount": "0.12 m forward of and 0.07 m below the airframe centre, yaw locked to the airframe, stabilised (no roll)"},
            "pose": "drone position (ENU metres) and heading every frame, as in telemetry.json pose",
            "instruction": "text",
            "instruction_directions": "left / right / ahead / behind are relative to the drone's heading at the start pose "
                                      "(no compass directions are used)",
        },
        "action": {
            "decision_interval_s": 0.5,
            "sticks": {"x1": "yaw, + = rotate clockwise", "y1": "throttle, + = climb", "x2": "roll, + = move right",
                       "y2": "pitch, + = move forward", "range": [-1.0, 1.0], "quantisation": 0.1,
                       "response_at_full_stick": {"horizontal_mps": 8.0, "climb_mps": 3.0, "descent_mps": 2.0, "yaw_dps": 60.0}},
            "gimbal_pitch_deg": {"range": [0.0, 80.0], "quantisation": GIMBAL_STEP_DEG, "meaning": "positive = looking down",
                                 "slew_dps": 35.0},
            "stop": "ends the episode; the drone holds its position (the agent says: I am done)",
            "collision": T["collision_text"],
            "autopilot": "PX4 v1.14.3 SITL, Position mode, sticks over MAVLink MANUAL_CONTROL",
        },
        "coordinate_frame": "ENU metres (x east, y north, z up); yaw 0 = +x, counter-clockwise positive; camera pitch positive = down",
    }
    goal = {
        "episode_id": ep_id,
        "hidden": "evaluator only -- never give this file to the agent",
        "target": {"name": tgt["name"], "description": tgt["description"], "kind": tgt["kind"],
                   "xy": [round(v, 3) for v in tgt["xy"]], "point_xyz": [round(v, 3) for v in look],
                   **({"base_z": round(tgt["base_z"], 2)} if "base_z" in tgt else {}),
                   **({"top_z": round(tgt["top_z"], 2)} if tgt.get("top_z") is not None else {}),
                   **({"box": tgt["box"]} if tgt.get("box") else {}),
                   "in_view_at_start": view0, "start_turn_from_target_deg": bench["start_turn_from_target_deg"],
                   "side_at_start": side, "bearing_from_start_heading_deg": round(side_deg, 1)},
        "success": dict(T["success"], time_limit_s=t_limit, px4_failure_states=list(PX4_FAILURE_STATES),
                        px4_failure_source="MAVLink HEARTBEAT.system_status sent by PX4 v1.14.3 (recorded per frame in "
                                           "telemetry px4.system_status)",
                        rule=T["rule"](t_limit) + COMMON_RULE),
        "collision_criteria": {
            "version": ecol["version"],
            "rule": ecol["rule"],
            "touch": ecol["touch_rule"],
            "source": "NVIDIA PhysX contact report (omni.physx subscribe_contact_report_events, PhysxContactReportAPI threshold 0); "
                      "impulse = PxContactPairPoint.impulse, the impulse the PhysX solver applied at the contact point",
            "proximity": "pairs PhysX reports inside the 0.04 m contact offset without any impulse are recorded, never a failure",
            "landing_exception": m.get("landing"),
            "physics_hz": ecol["physics"]["hz"],
            "termination": "the episode ends right after the physics step of the first prohibited contact",
            "missing_data": "a flight without a verified collision record is INVALID, never collision-free",
        },
        "metrics": {
            "navigation": {
                "SR": "success (rule above, collision included)",
                "OSR": T["osr"],
                "NE": T["ne"],
                "SPL": "SR x reference_path_m / max(path_m, reference_path_m)",
                "heading_error_deg": "angle between the drone heading and the direction to the target at STOP",
            },
            "collision": {
                "collided": "true if a prohibited contact happened during the episode (episode failed)",
                "time_to_first_collision_s": "episode time of the first prohibited contact",
                "collision_part": "drone part of the first prohibited contact: rotor / airframe / landing_gear",
                "collision_impulse_ns": "PhysX impulse of the first prohibited contact (N s): how hard it was",
                "peak_contact_force_n": "the same PhysX impulse divided by the physics step (1/240 s): the mean contact force in "
                                        "that step (N)",
                "impact_speed_mps": "the drone's speed at the physics step of the first prohibited contact (Isaac Sim rigid-body "
                                    "velocity; from the frame just before the contact for records without it)",
                "min_clearance_m": "smallest distance from the drone body origin to any environment collider (terrain included) "
                                   "during the episode, measured by PhysX scene queries (overlap_sphere, 1 cm, range 30 m)",
                "min_clearance_non_terrain_m": "the same to the nearest collider that is not the terrain (trees, buildings, "
                                               "props), with its collider path: how close the flight came to an obstacle",
                "success_without_collision_rule": "the same success rule with contacts ignored, scored on a record-only flight "
                                                  "(contacts recorded, flight not stopped). UPPER BOUND: the simulator does not "
                                                  "model damage after a contact (no propeller breakage or thrust loss). Null when "
                                                  "the flight was stopped at the contact (what would have followed is unknown)",
                "official_rule": "SR above is always scored as if the flight ended at the first prohibited contact: a contact "
                                 "fails the episode, and nothing after it changes the result",
                "benchmark_level": "CR = collided episodes / valid episodes; collision gap = SR_without_collision_rule - SR "
                                   "(over the episodes where both are known)",
            },
        },
        "reference": {"path_m": round(ref_len, 1), "duration_s": dur, "frames": len(et), "actions": len(acts),
                      "stop_xyz": [round(v, 2) for v in geo.xyz(last)],
                      "files": "reference/trajectory.mp4, reference/action.json, reference/telemetry.json, reference/collisions.json",
                      "flown": {"work_dir": str(work), "flight_frames": [f0, f1], "flight_t": [t0, t1],
                                "verification": meta["verification"]},
                      "collision": "reference/collisions.json (monitor verified, no prohibited contact)"},
        "instruction_checks": {"paraphrase_similarity_to_canonical": sims, "similarity_limit": SIMILARITY_MAX,
                               "paraphrase_source": pp["source"]},
    }
    (dest / "task.json").write_text(json.dumps(task, indent=1))
    (dest / "goal.json").write_text(json.dumps(goal, indent=1))
    print(f"published {ep_id}: {len(et)} frames ({dur} s), {len(acts)} actions, path {ref_len:.0f} m, "
          f"target in view at start: {view0} -> {dest}")


if __name__ == "__main__":
    main()
