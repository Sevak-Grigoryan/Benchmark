"""Quality rules every published expert flight must meet (checked by jobs/make_batch.py before an episode is kept).

  camera_ahead   while the drone moves horizontally (>= 1.5 m/s) the camera looks where it flies: its heading within 60 deg
                 of the direction of travel and pitched at most 25 deg down. The drone has to find the target with a
                 forward view; it never travels staring at the ground or sideways. Vertical moves (take-off, landing,
                 settling above a target) are not travel and are not judged.
                 Orbit episodes (since 2026-11-24, the user's decision): while circling, the drone faces the target and
                 slides sideways, so a moving frame also passes when the target point is inside the image (90 % of the
                 half field of view). The approach is still judged by the camera-ahead rule: the target must be found.
    python3 simulation/flight_quality.py <Simulation_N folder>    -> prints the result, exit 1 if a rule fails
"""
import json
import math
import sys
from pathlib import Path

MOVING_MPS = 1.5
MAX_YAW_OFF_DEG = 60.0
MAX_PITCH_DEG = 25.0
MAX_BAD_FRACTION = 0.02
HFOV_DEG, VFOV_DEG = 47.169, 27.594   # the onboard camera (task.json observation.rgb.camera_intrinsics)
ON_TARGET_FOV_FRACTION = 0.9


def on_target(cam, look_at):
    """The target point is inside the image: horizontal and vertical angle from the optical axis within 90 % of half FOV."""
    dx, dy, dz = look_at[0] - cam["x"], look_at[1] - cam["y"], look_at[2] - cam["z"]
    yaw_off = abs((math.degrees(math.atan2(dy, dx)) - cam["yaw_deg"] + 180.0) % 360.0 - 180.0)
    down = math.degrees(math.atan2(-dz, max(math.hypot(dx, dy), 0.1)))          # positive = the target is below
    pitch_off = abs(down - (cam.get("pitch_deg") or 0.0))
    return yaw_off <= ON_TARGET_FOV_FRACTION * HFOV_DEG / 2 and pitch_off <= ON_TARGET_FOV_FRACTION * VFOV_DEG / 2


def camera_ahead(tele, look_at=None):
    """look_at: the target point of an orbit episode (a frame that faces it passes too); None for the other task types."""
    mov = bad = on_tgt = 0
    worst = None
    for f in tele:
        v = f["velocity"]
        hs = math.hypot(v["vx"], v["vy"])
        if hs < MOVING_MPS:
            continue
        mov += 1
        off = abs((f["camera"]["yaw_deg"] - math.degrees(math.atan2(v["vy"], v["vx"])) + 180.0) % 360.0 - 180.0)
        pitch = f["camera"].get("pitch_deg") or 0.0
        if off > MAX_YAW_OFF_DEG or pitch > MAX_PITCH_DEG:
            if look_at is not None and on_target(f["camera"], look_at):
                on_tgt += 1
                continue
            bad += 1
            worst = worst or {"frame": f["frame"], "yaw_off_travel_deg": round(off, 1), "pitch_deg": round(pitch, 1)}
    frac = bad / mov if mov else 0.0
    rule = (f"moving >= {MOVING_MPS} m/s: camera within {MAX_YAW_OFF_DEG:.0f} deg of travel, pitch <= {MAX_PITCH_DEG:.0f} deg "
            f"down, in all but {MAX_BAD_FRACTION:.0%} of those frames")
    if look_at is not None:
        rule += " (orbit: a frame with the target inside the image also passes)"
    return {"ok": frac <= MAX_BAD_FRACTION, "moving_frames": mov, "frames_not_looking_ahead": bad,
            "frames_facing_target": on_tgt, "fraction": round(frac, 4), "first_bad": worst, "rule": rule}


def check(sim_dir):
    sim = Path(sim_dir)
    tele = json.loads((sim / "reference/telemetry.json").read_text())
    look_at = None
    goal_f = sim / "goal.json"
    if goal_f.exists():
        goal = json.loads(goal_f.read_text())
        if goal.get("success", {}).get("task_type") == "orbit":
            look_at = goal["target"]["point_xyz"]
    return {"camera_ahead": camera_ahead(tele, look_at)}


if __name__ == "__main__":
    r = check(sys.argv[1])
    print(json.dumps(r, indent=1))
    sys.exit(0 if all(v["ok"] for v in r.values()) else 1)
