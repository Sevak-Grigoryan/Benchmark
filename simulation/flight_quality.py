"""Quality rules every published expert flight must meet (checked by jobs/make_batch.py before an episode is kept).

  camera_ahead   while the drone moves horizontally (>= 1.5 m/s) the camera looks where it flies: its heading within 60 deg
                 of the direction of travel and pitched at most 25 deg down. The drone has to find the target with a
                 forward view; it never travels staring at the ground or sideways. Vertical moves (take-off, landing,
                 settling above a target) are not travel and are not judged.
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


def camera_ahead(tele):
    mov = bad = 0
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
            bad += 1
            worst = worst or {"frame": f["frame"], "yaw_off_travel_deg": round(off, 1), "pitch_deg": round(pitch, 1)}
    frac = bad / mov if mov else 0.0
    return {"ok": frac <= MAX_BAD_FRACTION, "moving_frames": mov, "frames_not_looking_ahead": bad,
            "fraction": round(frac, 4), "first_bad": worst,
            "rule": f"moving >= {MOVING_MPS} m/s: camera within {MAX_YAW_OFF_DEG:.0f} deg of travel, pitch <= {MAX_PITCH_DEG:.0f} deg "
                    f"down, in all but {MAX_BAD_FRACTION:.0%} of those frames"}


def check(sim_dir):
    tele = json.loads((Path(sim_dir) / "reference/telemetry.json").read_text())
    return {"camera_ahead": camera_ahead(tele)}


if __name__ == "__main__":
    r = check(sys.argv[1])
    print(json.dumps(r, indent=1))
    sys.exit(0 if all(v["ok"] for v in r.values()) else 1)
