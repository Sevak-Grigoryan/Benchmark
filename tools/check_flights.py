"""Check flown expert flights (raw runs) against the valley's static obstacles and the camera-ahead rule.

    python3 tools/check_flights.py <run dir> [<run dir> ...]      (a run dir holds flight/telemetry.json)
For every flight: the closest the drone came to any static obstacle (vertical cylinders of
environments/Forest/scene/obstacle_sample/world_obstacles.json: every tree, rock, cliff and landmark), whether it was ever
inside one, and simulation/flight_quality.camera_ahead on the episode frames. A safety net next to PhysX: a flight that is
inside an obstacle with no contact reported means the collision model has a hole.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "simulation"))
import flight_quality as fq  # noqa: E402

O = np.array([o[:4] for o in json.loads((B / "environments/Forest/scene/obstacle_sample/world_obstacles.json").read_text())])
WHAT = [o[4] for o in json.loads((B / "environments/Forest/scene/obstacle_sample/world_obstacles.json").read_text())]


def check(run):
    t = json.loads((Path(run) / "flight/telemetry.json").read_text())
    fr = t["frames"] if isinstance(t, dict) else t
    best, inside = (1e9, None, None), []
    for f in fr[::3]:
        x, y, z = f["pose"]["x"], f["pose"]["y"], f["pose"]["z"]
        d = np.hypot(O[:, 0] - x, O[:, 1] - y) - O[:, 2]
        below = z < O[:, 3]
        k = int(np.argmin(np.where(below, d, 1e9)))
        if below[k] and d[k] < best[0]:
            best = (float(d[k]), WHAT[k], f.get("phase"))
        if below[k] and d[k] < 0 and not WHAT[k].startswith(("Nature", "DenseNature")):   # inside a cliff / rock / landmark
            inside.append((f["frame"], WHAT[k]))
    cam = fq.camera_ahead([f for f in fr if f.get("phase") == "episode"])
    return {"closest_m": round(best[0], 1), "closest_to": best[1], "phase": best[2], "inside_static": inside[:3],
            "camera_ahead": cam["ok"], "camera_not_ahead_fraction": cam["fraction"]}


if __name__ == "__main__":
    bad = 0
    for r in sys.argv[1:]:
        res = check(r)
        bad += bool(res["inside_static"]) or not res["camera_ahead"]
        print(Path(r).name, res)
    sys.exit(1 if bad else 0)
