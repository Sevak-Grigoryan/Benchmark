"""Offline preview of the benchmark plans of scenes easy_v2_<first> .. easy_v2_<last> (no Isaac Sim needed).

For every scene: the task the planner picks (type, target, distance / radius / direction), the trial it needed, the
instruction and how similar each paraphrase is to it (must stay < 0.8), and the camera-ahead rule
(simulation/flight_quality.py) on the episode flown by the offline pilot model.
    python3 tools/offline_preview.py 101 200            -> one line per scene + totals; exit 1 if a scene fails
Offline, the target boxes come from the layout circles and the static obstacles from
environments/Forest/scene/obstacle_sample/world_obstacles.json (written by jobs/audit_coverage.sbatch); inside Isaac Sim the
flight job measures both on the USD stage, so the real plan can differ slightly.
"""
import collections
import difflib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

B = Path(__file__).resolve().parents[1]
E = B / "environments"
sys.path[:0] = [str(B / "tools/offline_pxr_stub"), str(E / "common/vln_simulation"), str(E / "Forest"), str(E / "Forest/scene"),
                str(E / "common/sim"), str(B / "simulation"), str(B / "evaluation"), str(B / "collision")]
import benchmark_tasks as bt  # noqa: E402
import easy_v2  # noqa: E402
import flight_quality as fq  # noqa: E402
import forest_scene as fs  # noqa: E402
import missions  # noqa: E402
import natural_valley as nv  # noqa: E402
import pilot  # noqa: E402

missions.apply_pads(fs)


def ground(x, y):
    return fs.visible_ground(nv, x, y, easy_v2.SEED)


def episode_camera(m):
    """flight_quality.camera_ahead on the episode part of the offline pilot flight."""
    fr, _ = pilot.simulate_offline(m["beats"], ground, m["start_xy"], math.radians(m["yaw_deg"]))
    i0, i1 = m["bench"]["episode_beats"]
    labels = {b.get("label") for b in m["beats"][i0:i1 + 1]}
    tele = []
    for a, c in zip(fr, fr[1:]):
        if a["beat"] in labels:
            dt = c["t"] - a["t"]
            tele.append({"frame": 0, "velocity": {"vx": (c["x"] - a["x"]) / dt, "vy": (c["y"] - a["y"]) / dt},
                         "camera": {"x": a["x"], "y": a["y"], "z": a["z"], "yaw_deg": math.degrees(a["yaw"]),
                                    "pitch_deg": a["gimbal"]}})
    look = m["bench"]["target"]["point_xyz"] if m["bench"]["task_type"] == "orbit" else None
    return fq.camera_ahead(tele, look)


def main():
    first, last = int(sys.argv[1]), int(sys.argv[2])
    cnt, bad = collections.Counter(), []
    for idx in range(first, last + 1):
        lay = easy_v2.layout(idx, nv, fs)
        try:
            m = bt.plan(idx, SimpleNamespace(layout=lay, ground=ground, bridge=None))
        except Exception as e:  # noqa: BLE001
            bad.append((idx, f"no plan: {e}"))
            continue
        b, ins = m["bench"], m["bench"]["instructions"]
        sim = max(difflib.SequenceMatcher(None, ins["instruction"].lower(), p.lower()).ratio() for p in ins["paraphrases"])
        cam = episode_camera(m)
        p = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in b["params"].items()
             if k in ("standoff_m", "radius_m", "direction", "height_above_top_m")}
        cnt[b["task_type"]] += 1
        if sim >= 0.8:
            bad.append((idx, f"paraphrase similarity {sim:.2f}"))
        if not cam["ok"]:
            bad.append((idx, f"camera not ahead in {cam['fraction']:.0%} of moving frames"))
        print(f"{idx} {b['task_type']:8s} {b['target']['name']:10s} {p} trial {b['trial']} sim {sim:.2f} "
              f"camera_ahead {cam['ok']}  | {ins['instruction']}")
    print(dict(cnt), "problems:", bad or "none")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
