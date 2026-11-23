"""Login-node driver for a batch of benchmark episodes (Forest scenes easy_v2_101 .. easy_v2_200 -> Simulation_1 .. 100).

    cd <this repository>
    nohup python3 jobs/make_batch.py --first 101 --last 200 --max-jobs 10 >> logs/make_batch.log 2>&1 &

For every scene, until it is published or out of attempts:
  1. submit jobs/gpu_fly_and_render.sbatch (expert flight + onboard video) with BENCH_TASK_ATTEMPT = the attempt;
     at most --max-jobs GPU jobs at a time; a job that hit a known-bad GPU (exit 76, "BAD GPU") is resubmitted with that
     node excluded, same attempt;
  2. when the flight and every video slice exist: publish $BENCH_DATA/Easy/Forest/Simulation_<idx - 100>, score the expert
     flight (results/reference_score.json) and run the evaluator self-test (results/checker_selftest.json);
  3. the episode is kept only if the expert flight succeeds by the published rule and every self-test case passes.
     Otherwise (collision, PX4 abort, publish refused, expert not successful) the run folder and any published files are
     moved aside (nothing is deleted) and the next attempt flies another variant of the same scene (other target /
     direction / distance), up to --attempts.
State is kept in logs/make_batch_state.json, so the driver can be restarted and continues where it was.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

B = Path(__file__).resolve().parents[1]


def config():
    """The paths of jobs/config.sh, as the shell would see them."""
    keys = ("BENCH_CODE", "BENCH_DATA", "BENCH_RUNS", "SLURM_PARTITION", "EXCLUDE_NODES")
    out = subprocess.run(["bash", "-c", f'source "{B}/jobs/config.sh"; printf "%s\\0" ' + " ".join(f'"${k}"' for k in keys)],
                         capture_output=True, text=True, check=True).stdout
    return dict(zip(keys, out.split("\0")))


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def job_alive(j):
    return bool(sh(["squeue", "-h", "-j", str(j)]).stdout.strip())


def done_units(run, ep):
    """Flight finished and every episode video slice rendered."""
    fl = run / "flight"
    if not (fl / "FLIGHT_DONE").exists():
        return False
    if not (fl / "FLIGHT_OK").exists():
        return True                                             # finished, but not OK: nothing to render
    r = sh([sys.executable, str(B / "simulation/episode_frames.py"), str(fl)])
    if r.returncode:
        return False
    f0, f1 = map(int, r.stdout.split())
    return all((run / "slices" / f"slice_{s:06d}.mp4").exists() for s in range(f0, f1, 900))


def publish_and_check(run, dest):
    """-> (ok, why). Publishes, scores the expert flight and runs the self-test."""
    if not (run / "flight/FLIGHT_OK").exists():
        res = json.loads((run / "flight/meta.json").read_text()).get("verification", {}) if (run / "flight/meta.json").exists() else {}
        return False, f"flight not OK ({res.get('outcome', 'no meta.json')})"
    r = sh([sys.executable, str(B / "publishing/publish_episode.py"), "--work", str(run), "--dest", str(dest)])
    if r.returncode:
        return False, f"publish refused: {(r.stderr or r.stdout).strip().splitlines()[-1][:200]}"
    ref = dest / "reference"
    r = sh([sys.executable, str(B / "evaluation/evaluate.py"), str(dest), "--telemetry", str(ref / "telemetry.json"),
            "--actions", str(ref / "action.json"), "-o", str(dest / "results/reference_score.json")])
    score = json.loads((dest / "results/reference_score.json").read_text()) if r.returncode == 0 else {}
    if not score.get("success"):
        return False, f"expert flight not successful: {score.get('episode_end', r.stderr[-200:])}"
    r = sh([sys.executable, str(B / "simulation/flight_quality.py"), str(dest)])
    if r.returncode:
        return False, f"flight quality: {r.stdout.strip()[-300:]}"
    r = sh([sys.executable, str(B / "evaluation/evaluator_selftest.py"), str(dest), "-o", str(dest / "results/checker_selftest.json")])
    fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
    if r.returncode or fails:
        return False, f"self-test failed: {fails[:2] or r.stderr[-200:]}"
    return True, score["episode_end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=101)
    ap.add_argument("--last", type=int, default=200)
    ap.add_argument("--max-jobs", type=int, default=35, help="the researcher QOS allows 35 GPUs per user")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--sim-offset", type=int, default=100, help="Simulation_<idx - offset>")
    a = ap.parse_args()
    C = config()
    runs, data = Path(C["BENCH_RUNS"]), Path(C["BENCH_DATA"])
    logs = B / "logs"
    state_f = logs / "make_batch_state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {}
    for idx in range(a.first, a.last + 1):
        st.setdefault(str(idx), {"attempt": 0, "job": None, "exclude": C["EXCLUDE_NODES"], "status": "todo", "history": []})
    save = lambda: state_f.write_text(json.dumps(st, indent=1))
    save()
    env = dict(os.environ, **{k: C[k] for k in ("BENCH_CODE", "BENCH_DATA", "BENCH_RUNS")})
    while True:
        running = [k for k, v in st.items() if v["status"] == "running"]
        for k in running:
            v = st[k]
            if job_alive(v["job"]):
                continue
            idx = int(k)
            ep = f"vlncol_forest_{idx:03d}"
            run, dest = runs / ep, data / "Easy/Forest" / f"Simulation_{idx - a.sim_offset}"
            out = logs / f"VLNBench-Forest-{idx:03d}_{v['job']}.out"
            text = out.read_text(errors="ignore") if out.exists() else ""
            if "BAD GPU" in text:
                node = next(l.split()[2] for l in text.splitlines() if l.startswith("BAD GPU"))
                v["exclude"] += f",{node}"
                v["status"] = "todo"
                log(f"{ep}: bad GPU on {node}, resubmit (same attempt {v['attempt']})")
                continue
            if not done_units(run, ep):
                v["status"] = "todo"
                v.setdefault("incomplete", 0)
                v["incomplete"] += 1
                log(f"{ep}: job {v['job']} ended with units missing (#{v['incomplete']}), resubmit")
                if v["incomplete"] >= 3:
                    v["status"], v["incomplete"] = "retry", 0
                continue
            ok, why = publish_and_check(run, dest)
            v["history"].append({"attempt": v["attempt"], "job": v["job"], "ok": ok, "why": why})
            if ok:
                v["status"] = "published"
                log(f"{ep}: PUBLISHED {dest.name} (attempt {v['attempt']}): {why}")
            else:
                v["status"] = "retry"
                log(f"{ep}: attempt {v['attempt']} rejected: {why}")
        for k, v in st.items():
            if v["status"] != "retry":
                continue
            idx = int(k)
            ep = f"vlncol_forest_{idx:03d}"
            run, dest = runs / ep, data / "Easy/Forest" / f"Simulation_{idx - a.sim_offset}"
            aside = runs / "_rejected" / f"{ep}_attempt{v['attempt']}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            if run.exists():
                shutil.move(str(run), str(aside))
            if dest.exists():
                shutil.move(str(dest), str(aside / "published_then_rejected"))
            v["attempt"] += 1
            v["status"] = "todo" if v["attempt"] < a.attempts else "gave_up"
            log(f"{ep}: moved aside to {aside}; " + (f"next attempt {v['attempt']}" if v["status"] == "todo" else "GAVE UP"))
        free = a.max_jobs - sum(v["status"] == "running" for v in st.values())
        for k, v in sorted(st.items(), key=lambda kv: int(kv[0])):
            if free <= 0:
                break
            if v["status"] != "todo":
                continue
            idx = int(k)
            ep, scene = f"vlncol_forest_{idx:03d}", f"easy_v2_{idx:03d}"
            name = f"VLNBench-Forest-{idx:03d}"
            r = sh(["sbatch", "--parsable", "--export=ALL", "-p", C["SLURM_PARTITION"], f"--exclude={v['exclude']}", "-J", name,
                    "-o", str(logs / "%x_%j.out"), "-e", str(logs / "%x_%j.err"), str(B / "jobs/gpu_fly_and_render.sbatch"),
                    scene, ep, str(runs), ""], env=dict(env, BENCH_TASK_ATTEMPT=str(v["attempt"])))
            if r.returncode:
                log(f"{ep}: sbatch failed: {r.stderr.strip()}")
                break
            v["job"], v["status"] = int(r.stdout.strip()), "running"
            free -= 1
            log(f"{ep}: submitted job {v['job']} (attempt {v['attempt']}, exclude {v['exclude']})")
        save()
        counts = {}
        for v in st.values():
            counts[v["status"]] = counts.get(v["status"], 0) + 1
        if all(v["status"] in ("published", "gave_up") for v in st.values()):
            log(f"ALL DONE {counts}")
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
