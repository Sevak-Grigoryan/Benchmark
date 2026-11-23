"""Collect GPU / Isaac Sim failures in ONE folder, so we can later see why a GPU (or a node) fails.

Every incident gets its own sub-folder in $GPU_INCIDENTS (jobs/config.sh; default $BENCH_RUNS/_gpu_incidents):
    <time>_<node>_<gpu uuid 8>_job<id>_<stage>_att<n>/
        incident.json         node, GPU uuid, driver, job, stage, exit code, kind, where it stopped, error counts
        log_excerpt.txt       first 150 lines + every distinct error line (max 400) + crash stack + last 150 lines
        nvidia_smi_start.txt  `nvidia-smi -q` of the job's GPU at job start (when the job script took one)
        nvidia_smi_now.txt    `nvidia-smi -q` right after the failure (only when recorded inside the job)
and one line in $GPU_INCIDENTS/INDEX.tsv (one line per incident, easy to sort / count by GPU or node).

    python3 tools/gpu_incident.py record --log L --rc RC --stage flight|render --job J --node N --gpu UUID [--attempt A]
        called by jobs/gpu_fly_and_render.sbatch after every Isaac Sim run; records only if the run failed (rc != 0)
        or its log shows a GPU problem (no device / device lost / CUDA error), even when rc == 0
    python3 tools/gpu_incident.py scan <dir> ...      back-fill: every *_<job>_<attempt>.log under the dirs
    python3 tools/gpu_incident.py summary             incidents per GPU and per node
Logs are only read (streamed: a hung render can write a 7 GB log), never moved or changed.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

B = Path(__file__).resolve().parents[1]

SIGNATURES = {   # kind -> regex; checked on every line
    "no_device": re.compile(r"No device could be created|Failed to create any GPU devices|cudaErrorNoDevice|"
                            r"no CUDA-capable device|No usable CUDA device|GPU Foundation is not initialized|"
                            r"no suitable CUDA GPU was found"),
    "device_lost": re.compile(r"DEVICE_LOST|VK_ERROR_DEVICE_LOST|GPU crash|Xid|fallen off the bus"),
    "cuda_error": re.compile(r"CUDA error(?! 100)|CUDA_ERROR|cudaError(?!NoDevice)"),
    "out_of_memory": re.compile(r"^(?!Reasons for the failure).*(out of memory|OUT_OF_MEMORY|bad_alloc)", re.I),
    "segfault": re.compile(r"Segmentation fault|Fatal Python error|SIGSEGV|core dumped"),
    "abort": re.compile(r"Aborted|SIGABRT|terminate called"),
}
NOISE = re.compile(r"getMaterialFromInternalFaceIndex|Double sided attribute|carb\.audio")
DRIVER = re.compile(r"Driver Version: (\S+)")
MILESTONES = [("kit started", "Simulation App Startup Complete"), ("scene built", "extra build:"),
              ("coverage gate", "coverage gate:"), ("contact monitor", "contact monitor on"),
              ("PX4 armed", "Armed by external command"), ("recording", "recording starts"),
              ("flight result", "[x500] RESULT"), ("render booted", "[slice] booted")]
RC_KIND = {134: "abort", 139: "segfault", 137: "killed (SIGKILL / out of memory)", 143: "killed (SIGTERM / scancel)"}


def root():
    r = os.environ.get("GPU_INCIDENTS")
    if not r:
        out = subprocess.run(["bash", "-c", f'source "{B}/jobs/config.sh"; echo "$GPU_INCIDENTS"'], capture_output=True, text=True)
        r = out.stdout.strip()
    return Path(r)


def read_log(path, head_n=150, tail_n=150, max_errors=400):
    """Stream the log once -> facts (works on multi-GB logs)."""
    head, errors, seen = [], [], set()
    counts = collections.Counter()
    first = {}
    reached, driver, stack, in_stack, last_x500 = [], None, [], False, None
    n = 0
    with open(path, errors="replace") as f:
        for line in f:
            n += 1
            line = line.rstrip("\n")
            if n <= head_n:
                head.append(line)
            if driver is None and (m := DRIVER.search(line)):
                driver = m.group(1)
            for name, key in MILESTONES:
                if name not in reached and key in line:
                    reached.append(name)
            if line.startswith("[x500]"):
                last_x500 = line[:300]
            if line.startswith("Current thread") or line.startswith("Fatal Python error"):
                in_stack, stack = True, [line]
                continue
            if in_stack:
                if line.strip() and len(stack) < 40:
                    stack.append(line[:300])
                else:
                    in_stack = False
            if NOISE.search(line):
                continue
            for kind, rx in SIGNATURES.items():
                if rx.search(line):
                    counts[kind] += 1
                    first.setdefault(kind, line[:300])
                    key = re.sub(r"^\S+Z \[[\d,]+ms\] ", "", line)[:200]
                    if key not in seen and len(errors) < max_errors:
                        seen.add(key)
                        errors.append(line[:400])
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 400_000))
        tail = f.read().decode(errors="replace").splitlines()[-tail_n:]
    return dict(lines=n, size_bytes=size, head=head, tail=tail, errors=errors, counts=dict(counts), first=first,
                reached=reached, driver=driver, stack=stack, last_x500=last_x500)


def classify(rc, facts, hang):
    kinds = []
    if hang:
        kinds.append("hang (no progress, killed)")
    if facts["counts"].get("no_device") or facts["driver"] == "0":
        kinds.append("gpu_no_device")
    if facts["counts"].get("device_lost"):
        kinds.append("gpu_device_lost")
    if facts["counts"].get("out_of_memory"):
        kinds.append("out_of_memory")
    if rc in RC_KIND:
        kinds.append(RC_KIND[rc])
    elif facts["counts"].get("segfault"):
        kinds.append("segfault")
    elif rc not in (None, 0):
        kinds.append(f"exit {rc}")
    if facts["counts"].get("cuda_error") and not kinds:
        kinds.append("cuda_error")
    if kinds and "kit started" not in facts["reached"] and not hang:
        kinds.append("during Isaac Sim start-up")
    return kinds


def nvsmi(gpu):
    try:
        args = ["nvidia-smi", "-q"] + (["-i", f"GPU-{gpu}" if not gpu.startswith("GPU-") else gpu] if gpu else [])
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}"


def copy_kit_files(log, job, d, max_log_bytes=30_000_000):
    """Copy the NVIDIA GPU crash dumps (*.nv-gpudmp, written by Isaac Sim on a Vulkan GPU crash; open with Nsight
    Graphics / Nsight Aftermath) and Isaac Sim's own kit logs of this job's attempt into the incident folder."""
    import shutil
    t0 = Path(log).stat().st_mtime
    lo = t0 - 6 * 3600
    m = re.match(r"(.*_)(\d)\.log$", Path(log).name)
    if m and int(m.group(2)) > 1:                       # attempt n starts after attempt n-1's log was last written
        prev = Path(log).with_name(f"{m.group(1)}{int(m.group(2)) - 1}.log")
        if prev.exists():
            lo = prev.stat().st_mtime
    copied = []
    for kit in list(Path(log).parent.glob(f"kit_*_{job}")) + list(Path(log).parent.glob(f"slices/kit_*_{job}")):
        for f in kit.glob("kit-cache/logs/Kit/*/*/*"):
            try:
                st = f.stat()
            except OSError:
                continue
            # same attempt: written after the previous attempt ended and by this attempt's last log write (+10 s)
            if not (f.suffix in (".nv-gpudmp", ".log") and lo < st.st_mtime <= t0 + 10):
                continue
            if f.suffix == ".log" and st.st_size > max_log_bytes:
                continue
            dst = d / "kit" / f.name
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst)
                copied.append(f.name)
    return copied


def record(log, rc, stage, job, node, gpu, attempt=None, hang=False, smi_start=None, in_job=False, note=""):
    log = Path(log)
    if not log.exists():
        return None
    facts = read_log(log)
    kinds = classify(rc, facts, hang)
    if not kinds:
        return None
    R = root()
    R.mkdir(parents=True, exist_ok=True)
    when = time.strftime("%Y%m%d-%H%M%S", time.localtime(log.stat().st_mtime))
    name = f"{when}_{node}_{(gpu or 'unknown')[:8]}_job{job}_{stage}" + (f"_att{attempt}" if attempt else "")
    d = R / name
    if d.exists():
        return d                                  # already recorded (scan run twice)
    d.mkdir()
    inc = {"node": node, "gpu_uuid": gpu, "driver_version": facts["driver"], "job": job, "attempt": attempt,
           "stage": stage, "exit_code": rc, "kind": kinds, "note": note,
           "log": str(log), "log_lines": facts["lines"], "log_bytes": facts["size_bytes"],
           "log_last_modified": time.strftime("%F %T %z", time.localtime(log.stat().st_mtime)),
           "reached": facts["reached"], "stopped_after": facts["reached"][-1] if facts["reached"] else "nothing",
           "last_x500_line": facts["last_x500"], "error_counts": facts["counts"], "first_error": facts["first"],
           "recorded": time.strftime("%F %T %z"), "recorded_in_job": in_job}
    (d / "incident.json").write_text(json.dumps(inc, indent=1))
    parts = ["== first 150 lines", *facts["head"], "", f"== distinct error lines ({len(facts['errors'])})", *facts["errors"], ""]
    if facts["stack"]:
        parts += ["== crash stack", *facts["stack"], ""]
    parts += ["== last 150 lines", *facts["tail"]]
    (d / "log_excerpt.txt").write_text("\n".join(parts) + "\n")
    if smi_start and Path(smi_start).exists():
        (d / "nvidia_smi_start.txt").write_text(Path(smi_start).read_text(errors="replace"))
    if in_job:
        (d / "nvidia_smi_now.txt").write_text(nvsmi(gpu))
    inc["copied"] = copy_kit_files(log, job, d)
    (d / "incident.json").write_text(json.dumps(inc, indent=1))
    idx = R / "INDEX.tsv"
    new = not idx.exists()
    with open(idx, "a") as f:
        if new:
            f.write("when\tnode\tgpu\tdriver\tjob\tattempt\tstage\trc\tkind\tstopped_after\tfolder\n")
        f.write("\t".join(str(x) for x in (when, node, gpu, facts["driver"], job, attempt, stage, rc, "; ".join(kinds),
                                          inc["stopped_after"], name)) + "\n")
    return d


LOGNAME = re.compile(r"^(flight_attempt|slice_\d+|debug|video)_(\d+)_(\d)\.log$")


def job_info(job):
    """-> (node, gpu uuid, {attempt: rc}) from the job's Slurm .out (first line 'gpu <node> <uuid>')."""
    node = gpu = None
    rcs = {}
    for out in (B / "logs").glob(f"*_{job}.out"):
        for line in out.read_text(errors="replace").splitlines():
            p = line.split()
            if len(p) >= 3 and p[0] in ("gpu", "BAD") and node is None:
                node, gpu = (p[1], p[2]) if p[0] == "gpu" else (p[2], p[3])
            if m := re.search(r" attempt (\d) rc=(\d+)", line):
                rcs[int(m.group(1))] = int(m.group(2))
    return node, gpu, rcs


def scan(dirs, hang_jobs=()):
    done = 0
    for top in dirs:
        for log in sorted(Path(top).rglob("*.log")):
            m = LOGNAME.match(log.name)
            if not m or "kit_" in str(log.parent.name):
                continue
            kind, job, att = m.group(1), int(m.group(2)), int(m.group(3))
            node, gpu, rcs = job_info(job)
            stage = "flight" if kind == "flight_attempt" else "render"
            rc = rcs.get(att) if stage == "flight" else None
            hang = str(job) in hang_jobs
            d = record(log, rc, stage, job, node or "unknown", gpu, attempt=att, hang=hang, note="back-filled by scan")
            if d:
                done += 1
                print("recorded", d.name)
    print(f"{done} incident(s) in {root()}")


def summary():
    idx = root() / "INDEX.tsv"
    if not idx.exists():
        print("no incidents")
        return
    rows = [l.split("\t") for l in idx.read_text().splitlines()[1:] if l.strip()]
    by_gpu = collections.defaultdict(list)
    for r in rows:
        by_gpu[(r[1], r[2])].append(r)
    print(f"{len(rows)} incidents, {len(by_gpu)} GPUs\n")
    print(f"{'node':8} {'gpu':38} {'n':>3}  kinds")
    for (node, gpu), rs in sorted(by_gpu.items(), key=lambda kv: -len(kv[1])):
        kinds = collections.Counter(r[8] for r in rs)
        print(f"{node:8} {gpu:38} {len(rs):3}  " + "; ".join(f"{k} x{n}" for k, n in kinds.most_common()))
    print()
    for node, n in collections.Counter(r[1] for r in rows).most_common():
        print(f"{node}: {n}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--log", required=True)
    r.add_argument("--rc", type=int, required=True)
    r.add_argument("--stage", required=True)
    r.add_argument("--job", required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--gpu", default="")
    r.add_argument("--attempt", type=int)
    r.add_argument("--smi-start")
    s = sub.add_parser("scan")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--hang-jobs", default="", help="comma-separated job ids that hung and were cancelled")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "record":
        d = record(a.log, a.rc, a.stage, a.job, a.node, a.gpu, a.attempt, smi_start=a.smi_start, in_job=True)
        if d:
            print(f"GPU INCIDENT recorded: {d}")
    elif a.cmd == "scan":
        scan(a.dirs, tuple(x for x in a.hang_jobs.split(",") if x))
    else:
        summary()


if __name__ == "__main__":
    sys.exit(main())
