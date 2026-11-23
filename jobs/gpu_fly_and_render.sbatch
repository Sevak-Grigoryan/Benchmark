#!/bin/bash
# ONE GPU job of the VLN X500 benchmark (Forest):
# flies the benchmark episode and every collision test one after another (one Isaac Sim process each), then renders the
# episode frames (render_video_slice.py), the opt-in external debug video of the sideways test and one
# video per collision test.
#   sbatch jobs/gpu_fly_and_render.sbatch <scene> <episode mission> <work root> "<test names>"   (make_episode.sh does this)
# Units already done (FLIGHT_DONE / slice mp4 / debug mp4 present) are skipped, so a resubmitted job continues.
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=09:00:00
set -uo pipefail
SCENE="$1"; EPISODE="$2"; ROOT="$3"; TESTS="$4"
# make_episode.sh submits with --export=ALL, so BENCH_CODE (this repository) is set; Slurm runs a copy of this script,
# so its own location cannot be used
source "${BENCH_CODE:?submit through jobs/make_episode.sh or export BENCH_CODE}/jobs/config.sh"
B=$BENCH_CODE
CB=$B/environments
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
TMPD="$ROOT/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache"
export HOME="$TMPD/home" CUDA_CACHE_PATH="$TMPD/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$TMPD/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
cd $CB/common/sim

BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970"
# known-bad GPUs: this codebase's list, plus the list the X500 dataset jobs keep (if present) so both share findings
BAD_GPUS="$BAD_GPUS $(awk '{print $1}' $B/jobs/bad_gpus.txt "$SHARED_BAD_GPUS" 2>/dev/null)"
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ "$(echo "$UUIDS" | grep -c GPU-)" -gt 1 ]; then UUIDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); fi
GPU_UUID=${UUIDS#GPU-}
echo "gpu $(hostname) $GPU_UUID"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU $(hostname) $GPU_UUID -- skipped"; exit 76;; esac; done

# GPU incidents: every failed Isaac Sim run (and every run whose log shows a GPU problem, even with exit 0) is recorded in
# $GPU_INCIDENTS/<time>_<node>_<gpu>_job<id>_<stage>_att<n>/ with a log excerpt, the GPU crash dump and nvidia-smi -q
# taken at job start and right after the failure; INDEX.tsv there has one line per incident (tools/gpu_incident.py)
nvidia-smi -q -i "GPU-$GPU_UUID" > "$TMPD/nvidia_smi_start.txt" 2>&1
incident() {  # incident <log> <rc> <stage> <attempt>
  python3 $B/tools/gpu_incident.py record --log "$1" --rc "$2" --stage "$3" --attempt "$4" --job "$SLURM_JOB_ID" \
    --node "$(hostname)" --gpu "$GPU_UUID" --smi-start "$TMPD/nvidia_smi_start.txt" || true
}

# fly <mission> <dir> <vehicle id>: two attempts against Isaac Sim start-up failures
fly() {
  local m=$1 d=$2 vid=$3 att L rc
  [ -f "$d/flight/FLIGHT_DONE" ] && { echo "$m: already flown ($(cat "$d/flight/FLIGHT_DONE"))"; return 0; }
  mkdir -p "$d"
  for att in 1 2; do
    L="$d/flight_attempt_${SLURM_JOB_ID}_${att}.log"
    X500_NO_BAKE=1 python $B/simulation/flight_runner.py --env Forest --scene "$SCENE" --mission "$m" --out "$d/flight" --vehicle-id "$vid" > "$L" 2>&1
    rc=$?
    pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i $vid " 2>/dev/null
    incident "$L" $rc flight $att
    echo "$m attempt $att rc=$rc: $(grep -h -m1 '\[x500\] RESULT' "$L" | cut -c1-300)"
    if [ -f "$d/flight/FLIGHT_DONE" ] || grep -q "recording starts" "$L"; then return 0; fi
  done
}

# Vehicle id per flight: two flights on the same node with the same id share PX4's ports (TCP 4560+id, UDP 14540+id) and
# read each other's PX4 (2026-11-16: jobs 386175/386176 both used id 0 on gpu07 -> px4_heading_mismatch). PX4 v1.14 SITL
# gives a unique offboard port only to instances 0..9 (px4-rc.mavlink sends every instance > 9 to 14549), so ids stay in
# 0..9: start at job id % 10 and take the first id whose ports are all free on this node.
pick_vid() {  # a port is free if this node lets us bind it right now (ss is not installed on the nodes)
  python3 - "$SLURM_JOB_ID" <<'PY'
import socket, sys
job = int(sys.argv[1])
def free(port, kind):
    s = socket.socket(socket.AF_INET, kind)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()
for k in range(10):
    i = (job + k) % 10
    if free(4560 + i, socket.SOCK_STREAM) and all(free(p + i, socket.SOCK_DGRAM) for p in (14540, 14580, 18570)):
        print(i)
        break
else:
    print(job % 10)
PY
}
fly "$EPISODE" "$ROOT/$EPISODE" "$(pick_vid)"
for n in $TESTS; do fly "colltest_$n" "$ROOT/colltest_$n" "$(pick_vid)"; done

# episode frames -> slices (RGB, same scene), ONE AT A TIME: parallel Forest renders ran out of the 160 GB (job 390131)
EW="$ROOT/$EPISODE"
if [ -f "$EW/flight/FLIGHT_OK" ]; then
  read F0 F1 < <(python3 $B/simulation/episode_frames.py "$EW/flight")
  echo "episode window [$F0, $F1)"
  for ((s = F0; s < F1; s += 900)); do
    e=$((s + 900 < F1 ? s + 900 : F1)); out="$EW/slices/slice_$(printf %06d $s).mp4"
    [ -f "$out" ] && continue
    ( for att in 1 2; do
        L="$EW/slice_${s}_${SLURM_JOB_ID}_${att}.log"
        XDG_CACHE_HOME="$TMPD/cache_$s" OPTIX_CACHE_PATH="$TMPD/optix_$s" \
          python render_video_slice.py --env Forest --telemetry "$EW/flight/telemetry.json" --start "$s" --end "$e" --out "$out" \
          --mission "$SCENE" > "$L" 2>&1
        rc=$?
        incident "$L" $rc render $att
        [ $rc -eq 0 ] && break
        grep -q "\[slice\] booted" "$L" && break
      done )
  done
fi

# opt-in debug video (sideways test)
DW="$ROOT/colltest_debug_sideways"
if [ -f "$DW/flight/FLIGHT_DONE" ] && [ ! -f "$DW/debug/debug_external_onboard.mp4" ]; then
  for att in 1 2; do
    L="$DW/debug_${SLURM_JOB_ID}_${att}.log"
    python $B/simulation/debug_video.py --env Forest --scene "$SCENE" --flight "$DW/flight" --out "$DW/debug" > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
fi

# one video per collision test: last 12 s of the test's episode up to the contact, external camera following from the side
# (re-created fixtures, contact marked) next to the onboard view; every 2nd frame (15 fps), last frame held 2 s
for n in $TESTS; do
  [ "$n" = debug_sideways ] && continue
  TW="$ROOT/colltest_$n"
  [ -f "$TW/flight/FLIGHT_DONE" ] && [ -f "$TW/flight/telemetry.json" ] || continue
  [ -f "$TW/video/collision_test.mp4" ] && continue
  for att in 1 2; do
    L="$TW/video_${SLURM_JOB_ID}_${att}.log"
    python $B/simulation/debug_video.py --env Forest --scene "$SCENE" --flight "$TW/flight" --out "$TW/video" --camera side \
      --max-seconds 12 --stride 2 --hold-s 2 --bank-samples 0 --name collision_test > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
  echo "colltest_$n video: $(ls "$TW/video/collision_test.mp4" 2>/dev/null || echo MISSING)"
done
echo "job done"
