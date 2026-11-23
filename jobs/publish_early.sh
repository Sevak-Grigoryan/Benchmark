#!/bin/bash
# Optional login-node helper next to make_episode.sh: publish the data files as soon as the flight is verified, the video later.
#   nohup bash jobs/publish_early.sh <forest easy v2 idx> <Simulation_N> <make_episode.sh pid> >> logs/publish_early_<idx>.log 2>&1 &
# 1. As soon as the expert flight is OK, publish the episode WITHOUT the video (publish_episode.py --no-video:
#    task.json, goal.json, reference/{action,telemetry,collisions}.json), then score it and run the checker self-test.
# 2. After the driver has finished (slices rendered), add reference/trajectory.mp4 (publish_episode.py's own
#    "published earlier with --no-video" path: it only stitches the video and checks the frame count).
# make_episode.sh then sees task.json and skips its own publish, but still re-scores and self-tests.
set -u
IDX=${1:?idx}; SIM=${2:?Simulation_N}; DRIVER=${3:?driver pid}
source "$(dirname "$(readlink -f "$0")")/config.sh"
B=$BENCH_CODE
EP=vlncol_forest_$(printf %03d "$IDX")
WORK=$BENCH_RUNS/$EP
DEST=$BENCH_DATA/Easy/Forest/$SIM
log() { echo "$(date '+%F %T') $*"; }

score() {
  python3 $B/evaluation/evaluate.py "$DEST" --telemetry "$DEST/reference/telemetry.json" --actions "$DEST/reference/action.json" \
    -o "$DEST/results/reference_score.json" | grep -E '"success"|"episode_end"|"navigation_error_m"|"collided"'
  python3 $B/evaluation/evaluator_selftest.py "$DEST" -o "$DEST/results/checker_selftest.json" | grep -c PASS | sed 's/^/    self-test PASS cases: /'
}

log "waiting for $WORK/flight/FLIGHT_OK (driver $DRIVER)"
while [ ! -f "$WORK/flight/FLIGHT_OK" ] && kill -0 "$DRIVER" 2>/dev/null; do sleep 60; done
if [ -f "$WORK/flight/FLIGHT_OK" ] && [ ! -f "$DEST/task.json" ]; then
  sleep 10   # FLIGHT_OK is the last file the runner writes; give the filesystem a moment
  if python3 $B/publishing/publish_episode.py --work "$WORK" --dest "$DEST" --no-video; then
    log "published WITHOUT video -> $DEST"; score
  else
    log "PUBLISH (no video) FAILED"
  fi
fi

while kill -0 "$DRIVER" 2>/dev/null; do sleep 60; done
if [ -f "$DEST/task.json" ] && [ ! -f "$DEST/reference/trajectory.mp4" ]; then
  python3 $B/publishing/publish_episode.py --work "$WORK" --dest "$DEST" && log "video added" || log "ADD VIDEO FAILED (slices missing?)"
fi
ls -la "$DEST" "$DEST/reference" 2>/dev/null
log "finished"
