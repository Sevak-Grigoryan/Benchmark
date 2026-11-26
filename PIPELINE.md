# Benchmark pipeline

There are seven stages, from choosing a scene to scoring an agent.
- Stages 1–5 for the 100 benchmark scenes run with one command, `jobs/make_batch.py` (one scene: `jobs/make_episode.sh`).
- Stage 6 runs the same evaluator on an agent's flight.

Three rules hold for every episode (the user's principles, enforced by code, not by hand):
1. **Unambiguous task**: the target is a thing that is unique in the scene, described by what it looks like; every number
   (distance, radius) says what it is measured from; sides are left / right from the start heading, never compass words.
2. **PhysX sees everything**: every visible object is collidable (or explicitly listed as not solid) before a flight may
   start (coverage gate); the planner keeps the whole route away from every static obstacle.
3. **The drone looks where it flies**: while moving, the camera points along the direction of travel (12 deg down); the
   drone has to find the target with a forward view (`simulation/flight_quality.py` rejects any flight that does not).

```
 STAGE 0  SETUP (once)            conda env + PX4 build + assets
     │
 STAGE 1  DEFINE THE TASK         scene  +  target  +  instruction
     │
 STAGE 2  FLY THE EXPERT          Isaac Sim + PX4 + PhysX            (GPU)
     │                            → Benchmark_runs/<run>/flight/
 STAGE 3  RENDER THE VIDEO        onboard camera frames → mp4        (GPU)
     │                            → Benchmark_runs/<run>/slices/
 STAGE 4  PUBLISH THE EPISODE     cut the task part, write the files (login node)
     │                            → Benchmark/Easy/Forest/Simulation_N/
 STAGE 5  VERIFY THE EPISODE      score the expert + evaluator self-test
     │                            → Simulation_N/results/
 STAGE 6  EVALUATE AN AGENT       agent's flight → evaluate.py → SR / OSR / NE / SPL / collisions
```

---

## Stage 0. Setup (once)

| What | Where |
|---|---|
| Python env with Isaac Sim 5.1, pymavlink, scipy | conda env named in `jobs/config.sh` (default `behavior_h100`) |
| PX4 v1.14.3 SITL build + Pegasus | inside: `environments/common/px4_env` (patches: `environments/common/px4_patches/`) |
| Forest assets | inside: `environments/common/assets/polyhaven-v2` |
| Isaac Sim render settings | inside: `environments/common/isaac_settings` |
| Paths (episodes, raw flights, conda, Slurm) | `jobs/config.sh` |

## Stage 1. Define the task

| Input | File | Content |
|---|---|---|
| Scene | `environments/Forest/easy_v2.py` | scene `easy_v2_<idx>`: the valley plus a props layout (houses, barns, cars, herds) |
| Task (scenes 101..200) | `simulation/benchmark_tasks.py` | one task per scene, type by `(idx - 101) % 4`: **inspect** (stop D = 1..10 m from the target, facing it), **orbit** (circle it once, R m from its centre, facing it: nose and camera on the target the whole circle, since 2026-11-24; direction said as "clockwise / counter-clockwise seen from above"), **flyover** (stop above its centre, camera down), **land** (land D = 5..10 m from a house / barn / lookout tower / cairn). Tolerance 5 m. Target, numbers, start pad, instruction + 3 paraphrases + short form, and the expert's beats. |
| Route check | `simulation/benchmark_tasks.route_clear` | flies the beats with the offline pilot model; every point keeps 3 m (1 m on take-off / landing) from every static obstacle (`collision/collision_shapes.static_obstacles`: trees, rocks, cliffs, landmarks) and prop |
| Route planner (optional) | `simulation/astar_route.py` | only with `BENCH_ROUTE_PLANNER=astar` (`BENCH_ROUTE_AGL=<m>`: preferred height): the transit legs start → target and target → post-roll pad are planned with A* on an occupancy grid built from the same obstacles (2 m cells, 2 m layers, 4.5 m keep-out), simplified to a few `goto` beats and then checked by the route check like any plan. Unset: the straight route, plans identical to the Forest Easy batch |
| Instruction (old scenes 1..100) | `publishing/instructions.json` | instruction, 3 paraphrases and a short form per run |

## Stage 2. Fly the expert (GPU)

**Command** (done by `jobs/make_episode.sh`): `jobs/gpu_fly_and_render.sbatch` runs `simulation/flight_runner.py`.

What happens inside one flight:

| Step | Code |
|---|---|
| 1. Build the Forest scene | `environments/common/sim/scene_loader.py` → `environments/Forest/scene/forest_bridge.py`, `forest_scene.py`, `easy_v2.py` |
| 2. Spawn the X500 and start PX4 SITL | `environments/common/sim/x500_drone.py` (Pegasus + PX4 over MAVLink) |
| 3. Prepare collisions | `collision/collision_shapes.py`: rotor discs; tree / rock / log proxy shapes; **exact mesh colliders** for every other instanced object (valley cliffs, rockslide, waterfall rocks, river stones); **colliders for every remaining visible prim** (prop details, river water / banks, distant terrain); CCD. Then the **coverage gate**: any visible prim still without a collider -> no flight (`flight/coverage_gate.json`). `collision/contact_monitor.py`: PhysX contact reports. |
| 4. Fly the plan | `simulation/expert_flight_plan.py` → `simulation/benchmark_tasks.plan` builds the plan on the real stage; `environments/common/sim/pilot.py` + `simulation/stick_pilot.py` fly it with sticks. |
| 5. Every physics step (240 Hz) | the contact monitor judges each contact (impulse ≠ 0 = touch). The first touch in the episode stops the flight. |
| 6. Every video frame (30 Hz) | a telemetry record: pose, velocity, camera, PX4 status, clearance |

The flight has three phases. Only the middle one becomes the episode.

```
 pre-roll                      EPISODE                                  post-roll
 take off, climb, turn    →    start hovering → fly to target →    →   fly to landing pad, land
 (not published)               stop at D, face it, STOP                 (not published)
                               (published + evaluated)
```

**Output:** `Benchmark_runs/vlncol_forest_<idx>/flight/`

| File | Content |
|---|---|
| `telemetry.json` | one record per video frame, for the whole flight |
| `action.json` | stick commands |
| `collision.json` | contact monitor record (phases, contacts, first prohibited touch) |
| `meta.json` | scene, mission, camera intrinsics, verification |
| `px4.ulg`, `px4_events.json`, `statustext.json` | PX4 logs |
| `FLIGHT_OK` | written last, only if the flight is valid |

## Stage 3. Render the video (GPU)

| What | Code | Output |
|---|---|---|
| Re-play the recorded poses in the same scene and render the onboard camera, 900 frames per slice | `environments/common/sim/render_video_slice.py` | `Benchmark_runs/<run>/slices/slice_<frame>.mp4` |

## Stage 4. Publish the episode (login node)

**Command:** `publishing/publish_episode.py --work Benchmark_runs/<run> --dest Benchmark/Easy/Forest/Simulation_N`

| Step | Code |
|---|---|
| Find the episode frames (between pre-roll and post-roll) | `simulation/episode_frames.py` |
| Check: flight OK, collision monitor verified, expert never touched anything, no PX4 failure, instructions valid | `publishing/publish_episode.py` |
| Convert the collision record to episode time | `collision/episode_collisions.py` |
| Write the episode | `publishing/publish_episode.py` |

**Output:** `Benchmark/Easy/Forest/Simulation_N/`
```
task.json          what the agent gets
goal.json          hidden: success rule, collision criteria, metrics
reference/         expert demo: action.json, telemetry.json, trajectory.mp4, collisions.json
results/           (filled in stage 5)
```

## Stage 5. Verify the episode

| What | Code | Output | Must be |
|---|---|---|---|
| Score the expert flight | `evaluation/evaluate.py` | `results/reference_score.json` | success = true, collided = false |
| Camera looks where the drone flies | `simulation/flight_quality.py` | (printed) | ok |
| Evaluator self-test (edited copies: collision, wrong distance, no STOP, …) | `evaluation/evaluator_selftest.py` | `results/checker_selftest.json` | all PASS |

`jobs/make_batch.py` runs stages 2–5 for every scene; an episode that fails any check is moved to
`Benchmark_runs/_rejected/` and the next attempt flies another variant of the scene.

## Tools (offline, login node)

| Tool | What it does |
|---|---|
| `tools/offline_preview.py 101 200` | the plan of every scene without Isaac Sim: task, numbers, instruction, paraphrase similarity (< 0.8), camera-ahead rule |
| `tools/check_flights.py <run dirs>` | flown flights against every static obstacle and the camera-ahead rule (safety net next to PhysX) |
| `jobs/audit_coverage.sbatch <scene> <mission> <out>` | GPU, no flight: the coverage gate report and the static obstacle list (`world_obstacles.json`, copied to `environments/Forest/scene/obstacle_sample/` for the offline tools) |
| `tests/test_task_types.py`, `tests/test_contact_monitor.py` | offline tests of the four success rules and the contact rules |
| `tests/test_astar_route.py [first last agl]` | offline test of the A* planner: self-test, every scene planned at a low height (default 10 m), legs, detour, clearance, camera-ahead rule |
| `tools/offline_pxr_stub/` | stand-in for `pxr` so the offline tools can import the scene code; never used by a GPU job |
| `tools/gpu_incident.py summary` | failures per GPU and per node from `$GPU_INCIDENTS/INDEX.tsv` (see GPU incidents below) |

## GPU incidents (why a GPU failed)

Every GPU job records each failed Isaac Sim run (exit code != 0), and each run whose log shows a GPU problem even with
exit 0 (no device, device lost, CUDA error), in **one folder**: `$GPU_INCIDENTS` (`jobs/config.sh`, default
`Benchmark_runs/_gpu_incidents`). One sub-folder per incident, `<time>_<node>_<gpu uuid>_job<id>_<stage>_att<n>/`:
`incident.json` (node, GPU, driver, exit code, kind, how far the run got), `log_excerpt.txt` (start, every distinct error
line, crash stack, end), `kit/` (Isaac Sim's own log and the NVIDIA GPU crash dump `*.nv-gpudmp`, opened with Nsight
Graphics), `nvidia_smi_start.txt` / `nvidia_smi_now.txt` (`nvidia-smi -q` of the GPU at job start and right after the
failure). `INDEX.tsv` has one line per incident. A GPU that keeps failing goes into `jobs/bad_gpus.txt`; a node whose GPUs
keep failing goes into `EXCLUDE_NODES`. The kernel's Xid errors are only in the node's `dmesg` (admins).
Back-fill from old logs: `python3 tools/gpu_incident.py scan <run dirs> [--hang-jobs <ids>]`.

## Stage 6. Evaluate an agent

```
task.json ──► AGENT ──sticks / STOP──► Isaac Sim + PX4 + PhysX ──► telemetry.json + collisions.json ──► evaluate.py ──► score
                ▲                              │                                                          ▲
                └────── camera image ◄─────────┘                                                     goal.json
```

| Step | Code | Status |
|---|---|---|
| Run the agent in the simulator (same scene, start pose, contact monitor) | agent runner | **not written yet** |
| Score one flight | `evaluation/evaluate.py` (uses `evaluation/camera_geometry.py`) | ready |
| Average over episodes (SR, OSR, NE, SPL, CR) | aggregation script | **not written yet** |

---

## One command for stages 2–5

```bash
cd <this folder>
nohup bash jobs/make_episode.sh <idx> "" Simulation_N >> logs/make_episode_<idx>.log 2>&1 &
```
The driver:
1. submits the GPU job;
2. resubmits it after a bad GPU (exit 76);
3. waits until the flight and every video slice exist;
4. then publishes, scores and self-tests.

## Names

| Name | Meaning |
|---|---|
| `easy_v2_<idx>` | the scene: Forest valley + Easy v2 props layout number `<idx>` |
| `vlncol_forest_<idx>` | the expert flight run in that scene (folder in `Benchmark_runs/`) |
| `Simulation_N` | the published episode (folder in `Benchmark/Easy/Forest/`) |
| pre-roll / episode / post-roll | flight phases; only the episode is published |
| collision-v2 | the collision rule: PhysX impulse ≠ 0 = touch |
