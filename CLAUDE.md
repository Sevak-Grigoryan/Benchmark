# CLAUDE.md: VLN X500 benchmark (final codebase, since 2026-11-19)

Answer the user in the language they write in. They often write in Armenian.

- **Code:** this folder. See README.md for the layout.
- **Data:** `/mnt/weka/sgrigoryan/Benchmark`.
- **Raw flights:** `/mnt/weka/sgrigoryan/Benchmark_runs`.
- **Old versions:** archived in `/mnt/weka/sgrigoryan/_archive_benchmark_old`. Never delete them.
- `/mnt/weka/sgrigoryan/dataset_codebase` (the X500 dataset) is a separate project. Do not change it. Nothing here uses it: PX4, Pegasus, assets and every tool are inside this folder.

## Decisions made by the user (keep them)

- **Collisions:** decided only by the simulator.
  - PhysX contact impulse ≠ 0 means a touch (`collision/contact_monitor.py`, collision-v2).
  - The first touch ends the flight and fails the episode. There are no thresholds of ours.
- **Flight failure:** decided only by PX4 (HEARTBEAT system_status CRITICAL / EMERGENCY / FLIGHT_TERMINATION). No tilt, height or speed limits of ours.
- **Lead's corrections (2026-11-19):**
  - the success distance is 3D (drone to the target's box);
  - instructions use left / right / ahead / behind relative to the start heading, never compass words.
- **Episode layout:** task.json, goal.json (hidden), reference/{action,telemetry,collisions}.json + trajectory.mp4, results/.
- **Each episode has its own goal.json success conditions.** New task types (e.g. orbit) add their own checks in evaluate.py.
- **The user's three principles (2026-11-19):**
  1. tasks must be unambiguous: a target unique in the scene; distances say what they are measured from; land = "land about
     D m from <structure>" (no coloured pads, no unmarked clearings);
  2. PhysX must see everything: coverage gate (no flight if a visible prim has no collider), route check against every
     static obstacle;
  3. the drone never travels watching the ground or sideways: camera ahead, it must find the target
     (`simulation/flight_quality.py`).
- **Orbit wording:** both forms mixed ("clockwise seen from above" and "keeping it on your right").
- **Tolerance:** 5 m for every task type.
- **Train/test split:** left for the end, by the user. Recommended by scene.
- The user's commits never get a Co-Authored-By Claude trailer.

## Cluster

- The login node has 4 GB / 2 CPU. A Forest flight needs one H100 and 160 GB, so use Slurm `research`.
- Bad GPUs exit 76 and get resubmitted. Add new ones to `jobs/bad_gpus.txt`.
- PX4 vehicle ids must stay in 0..9: v1.14 gives unique MAVLink ports only to those.
- Do not edit `jobs/make_episode.sh` while a driver runs.
