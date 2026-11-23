# Official documentation behind collision detection and safety metrics (v0.3b, 2026-11-14)

This file maps each item the benchmark measures to:
- the component that produces it (PhysX, Isaac Sim or PX4);
- the official documentation for it;
- where our code reads it.

**What is ours:** our code only reads, records and scores these values. It never decides a collision, and it sets no tilt, height or speed limit.

**What is not ours:** detection is done by NVIDIA PhysX. Flight failure is what PX4 itself reports.

---

## 1. Collision detection — NVIDIA PhysX (via Isaac Sim / omni.physx)

| Used for | Official documentation | In our code |
|---|---|---|
| Contact reports per physics step (240 Hz) for every drone rigid body; threshold 0 so every contact is reported | Omniverse dev guide, *Rigid Body Physics → contact reports*: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html · `PhysxContactReportAPI` schema: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_contact_report_a_p_i.html | `contact_monitor.py` (`PhysxContactReportAPI.Apply`, `subscribe_contact_report_events`) |
| **The collision rule:** a contact counts only if PhysX applied an impulse at a contact point; contacts with zero impulse are "proximity" | PhysX 5.4.1 API, `PxContactPairPoint` (position, normal, **separation**, **impulse**): https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxContactPairPoint.html · `PxContactPair`: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxContactPair.html | `contact_monitor.py`: `touch = any(impulse != 0)` |
| Contact offset (4 cm), rest offset (0), swept CCD so fast motion cannot tunnel through thin objects | PhysX 5.4 *Advanced Collision Detection* (contact offset, rest offset, CCD): https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/AdvancedCollisionDetection.html | `contact_monitor.py` (offsets), `collision_proxies.enable_ccd` |
| **Peak contact force** = impulse / physics step | `PxContactPairPoint::impulse` — "the impulse applied at the contact point, in world space. Divide by the simulation time step to get a force value" (PhysX API reference, same page as above; wording from https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/apireference/files/structPxContactPairPoint.html) | `evaluate.py`: `peak_contact_force_n = total_impulse_ns × 240` |
| **Clearance** = distance from the drone to the nearest collider, measured by PhysX scene queries (sphere overlap, radius bisected to 1 cm, range 30 m) | PhysX 5.4 *Scene Queries* (overlap queries): https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/SceneQueries.html · omni.physx sphere-overlap node (same query): https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/_build/ogn/docs/omni.physx.graph/OgnPhysXSceneQueryOverlapSphereAny.html | `flight_runner.py` → `clearance()`; `evaluate.py` → `min_clearance_m`, `min_clearance_non_terrain_m` |

**Note:** I could not verify the exact Python signature of `get_physx_scene_query_interface().overlap_sphere` from the docs on the login node. The call is wrapped so that a mismatch can never stop a flight: it records `null` and the error. The first new flight (jobs 386176 / 386175) will confirm it.

### Isaac Sim Contact Sensor — read, deliberately NOT used

- Docs (Isaac Sim 5.1): https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_physics_contact.html
- It is built on the same PhysX contact report, but adds force thresholds that someone has to choose and time-averaging.
- We read the raw PhysX report instead, so no threshold of ours enters.

## 2. Drone motion and impact speed — Isaac Sim rigid body

| Used for | Source | In our code |
|---|---|---|
| **Impact speed**: the drone's linear velocity at the physics step of the first prohibited contact | Isaac Sim / PhysX rigid-body state, read through Pegasus (`drone.state.linear_velocity`) | `flight_runner.py` passes it to the monitor, which stores `drone_velocity_mps`; `evaluate.py` gives `impact_speed_mps` |

## 3. Flight failure (replaces our 60° tilt / 2 m height limits) — PX4 v1.14.3

| Used for | Official documentation | In our code |
|---|---|---|
| PX4's **Failure Detector** (attitude failure: roll/pitch > `FD_FAIL_R/P` for `FD_FAIL_*_TTRI`) and the failsafes | PX4 User Guide v1.14, *Safety Configuration*: https://docs.px4.io/v1.14/en/config/safety.html#failure-detector (local copy: `Benchmark_collision/docs/px4_failure_detector/px4_docs_v1.14_config_safety.md`) | — (PX4 decides) |
| **Flight termination** (motors off, no recovery) | PX4 User Guide v1.14, *Flight Termination*: https://docs.px4.io/v1.14/en/advanced_config/flight_termination.html (local copy in the same folder) | Warehouse `flight_runner.py` cuts the motors when PX4 reports `FLIGHT_TERMINATION` (PX4 SITL stops sending actuator commands; Pegasus kept the last ones) |
| How PX4 reports its state: **HEARTBEAT.system_status** (`MAV_STATE`: ACTIVE, CRITICAL, EMERGENCY, FLIGHT_TERMINATION) | MAVLink common messages, `HEARTBEAT`: https://mavlink.io/en/messages/common.html#HEARTBEAT · `MAV_STATE`: https://mavlink.io/en/messages/common.html#MAV_STATE | `flight_runner.py` records `px4.system_status` every frame plus `px4_events.json`; `evaluate.py` fails the flight on CRITICAL / EMERGENCY / FLIGHT_TERMINATION |
| PX4 source of the detector (to read the exact behaviour) | PX4-Autopilot v1.14.3 `src/modules/commander/failure_detector/` (local copies: `FailureDetector.cpp`, `failure_detector_params.c` in the same folder) | — |

## 4. What remains a benchmark choice (task definition, not safety)

These numbers define **the task**. They don't limit how the agent flies:
- the stand-off distance D and its tolerance τ = max(10 m, 0.3·D);
- the target-in-frame margin (10 %);
- the time limit (2 × the expert's time, rounded up to 10 s).

Route, speed, height, tilt and hover time are all the agent's own decision.

## 5. Success / navigation metrics — standard VLN definitions

| Metric | Definition here | Source |
|---|---|---|
| **SR** (success rate) | mean of S over episodes | Anderson et al., *On Evaluation of Embodied Navigation Agents*, 2018 — https://arxiv.org/abs/1807.06757 |
| **SPL** | S · L_ref / max(L, L_ref) | same paper (SPL is defined there) |
| **NE** (navigation error) | \|d_T − D\| at STOP | Anderson et al., *Vision-and-Language Navigation* (R2R), CVPR 2018 — https://arxiv.org/abs/1711.07280 |
| **OSR** (oracle success) | the goal state was reached at any time during the flight | R2R paper, same as above |
| Aerial VLN setting (drone, continuous actions) | — | Liu et al., *AerialVLN*, ICCV 2023 — https://arxiv.org/abs/2308.06735 |

**Our change to the standard definitions:**
- "goal reached" means being at stand-off distance D ± τ with the target in frame, not "within 3 m of a point".
- S also requires two things: no PhysX prohibited contact (section 1) and no PX4 failure state (section 3).

## 6. How success and collision metrics depend on each other

The dependency goes **one way only**:
- **Success uses the collision result.** A prohibited contact means S = 0. The flight also stops there, which affects OSR, NE and SPL.
- **Collision metrics never use success.** These come straight from PhysX and Isaac Sim for every flight, whether it succeeded or not:
  - collided, time to first collision, part, impulse, force, impact speed, clearance;
  - **CR** = the share of episodes with a collision.
- **SR without the collision rule** (record-only flights only) removes the dependency on purpose. Comparing it with SR gives the **collision gap**: how many successes are lost only because of collisions.
