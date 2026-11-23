# All machine-specific paths of the pipeline, in one place. Change these when the code, the data or the cluster change.
# Each can also be overridden from the environment (export BENCH_DATA=... before running jobs/make_episode.sh).

BENCH_CODE=${BENCH_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # this repository
BENCH_DATA=${BENCH_DATA:-/mnt/weka/sgrigoryan/Benchmark}                     # published episodes (Simulation_N)
BENCH_RUNS=${BENCH_RUNS:-/mnt/weka/sgrigoryan/Benchmark_runs}                # raw flights + render slices
CONDA_ROOT=${CONDA_ROOT:-/mnt/weka/shared-cache/miniforge3}                  # conda install (also has ffmpeg / ffprobe)
CONDA_ENV=${CONDA_ENV:-behavior_h100}                                        # env with Isaac Sim 5.1 + pymavlink + scipy
EXTRA_LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:-$CONDA_ROOT/envs/flowr_root/lib}
SLURM_PARTITION=${SLURM_PARTITION:-research}
GPU_INCIDENTS=${GPU_INCIDENTS:-$BENCH_RUNS/_gpu_incidents}               # every GPU / Isaac Sim failure, one folder each (tools/gpu_incident.py)
EXCLUDE_NODES=${EXCLUDE_NODES:-gpu01,gpu02,gpu06}                                  # gpu01,gpu02 drained after the 2026-11-18 GPU faults, gpu06 after 3 GPU failures on 2026-11-19; single bad GPUs are caught by jobs/bad_gpus.txt
SHARED_BAD_GPUS=${SHARED_BAD_GPUS:-}                                          # optional extra list of bad GPU uuids (jobs/bad_gpus.txt is always read)
export BENCH_CODE BENCH_DATA BENCH_RUNS CONDA_ROOT CONDA_ENV GPU_INCIDENTS
