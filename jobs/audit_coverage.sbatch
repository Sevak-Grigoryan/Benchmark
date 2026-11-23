#!/bin/bash
# Collision-coverage audit of one benchmark scene, no flight: builds the scene exactly as a flight job does, adds every
# collider (static pass, vegetation proxies, exact rock / cliff meshes) and writes
#   <out>/coverage_gate.json     every PointInstancer and how it is covered, every visible prim without a collider ("gaps")
#   <out>/world_obstacles.json   the planner's static obstacles (x, y, r, top_z, what)
#   sbatch --export=ALL jobs/audit_coverage.sbatch <scene> <mission> <out dir>
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
set -uo pipefail
SCENE="$1"; MISSION="$2"; OUT="$3"
source "${BENCH_CODE:?export BENCH_CODE}/jobs/config.sh"
B=$BENCH_CODE
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
TMPD="$OUT/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache" HOME="$TMPD/home" CUDA_CACHE_PATH="$TMPD/cuda-cache"
mkdir -p "$HOME"
cd $B/environments/common/sim
BENCH_AUDIT_ONLY=1 X500_NO_BAKE=1 python $B/simulation/flight_runner.py --env Forest --scene "$SCENE" --mission "$MISSION" \
  --out "$OUT/flight" --vehicle-id 0
echo "rc=$?"
