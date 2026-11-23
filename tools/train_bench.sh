#!/usr/bin/env bash
# Measure RL training throughput of Unitree-Go2-Flat for several env counts.
# Usage: tools/train_bench.sh <unitree_rl_mjlab dir> <out dir> <iterations> <num_envs>...
set -u
repo=$(realpath "$1"); out=$(realpath -m "$2"); iters=$3; shift 3
mkdir -p "$out"
cd "$repo" || exit 1

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader > "$out/gpu_idle.csv"
.venv/bin/python -c "import mujoco, mujoco_warp, warp, torch, mjlab, importlib.metadata as m; \
print({'mujoco': mujoco.__version__, 'mujoco_warp': m.version('mujoco-warp'), \
'warp': warp.__version__, 'torch': torch.__version__, 'mjlab': m.version('mjlab'), \
'rsl_rl': m.version('rsl-rl-lib')})" > "$out/versions.txt" 2>/dev/null
git -C "$repo" rev-parse HEAD > "$out/unitree_rl_mjlab_commit.txt"

for n in "$@"; do
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
    -lms 500 > "$out/gpu_$n.csv" &
  sampler=$!
  timeout 1800 .venv/bin/python scripts/train.py Unitree-Go2-Flat \
    --env.scene.num-envs "$n" --agent.max-iterations "$iters" \
    --agent.logger tensorboard --agent.run-name "bench$n" > "$out/train_$n.log" 2>&1
  echo "num_envs=$n exit=$?" | tee -a "$out/exit_codes.txt"
  kill "$sampler"; wait "$sampler" 2>/dev/null
done
