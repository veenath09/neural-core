#!/bin/zsh
# seed-0 simulator RL: full and context_free in parallel
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
INIT=/Users/veenathwickramaarachchi/Documents/SLIIT/final-research/neural-core/runs/routerbench
run() {
  local c=$1
  .venv/bin/python -u -m train.rl --config $c --seed 0 --init $INIT/$c-s0 > $INIT/$c-s0/rl.log 2>&1
  .venv/bin/python -m eval.rl_frontier $INIT/$c-s0/rl >> $INIT/$c-s0/rl.log 2>&1
  echo "done $c exit $?"
}
for c in full context_free; do run $c & done
wait
