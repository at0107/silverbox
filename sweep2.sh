#!/bin/bash
# Sweep 2: robustness check. a04, a07, a12 settings from sweep 1, steps=4000, seeds 0,1,2 (9 jobs <= 12 cores)
cd "$(dirname "$0")"   # run from the repo folder
mkdir -p logs
for s in 0 1 2; do
  echo "{\"name\":\"s2_a04_seed$s\",\"act\":\"silu\",\"width\":64,\"depth\":3,\"win\":64,\"steps\":4000,\"seed\":$s}"
  echo "{\"name\":\"s2_a07_seed$s\",\"act\":\"tanh\",\"skip\":true,\"phys_init\":true,\"win\":64,\"steps\":4000,\"seed\":$s}"
  echo "{\"name\":\"s2_a12_seed$s\",\"act\":\"tanh\",\"skip\":true,\"phys_init\":true,\"win\":32,\"win_final\":256,\"steps\":4000,\"seed\":$s}"
done | xargs -P 9 -d '\n' -I{} sh -c './venv/bin/python silverbox_pinn.py '"'"'{}'"'"' >> logs/sweep2.log 2>&1'
echo SWEEP2 DONE >> logs/sweep2.log
