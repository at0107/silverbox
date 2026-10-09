#!/bin/bash
# Sweep 1: 12 configs in parallel (one per CPU core)
cd "$(dirname "$0")"   # run from the repo folder
CONFIGS=$(cat <<'EOF'
{"name":"a01_tanh64x3_w64","act":"tanh","width":64,"depth":3,"win":64}
{"name":"a02_tanh32x2_w64","act":"tanh","width":32,"depth":2,"win":64}
{"name":"a03_tanh128x3_w64","act":"tanh","width":128,"depth":3,"win":64}
{"name":"a04_silu64x3_w64","act":"silu","width":64,"depth":3,"win":64}
{"name":"a05_relu64x3_w64","act":"relu","width":64,"depth":3,"win":64}
{"name":"a06_tanh64x3_skip_w64","act":"tanh","skip":true,"win":64}
{"name":"a07_tanh64x3_skipphys_w64","act":"tanh","skip":true,"phys_init":true,"win":64}
{"name":"a08_tanh64x3_w128","act":"tanh","win":128}
{"name":"a09_tanh64x3_curr32-256","act":"tanh","win":32,"win_final":256,"steps":4000}
{"name":"a10_tanh64x3_lr1e-3_w64","act":"tanh","lr":1e-3,"win":64}
{"name":"a11_tanh64x3_rmsprop_w64","act":"tanh","opt":"rmsprop","lr":1e-3,"win":64}
{"name":"a12_tanh64x3_skipphys_curr32-256","act":"tanh","skip":true,"phys_init":true,"win":32,"win_final":256,"steps":4000}
EOF
)
mkdir -p logs
echo "$CONFIGS" | xargs -P 12 -d '\n' -I{} sh -c './venv/bin/python silverbox_pinn.py '"'"'{}'"'"' >> logs/sweep1.log 2>&1'
echo SWEEP1 DONE >> logs/sweep1.log
