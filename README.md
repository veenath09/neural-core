# neural-core

Prototype of the C1 adaptive orchestration controller (J26-DS-359). Plan: `docs/IMPLEMENTATION_PLAN_neural_core.md`.

## Setup

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt

## Data pipeline

    .venv/bin/python -m data.routerbench   # table + split + pool.json -> data/processed/ (raw download -> data/raw/)
    .venv/bin/python -m data.embed         # bge-small embeddings -> data/processed/embeddings.npy
    .venv/bin/python -m data.episodes      # episode skeletons -> data/processed/episodes_{split}.parquet
    .venv/bin/python -m eval.baselines     # baseline numbers on test -> runs/baselines_test.csv

## Training

    .venv/bin/python -m eval.baselines val                    # reference numbers on validation
    .venv/bin/python -m train.train --config memory_off --seed 0
    .venv/bin/python -m eval.diagnostics runs/memory_off-s0  # residuals.png, memory_trace.csv

Configs: `full`, `gate_off`, `gate_stateless`, `memory_off`, `context_free`.

## Tests

    .venv/bin/python -m pytest
