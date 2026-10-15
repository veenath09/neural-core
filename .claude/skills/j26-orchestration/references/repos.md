# GitHub Repository References

All repositories below were checked to exist (October 2026). Licence "check" means read the repo's LICENSE before copying any code; reading for reference needs no check. Copied or adapted code gets the header in `assets/templates/adapted_header.py` and, where the licence requires, the upstream licence file at the repo root.

## Architecture

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [sapientinc/HRM-Text](https://github.com/sapientinc/HRM-Text) | Apache-2.0 | H/L forward loop, exit-norm placement, backprop-horizon warmup (`models/baselines/hrm_nocarry_bp_warmup.py`, `config/arch/net/hrm.yaml`) | C1-04 |
| [sapientinc/HRM](https://github.com/sapientinc/HRM) | check | Original HRM, ACT halting, residual plots | C1-07, C1-13 |
| [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) | check | Compact recursive training loop (reference only) | C1 |
| [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch) | check | TTT-Linear update and mini-batch inner step | C1-08 |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | check (unofficial) | Surprise-gated memory update | C1-08, C1-09 |
| [mlech26l/ncps](https://github.com/mlech26l/ncps) | check | CfC and LTC cells in PyTorch | C1-09 |
| [raminmh/CfC](https://github.com/raminmh/CfC) | check | Original CfC code | C1-09 |

## Data and benchmarks

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [withmartian/routerbench](https://github.com/withmartian/routerbench) | check | Loaders and evaluation; data also on Hugging Face `withmartian/routerbench` | C3-01 |
| [ynulihao/LLMRouterBench](https://github.com/ynulihao/LLMRouterBench) | check | 33-model, 21-dataset routing data, 10 baseline routers | C3 next phase |
| [CommonstackAI/TwinRouterBench](https://github.com/CommonstackAI/TwinRouterBench) | Apache-2.0 | Step-level agent routing data with verified tiers | C3 external check |
| [MilkThink-Lab/RouterEval](https://github.com/MilkThink-Lab/RouterEval) | check | Large open-model routing records | optional |

## Routing baselines

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | check | Matrix-factorisation and BERT routers | C3-10 |
| [ZhangYiqun018/MTRouter](https://github.com/ZhangYiqun018/MTRouter) | check | Multi-turn router behaviour | related work |

## Embeddings

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | check | Runs the frozen encoder | C1-02 |
| [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | check | `bge-small-en-v1.5` source | C1-02 |

## Reinforcement learning

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | check | Single-file policy-gradient structure, advantage normalisation, entropy bonus | C2-06 |
| [ikostrikov/implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning) | check | Offline RL from logs (next phase) | C2-11 |

## Tooling (UI and monitoring)

| Repo | Licence | Take | Used by |
|---|---|---|---|
| [streamlit/streamlit](https://github.com/streamlit/streamlit) | check | Dashboard framework | UI |
| [duckdb/duckdb](https://github.com/duckdb/duckdb) | check | Query layer over Parquet; Postgres attach in Phase 3 | UI |
| [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) | check | Optional live training curves | C2 |
