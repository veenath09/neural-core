# Research References

Grouped by the role each paper plays. "Use" says what we take from it; "Our position" says how our work relates, which is what an examiner will ask. Numbers in brackets match the proposal's reference list where one exists. Read primary sources before citing a claim; do not cite from this table alone.

## Contents

1. Context-free routing and cascades
2. Step-level and agentic routing (closest prior work)
3. Learned orchestration
4. Benchmarks and datasets
5. Architecture: two-timescale reasoning
6. Architecture: memory and continuous-time networks
7. Halting and calibration
8. Reinforcement learning

## 1. Context-free routing and cascades

| Ref | Paper | Use | Our position |
|---|---|---|---|
| [1] | Ong et al., RouteLLM: Learning to Route LLMs with Preference Data, ICLR 2025, arXiv:2406.18665 | Strong/weak router baselines; cost-saving framing | Routes each query alone; we condition on workflow state. RouteLLM-style router is a baseline (C3-10) |
| [17] | Chen, Zaharia, Zou, FrugalGPT, TMLR 2024, arXiv:2305.05176 | Cascade pattern: cheap first, escalate if needed | Our halting is a learned, cost-aware escalation inside a workflow |
| [19] | Ding et al., Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing, ICLR 2024, arXiv:2404.14618 | Joint cost-quality objective | Same objective, applied per step with state |
| [20] | Song et al., IRT-Router, 2025 | Difficulty and ability on one scale | Our complexity head is a per-step, state-conditioned difficulty estimate |
| [21] | Somerstep et al., CARROT: A Cost-Aware Rate-Optimal Router, arXiv:2502.03261 | Per-query trade-off is well posed | We extend the trade-off over an episode with a budget |
| [22] | Sikeridis et al., PickLLM: Context-Aware RL-Assisted LLM Routing, arXiv:2412.12170 | RL for routing | RL is not our novelty; what the policy conditions on is |
| [23] | Yue et al., LLM Cascades with Mixture of Thoughts Representations, arXiv:2310.03094 | Escalation signal quality | Our verifier signal is noisy by design; we model that |
| [24] | Ramirez, Birch, Titov, Uncertainty-Based Two-Tier Selection, arXiv:2405.02134 | Uncertainty as escalation signal | Calibration of our confidence is tested under shift |
| [25] | Mei et al., OmniRouter, arXiv:2502.20576 | Budget-controllable routing | Budget enters our state per step |
| [26] | Jitkrittum et al., UniRoute: Routing to Unseen LLMs at Test Time, 2026 | Models as capability vectors | Our model head scores ModelCards the same way; tested by leave-one-model-out |
| [27] | Shafran et al., Rerouting LLM Routers, arXiv:2501.01818 | Adversarial manipulation of routers | Stated limitation; not addressed |
| [18] | Moslem and Kelleher, Dynamic Model Routing and Cascading: A Survey, arXiv:2603.04445 | Field overview | Positioning in the literature review |
| | R2-Router: A New Paradigm for LLM Routing with Reasoning, arXiv:2602.02823 | Recent routing direction | Related work |

## 2. Step-level and agentic routing (closest prior work)

These narrow the gap statement: conditioning on state per step is no longer new by itself. Our claim is about maintaining and revising an understanding of the task (two-timescale core, episode memory) and pricing further effort.

| Paper | Use | Our position |
|---|---|---|
| AgentRouter: Heterogeneous Model Routing for Cost-Optimal Multi-Step Agentic Workflows, ICML 2026, arXiv:2609.22951 | Step-level routing as sequential assignment; small classifier per step | Strongest baseline candidate. Classifies a snapshot per step into tiers; we keep an episode memory, plan on two timescales and decide halting |
| Agentic Routing: The Harness-Native Data Flywheel, arXiv:2607.11399 | Step-level selection conditioned on harness state | Same: snapshot features vs a revised task representation |
| MetaRoute-Bench: Evaluating Meta-Decision Policies for Agentic Workflow Routing, arXiv:2608.00107 | Task-aware adaptive vs fixed and one-shot policies | Close research question; possible evaluation source |
| Explainable Model Routing for Agentic Workflows, arXiv:2604.03527 | Interpretability of routing decisions | Our episode view and surprise trace are an interpretability aid |
| MTRouter (multi-turn router), repo `ZhangYiqun018/MTRouter` | Multi-turn routing, fewer switches, tolerance to transient errors | Related; compare behaviour on retries |

## 3. Learned orchestration

| Ref | Paper | Use | Our position |
|---|---|---|---|
| [3] | Trinity: An Evolved Coordinator for Language-Model Agents, arXiv:2512.04695 | Learned multi-model coordination | Plans ahead with no cost term; we decide per step with cost |
| [4] | Conductor: Learning to Orchestrate Agents in Natural Language, arXiv:2512.04388 | Learned workflow composition | Same contrast |
| [5] | Moonshot AI, Kimi K2.5: Parallel-Agent RL, arXiv:2602.02276 | Critical-path latency in rewards; reward shaping guards | Latency term kept separate in v1 |

## 4. Benchmarks and datasets

| Ref | Paper | Use |
|---|---|---|
| [2] | Hu et al., RouterBench: A Benchmark for Multi-LLM Routing Systems, arXiv:2403.12031 | Prototype data and simulator; per-model outcomes and costs |
| | Li et al., LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing, ACL Findings 2026, arXiv:2601.07206 | Next-phase replication on 33 recent models; finding that embedding choice matters little justifies a small frozen encoder |
| | TwinRouterBench: Fast Static and Live Dynamic Evaluation for Realistic Agentic LLM Routing, arXiv:2605.18859 | Real agent trajectories with verified tier labels; external validity check |
| | RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers, arXiv:2510.00202 | Router comparison methodology |
| | RouterEval, repo `MilkThink-Lab/RouterEval` | Large-scale open-model routing records (no cost data) |

## 5. Architecture: two-timescale reasoning

| Ref | Paper | Use | Our position |
|---|---|---|---|
| [10] | Wang et al., Hierarchical Reasoning Model, arXiv:2506.21734 | Slow H / fast L modules, hierarchical convergence, ACT halting | Basis of the core; our H reads the task, L reads execution |
| [11] | Ge, Liao, Poggio, HRM: Perspectives and Misconceptions, arXiv:2510.00355 | Halting saves compute rather than improving accuracy; hierarchy untested with different inputs | We claim economy of effort; specialisation probe tests the different-input case |
| [28] | Wang et al., HRM-Text: Efficient Pretraining Beyond Scaling, arXiv:2605.20613 | Separate H/L parameters at scale; MagicNorm; backprop-horizon warmup; gated self-attention; halting named as a natural complement | Source of `core.py` schedule and of the `attn` variant's gated-attention blocks (`attention.py`); our halting prices monetary cost |
| | Jolicoeur-Martineau, Less is More: Recursive Reasoning with Tiny Networks (TRM), arXiv:2510.04871 | Small recursive reasoner; MLPs beat attention on small fixed contexts; 1-step gradients are weak; weight EMA 0.999 | Considered and dropped as a configuration; source of the MLP-core choice, the fixed `z_L` init and the EMA (D11) |

## 6. Architecture: memory and continuous-time networks

| Ref | Paper | Use | Our position |
|---|---|---|---|
| [12] | Sun et al., Learning to (Learn at Test Time): RNNs with Expressive Hidden States, ICML 2025, arXiv:2407.04620 | TTT-Linear: hidden state as a model trained online | Our episode memory W |
| [13] | Behrouz, Zhong, Mirrokni, Titans: Learning to Memorize at Test Time, arXiv:2501.00663 | Surprise-gated memory updates; forgetting toward a prior | Closest prior to our gated memory; we take its forgetting term (no momentum); `gate_stateless` ablation tests whether the CfC form adds anything |
| [8] | Hasani et al., Liquid Time-constant Networks, AAAI 2021, arXiv:2006.04439 | Input-dependent time constant, bounded state | Liquid property of the gate |
| [9] | Hasani et al., Closed-form Continuous-time Neural Networks, Nature Machine Intelligence 2022, arXiv:2106.13898 | Closed form, no ODE solver | Gate cheap enough to run every decision |
| [6] | Vaswani et al., Attention Is All You Need, NeurIPS 2017 | Background | MLP blocks are the default core; the `attn` variant (D12) uses HRM-Text gated attention over six role tokens |
| [7] | Cho et al., RNN Encoder-Decoder, EMNLP 2014 | Background | GRU baseline considered and dropped |

## 7. Halting and calibration

| Ref | Paper | Use |
|---|---|---|
| [14] | Graves, Adaptive Computation Time for RNNs, arXiv:1603.08983 | Origin of learned halting; our cost is money, not internal compute |
| [15] | Guo et al., On Calibration of Modern Neural Networks, ICML 2017, arXiv:1706.04599 | Temperature scaling (our `Calibrator`) |
| [16] | Ovadia et al., Can You Trust Your Model's Uncertainty?, NeurIPS 2019, arXiv:1906.02530 | Calibration degrades under shift; motivates the shifted test set |

## 8. Reinforcement learning

| Ref | Paper | Use |
|---|---|---|
| [34] | Sutton and Barto, Reinforcement Learning: An Introduction, 2nd ed., 2018 | Monte Carlo policy gradient, baselines |
| [31] | Schulman et al., Proximal Policy Optimization, arXiv:1707.06347 | Considered; not used in v1 |
| [29] | Shao et al., DeepSeekMath (GRPO), arXiv:2402.03300 | Group baselines need several samples per input, unavailable in live single episodes; we use a bucketed baseline instead |
| [30] | DeepSeek-AI, DeepSeek-R1, arXiv:2501.12948 | Same |
| [32] | Tavakoli et al., Action Branching Architectures, AAAI 2018, arXiv:1711.08946 | One head per decision, joint log-prob as a sum |
| [33] | Kostrikov et al., Implicit Q-Learning, arXiv:2110.06169 | Offline RL from logged live episodes (next phase) |
