# Controller architecture

Default `full` config: d = 128, `arch="mlp"`, memory on, CfC gate. Variant configs are noted where they differ (see `CONFIGS` in `controller/model.py`).

## 1. End-to-end controller (`full` config)

```mermaid
flowchart TB
    subgraph IN["Inputs: Batch from features.view → collate"]
        direction LR
        TXT["Frozen bge-small embeddings<br/>task_row · step_row · trace_row<br/>384-d, not trained"]
        CAT["Categorical ids<br/>step_type · round 0-2 · phase CHOOSE/HALT"]
        BUD["budget feats 6<br/>left, spent, step idx, 1/n, log cost, 0"]
        TRF["trace feats 2×7<br/>verifier_ok, escalated, round,<br/>log cost, same_step, Δstep, present"]
        CARDS["Model cards M×C<br/>train acc per step type + log10 cost"]
    end

    subgraph ENC["StepEncoder (encoder.py): Linear / Embedding / 2-layer SiLU MLPs"]
        direction TB
        T1["task = Linear 384→d"]
        T2["budget = MLP 6→d→d"]
        T3["step = Linear text + Emb type + Emb round + Emb phase"]
        T4["trace k = Linear text + MLP 7→d→d + card of model k<br/>× present mask"]
        T5["card = MLP C→d→d, one per pool model"]
    end

    TXT --> T1 & T3 & T4
    CAT --> T3
    BUD --> T2
    TRF --> T4
    CARDS --> T5
    T5 -. "card of trace model" .-> T4

    H["h = RMSNorm of (task + budget)/2<br/>H-level input, slow context"]
    X["x = (step + Σ trace) / (1 + #present)<br/>L-level input, fast context"]
    T1 --> H
    T2 --> H
    T3 --> X
    T4 --> X

    subgraph MEM["EpisodeMemory (memory.py): TTT-Linear fast weight + CfC gate"]
        W["W: B×d×d fast weight<br/>persists across the episode"]
        WR["write: newest trace item<br/>when write=True"]
        RD["read: query from h<br/>RMSNorm of W·q"]
        WR --> W --> RD
    end
    T4 -- "trace[:,0]" --> WR
    H --> RD
    RD -- "+" --> X

    subgraph CORE["HLCore (core.py): HRM-style recurrent reasoning"]
        HL["2 H-cycles × 3 L-steps<br/>SwiGLU residual blocks + MagicNorm"]
    end
    H --> CORE
    X --> CORE
    CORE --> ZH["z_H: B×d<br/>= DecisionRecord.decision_state"]

    subgraph HEADS["Heads (heads.py): linear heads"]
        MH["model head: Linear 1 of SiLU of need z_H + card_m<br/>→ B×M logits, one per pool model"]
        HH["halt head: Linear d→1"]
        CH["complexity head: Linear d→1"]
    end
    ZH --> MH & HH & CH
    T5 -- "card M×d" --> MH

    MH --> O1["model_logits B×M"]
    HH --> O2["halt_logit B"]
    CH --> O3["complexity_logit B"]
```

## 2. H/L reasoning core: the recurrence and the two `arch` options

```mermaid
flowchart LR
    subgraph LOOP["HLCore.forward: z_L starts at a fixed trunc-normal vector, z_H starts at h"]
        direction TB
        L1["L step 1: z_L = L(z_L + z_H + x)"] --> L2["L step 2"] --> L3["L step 3"]
        L3 --> H1["H step: z_H = H(z_H + z_L + h)"]
        H1 --> L4["L steps 4-6"] --> H2["H step 2"] --> OUT["final z_H"]
    end

    subgraph MLP["arch = mlp: ReasoningModule, used for H and for L"]
        direction TB
        M1["x + SwiGLU(RMSNorm x)<br/>d → 2×352 → d"] --> M2["x + SwiGLU(RMSNorm x)"] --> M3["weightless RMSNorm<br/>MagicNorm"]
    end

    subgraph ATT["arch = attn: AttnModule over 6 role tokens + learned slot emb"]
        direction TB
        A0["tokens: task, budget, step, trace-1, trace-2, memory<br/>+ key mask for absent ones"]
        A1["AttnBlock × 2:<br/>Gated MHA, 4 heads, sigmoid gate per head<br/>then SwiGLU, both pre-norm residual"]
        A2["RMSNorm → take the step token as z_H"]
        A0 --> A1 --> A2
    end

    LOOP -.uses.-> MLP
    LOOP -.or.-> ATT
```

Training detail: gradients only flow through the last `bp_steps` module calls (a truncated BPTT "staircase"). The horizon widens from 2 to 5 during warmup, as in HRM-Text.

## 3. Episode memory write path (TTT + Titans forgetting + CfC gate)

```mermaid
flowchart TB
    OBS["obs = newest trace token, B×d"] --> K["k = normalize(Linear_k obs)"]
    OBS --> V["v = Linear_v obs"]
    K & V --> S["surprise s = v − W·k<br/>delta-rule prediction error"]
    S --> GIN["gate input: s/|s|, log|s|, obs<br/>2d+1 dims"]
    GIN --> GATE

    subgraph GATE["Write-rate gate, chosen by cfg.gate"]
        CFC["cfc: Closed-form Continuous-time cell, 16 units<br/>backbone Linear+LeCun-tanh → ff1, ff2, time_a, time_b<br/>t = σ(time_a·dt + time_b)<br/>c = f1·(1−t) + t·f2, c persists across writes<br/>rate = σ(readout c)"]
        STL["stateless: MLP 2d+1 → 32 → 1, sigmoid"]
        CST["const: one learned scalar"]
    end

    S --> FG["forget α = σ(Linear of log|s|, obs)<br/>bias −4, so it starts near 0"]
    GATE -- rate --> UPD
    FG -- α --> UPD
    UPD["W ← W0 + (1−α)(W − W0) + rate · s kᵀ"]
    UPD --> READ["read: q = normalize(Linear_q h)<br/>out = RMSNorm(W q)"]
```

## 4. Output schema: tensors → contract dataclasses

```mermaid
classDiagram
    direction LR
    class Output {
        model_logits : Tensor B×M
        halt_logit : Tensor B
        complexity_logit : Tensor B
        z_h : Tensor B×128
        memory : MemoryState or None
        stats : surprise, rate, alpha
        residuals : list of H/L name, norm
    }
    class MemoryState {
        W : Tensor B×d×d
        c : Tensor B×gate_units
    }
    class Strategy {
        step_id : str
        round : int 1..2
        model_id : str  ← argmax model_logits
        halt : stop | escalate  ← halt_logit, HALT phase
        tools : list~str~
        memory : none | session | retrieval
        depth : low | medium | high
        verification : none | self_check | cross_model
        topology : single | parallel | sub_agents
        complexity : float 0..1  ← σ complexity_logit
        confidence : float 0..1
        version : v1
    }
    class DecisionRecord {
        step_id : str
        round : int
        strategy : Strategy
        log_probs : dict
        entropy : dict
        complexity_band : int 0..4
        lam : float or None
        decision_state : 128 floats  ← z_h
        state_version : v1
    }
    class Target {
        succeeds : M floats, per-model success
        halt : float, oracle halt label
        band : float, fraction of pool that fails
    }
    Output --> MemoryState
    Output ..> Strategy : decoded into
    Output ..> DecisionRecord : logged as
    DecisionRecord --> Strategy
    Target ..> Output : supervises the 3 heads
```

## Summary of network types

| Component | Network type | Where |
|---|---|---|
| Text features | Frozen **bge-small** sentence embeddings + linear projection | `controller/encoder.py` |
| Categorical features | `nn.Embedding` for step type, round and phase | `controller/encoder.py` |
| Budget, trace and card encoders | 2-layer **MLP** (Linear → SiLU → Linear) | `controller/encoder.py` |
| Reasoning core | **HRM-style hierarchical recurrent** model with an H (slow) and an L (fast) module | `controller/core.py` |
| H/L module (`mlp`) | Pre-norm residual **SwiGLU** blocks + MagicNorm | `controller/core.py` |
| H/L module (`attn`) | **Gated multi-head self-attention** transformer over 6 role tokens | `controller/attention.py` |
| Memory | **Fast-weight associative memory**: TTT-Linear delta rule plus Titans-style forgetting | `controller/memory.py` |
| Write gate | **CfC** (Closed-form Continuous-time liquid neural net) cell | `controller/memory.py` |
| Heads | Linear heads. The model head scores each **model card** against a "need" vector, so the pool size M can change | `controller/heads.py` |

The decoding arrows in diagram 4 (argmax to `model_id`, sigmoid to `complexity`) show what the contracts intend; that decoding isn't in `controller/`. The `tools`, `memory`, `depth`, `verification` and `topology` fields in `Strategy` currently have no head, so they keep their defaults.
