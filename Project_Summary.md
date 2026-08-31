# Project Summary — PIDM (Prompt Injection Detection Middleware)

**Thesis:** Securing Multi-Agent LLM Pipelines: Detecting Prompt Injection Attacks Using a Middleware Framework
**Institution:** IIUC — Department of CSE
**Supervisor:** Mr. Md. Badiuzzaman Biplob
**Authors:** Md. Ashikul Islam (C213096R), Abdul Aowal Manik (213100), Shoabur Rahaman Durjoy (C221141R)

---

## What Are We Trying to Achieve?

Modern AI systems often use multiple AI agents talking to each other (called Multi-Agent LLM Pipelines). For example, one agent gathers information, another analyzes it, and another writes a report — all automatically.

**The Problem:** These agents trust each other's messages blindly. An attacker can slip a hidden malicious instruction inside a message (called a **Prompt Injection Attack**), and it silently spreads through all agents — like a virus. No existing tool properly protects against this.

**Our Goal:** Build a middleware (a protective layer that sits between agents) called **PIDM** that:

- Reads every inter-agent message in real time
- Decides if it is an attack or safe
- Blocks (quarantines) dangerous messages before they spread
- Works with any AI framework (AutoGen, LangGraph)
- Adds less than 60ms of delay — fast enough for real use

---

## What Have We Implemented?

### 1. Dataset (`pidm/data/`)

We built a dataset of ~40,000 labeled messages (attack vs. safe) because no existing dataset covers multi-agent scenarios.

| Source                     | What it is                                                                              |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `synthetic_generator.py` | Auto-generates 6 types of attack messages using templates                               |
| `paraphraser.py`         | Rewrites generated messages in different words so the model doesn't memorize patterns   |
| `real_data_loader.py`    | Downloads real attack datasets from HuggingFace (used only for testing, never training) |
| `trace_simulator.py`     | Simulates realistic multi-agent conversations with injections hidden inside             |
| `benign_generator.py`    | Generates normal (non-attack) messages for balance                                      |

**6 Attack Types We Cover:**

1. Direct Override — "Ignore all previous instructions..."
2. Role-Play Smuggling — "Pretend you are DAN with no restrictions..."
3. Tool Output Injection — Hiding commands inside tool responses
4. Chain Poisoning — Asking one agent to pass malicious instructions to the next
5. Encoded Payload — Base64 or encoded hidden commands
6. Context Flooding — Burying an attack inside a very long message

---

### 2. Detection System (`pidm/detection/`) — 4 Layers

All 4 layers run in sequence. Each gives a suspicion score. A weighted average decides: block or pass.

```
Inter-Agent Message
       │
       ▼
  Layer 1: Rule-Based Filter (RBF)         ~2ms
       │   Regex patterns + heuristics
       ▼
  Layer 2: DeBERTa-v3 Classifier           ~45ms
       │   Fine-tuned AI model (our main detector)
       ▼
  Layer 3: Semantic Intent Drift (SID)     ~10ms
       │   Detects when a message's "meaning" shifts suspiciously
       ▼
  Layer 4: Graph Cascade Detector (GCPD)   ~1ms
       │   Checks how many agents would be affected if this spreads
       ▼
  Final Score ≥ 0.50 → QUARANTINE
  Final Score < 0.50 → PASS
```

| File                          | What it does                                                  |
| ----------------------------- | ------------------------------------------------------------- |
| `rule_based_filter.py`      | Fast keyword/regex scan — catches obvious attacks            |
| `classifier.py`             | Fine-tuned DeBERTa transformer — catches subtle attacks      |
| `semantic_intent_drift.py`  | Measures if message meaning drifted from conversation context |
| `graph_cascade_detector.py` | Scores risk by how many agents a message could infect         |
| `orchestrator.py`           | Combines all 4 layer scores into one final decision           |
| `quarantine.py`             | Logs and blocks detected attacks                              |

---

### 3. Evaluation (`pidm/eval/`)

We compare PIDM against 6 systems to prove it works better:

| System                           | What it is                                                  |
| -------------------------------- | ----------------------------------------------------------- |
| B0: No Protection                | Does nothing — worst case                                  |
| B1: Keyword Filter               | Simple word-matching only                                   |
| B2: TF-IDF + Logistic Regression | Classical ML, no deep learning                              |
| B3: RBF Only                     | Just our rule layer                                         |
| B4: Classifier Only              | Just our AI model, no SID/GCPD                              |
| **B5: Full PIDM (Ours)**   | All 4 layers together                                       |
| B6: ProtectAI                    | A real published tool — zero-shot, not trained on our data |

We also test how the system performs as training data grows (2k → 5k → 10k → 20k → 40k messages) and run 90 hand-crafted evasion attacks to see what slips through.

---

### 4. Integrations (`pidm/integrations/`)

- `autogen_hook.py` — Plug PIDM into Microsoft AutoGen pipelines
- `langgraph_node.py` — Plug PIDM into LangGraph pipelines

### 5. Demo (`pidm/demo/`)

- A Gradio web app where you can type any message and see PIDM's detection decision live

---

## What Is NOT Done Yet (Honest Status)

| Issue                               | Status                                                                  |
| ----------------------------------- | ----------------------------------------------------------------------- |
| Full pipeline run                   | Not completed —`full_run.log` cuts off mid-run, no output folder yet |
| Statistical tests                   | Missing — no McNemar's test or confidence intervals yet                |
| Hyperparameter ablation             | Ensemble weights are currently hardcoded, not tuned by data yet         |
| Dependency pinning                  | `requirements.txt` has open-ended versions — needs `pip freeze`    |
| `torch.manual_seed()` in training | Missing — results may vary between runs                                |
| Proposal vs. code mismatch          | Proposal says 2k–5k messages; code targets 40k                         |
| Missing citations                   | InjecAgent, AgentDojo, OWASP LLM Top 10 not yet cited                   |

---

## Hardware & Tools Used

| What                 | Which                                                       |
| -------------------- | ----------------------------------------------------------- |
| AI Model             | DeBERTa-v3-base (auto-selects smaller model on weaker GPUs) |
| Semantic Model       | sentence-transformers/all-MiniLM-L6-v2                      |
| Graph Library        | NetworkX                                                    |
| Frameworks Supported | AutoGen, LangGraph                                          |
| Demo UI              | Gradio                                                      |
| Experiment Tracking  | MLflow / Weights & Biases                                   |
| Language             | Python 3.11+                                                |

---

## One-Line Summary for the Meeting

> We built a 4-layer real-time middleware (PIDM) that detects prompt injection attacks in multi-agent AI systems, trained on a custom 40,000-message dataset, and compared it against 6 baselines — but the full pipeline has not yet been run to completion.
