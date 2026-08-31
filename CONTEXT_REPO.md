# Context Repository: LLM Multi-Agent Security Thesis Project

> Generated: 2026-08-11
> Source documents: `Proposal.docx`, `Thesis Proposal 1.4_page-0001(2).pdf`, `Resources [LLM Multi Agent Security].xlsx`

---

## 1. Project Identity

| Field                    | Detail                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Thesis Title**   | "Securing Multi-Agent LLM Pipelines: Detecting Prompt Injection Attacks Using a Middleware Framework" |
| **Internal Title** | "Detecting Prompt Injection Attacks in Multi-Agent LLM Orchestration Pipelines"                       |
| **Institution**    | International Islamic University Chittagong (IIUC)                                                    |
| **Department**     | Computer Science and Engineering                                                                      |
| **Supervisor**     | Mr. Md. Badiuzzaman Biplob (Lecturer, CSE, IIUC)                                                      |
| **Student 1**      | Md. Ashikul Islam — ID: C213096R                                                                     |
| **Student 2**      | Abdul Aowal Manik — ID: 213100                                                                       |
| **Student 3**      | Shoabur Rahaman Durjoy — ID: C221141R                                                                |

---

## 2. Core System: PIDM

**Prompt Injection Detection Middleware (PIDM)** — a lightweight, model-agnostic middleware layer that sits inside multi-agent LLM pipelines to detect and quarantine prompt injection attacks in real-time without incurring significant latency or operational overhead.

### PIDM Architecture (3-layer)

```
[User Input / External Data Source]
          ↓
[Multi-Agent LLM Pipeline (AutoGen / LangGraph)]
          ↓
[PIDM — Prompt Injection Detection Middleware]
    ├─ Layer 1: Rule-Based Filter (RBF)
    │     Fast, deterministic first-pass using structural heuristics
    │     and semantic similarity thresholds.
    │     → If no suspicious pattern: pass message through
    │     → If suspicious: escalate to Layer 2
    ├─ Layer 2: Classifier (DeBERTa-v3-small / DistilBERT)
    │     Lightweight transformer for binary classification (benign / injected)
    └─ Layer 3: Decision & Quarantine Engine
          Routes flagged messages to quarantine queue + logs for audit

[Benign] → Forward to Next Agent → Task Execution
[Malicious] → Quarantine & Log → Security Alert → Security Monitoring
```

---

## 3. Problem Statement

Multi-agent LLM pipelines (AutoGen, LangGraph, CrewAI) operate on an **implicit benign trust model** — all inter-agent messages are assumed safe. This is exploitable because:

1. Messages can be poisoned by external APIs, databases, web search results, or user inputs.
2. A single compromised message can **cascade** across an entire pipeline (indirect prompt injection).
3. Existing keyword-based detection fails against semantically obfuscated attacks (Base64 encoding, role-play prompts, indirect instruction smuggling).
4. No standardized datasets for inter-agent message injection attacks exist.
5. Current methods (input preprocessing, robust instruction-following training, LLM-as-judge) are computationally expensive and not designed for multi-turn multi-agent scenarios.

---

## 4. Literature Review Summary

### Foundational Works

| Authors                 | Year | Contribution                                                                                                                          |
| ----------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Perez & Ribeiro         | 2022 | First formal characterisation of prompt injection; showed LLMs can be manipulated via adversarial inputs to ignore prior instructions |
| Greshake et al.         | 2023 | Extended to**indirect injection** — malicious instructions hidden in external content accessed by the model (arXiv:2302.12173) |
| Liu et al.              | 2024 | Taxonomy of direct, indirect, and compositional prompt injection types (arXiv:2310.12815)                                             |
| Wu et al. (AutoGen)     | 2023 | Programmable multi-agent pipelines — no built-in security features (arXiv:2308.08155)                                                |
| Chase (LangGraph)       | 2024 | Multi-agent collaboration framework — no built-in security                                                                           |
| Yang et al.             | 2024 | Demonstrated agent hijacking scenarios but no systematic detection/mitigation (arXiv:2402.11208)                                      |
| Lamport et al.          | 1982 | Byzantine fault tolerance and secure consensus in classical MAS                                                                       |
| Jain, Wei, Zheng et al. | 2023 | LLM-as-judge and input preprocessing for detection — computationally expensive                                                       |

### Key Research Gaps

1. Cascading multi-agent pipeline detection is poorly covered.
2. No minimally invasive model-agnostic middleware that minimizes false positives.
3. No standardized inter-agent message injection attack datasets.
4. Existing methods are ineffective against semantically obfuscated or indirect injection attacks.

---

## 5. Attack Taxonomy (6 Categories)

| # | Attack Type               | Description                                                           |
| - | ------------------------- | --------------------------------------------------------------------- |
| 1 | Direct Override           | Explicit instruction to ignore prior system prompt                    |
| 2 | Role-Play Smuggling       | Malicious instructions embedded in fictional/role-play context        |
| 3 | Tool Output Injection     | Injected payload delivered through tool/API response                  |
| 4 | Indirect Chain Poisoning  | Malicious content injected into shared memory/context that propagates |
| 5 | Encoded Payload Injection | Obfuscated attacks (Base64, ROT13, Unicode escapes)                   |
| 6 | Context Window Flooding   | Overwhelming the context window to drown out system instructions      |

**Threat model framework:** STRIDE (Spoofing, Tampering, Elevation of Privilege)

**Known vulnerability patterns from research:**

- **Prompt Infection:** Indirect injection where malicious prompts in shared data propagate across multiple agents
- **Chained Attacks:** Individual attacks may fail, but composed/chained attacks at architectural boundaries achieve 91–96% success rates
- **CORBA (Contagious Recursive Blocking):** Novel propagation + resource depletion pattern difficult to mitigate via alignment

---

## 6. Research Aim & Objectives

**Aim:** Develop a lightweight, model-agnostic PIDM capable of identifying and mitigating prompt injection attacks in multi-agent LLM orchestration pipelines.

**Objectives:**

1. Design and implement PIDM for real-time monitoring of inter-agent messages
2. Develop a comprehensive taxonomy of prompt injection attacks specific to multi-agent environments
3. Construct a labeled dataset of benign and adversarial inter-agent messages (2,000–5,000 messages)
4. Evaluate PIDM: Precision, Recall, F1-Score, False Positive Rate, Detection Latency
5. Test transferability across AutoGen and LangGraph frameworks
6. Establish a formal threat model for inter-agent communication security

---

## 7. Methodology (4 Phases)

### Phase 1 — Threat Modelling & Attack Taxonomy (Months 1–2)

- Analyze existing prompt injection attacks and extend to multi-agent contexts
- Define formal threat model using STRIDE
- Develop taxonomy of ≥6 attack categories
- **Deliverable:** Threat model document + attack taxonomy report

### Phase 2 — Dataset Construction (Months 2–3)

- Generate 2,000–5,000 labeled inter-agent messages using:
  - Real pipeline traces
  - Synthetic injections
  - Augmented benign messages
- Split: 70% train / 15% validation / 15% test
- Human-reviewed label quality assurance
- **Deliverable:** Labeled dataset (2K–5K messages)

### Phase 3 — PIDM Design & Implementation (Months 3–6)

- Rule-Based Filter (RBF): structural heuristics + semantic similarity thresholds
- Classifier Layer: DeBERTa-v3-small or DistilBERT (binary: benign/injected)
- Decision & Quarantine Engine: flagged message routing + audit logging
- **Deliverable:** PIDM library v1.0, GitHub repository

### Phase 4 — Evaluation & Analysis (Months 6–8)

- Evaluate on test set + live pipeline simulation
- Metrics: Precision, Recall, F1-Score, FPR, Detection Latency, Pipeline Throughput Impact
- Ablation study: RBF-only vs. Classifier-only vs. Full PIDM
- Cross-framework validation: AutoGen and LangGraph
- **Deliverable:** Evaluation report + performance metrics

### Write-up (Months 8–10)

- Thesis writing and revisions
- **Deliverable:** Final thesis submission

---

## 8. Technical Stack

| Component            | Technology                                  |
| -------------------- | ------------------------------------------- |
| MAS Frameworks       | AutoGen, LangGraph                          |
| LLM APIs             | Claude API (Anthropic), GPT-4o (OpenAI)     |
| Detection Model      | DeBERTa-v3-small / DistilBERT (HuggingFace) |
| Programming Language | Python 3.11+                                |
| Experiment Tracking  | MLflow or Weights & Biases                  |
| Dataset Management   | Pandas, HuggingFace Datasets                |
| Testing              | Pytest, custom pipeline simulation harness  |
| Version Control      | Git + GitHub                                |

---

## 9. Expected Contributions

1. First comprehensive taxonomy of prompt injection attacks in multi-agent LLM pipelines
2. Open-source, annotated dataset of adversarial inter-agent messages
3. PIDM: lightweight, model-agnostic middleware for real-time detection and mitigation
4. Empirical evaluation establishing baseline detection performance and operational feasibility
5. Formal threat model for securing inter-agent communication in autonomous AI systems

---

## 10. Expected Outcomes

1. PIDM capable of detecting and quarantining prompt injection attacks in multi-agent LLM pipelines
2. High detection accuracy with low false positives and minimal latency impact
3. Open-source dataset and middleware library for future research
4. Formal threat model guiding secure design of autonomous AI pipelines
5. Potential scalability for enterprise AI and research applications

---

## 11. Future Work

- Expand framework to other LLM orchestration frameworks (e.g., CrewAI, Haystack)
- Explore adaptive learning techniques to detect novel injection strategies
- Integrate PIDM into production-ready autonomous agent deployments

---

## 12. Resources Catalogue

### 12.1 Datasets

| Name                                            | URL                                                                                         | Notes                                                           |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| AI Agent Evasion Dataset                        | https://www.kaggle.com/datasets/cyberprince/ai-agent-evasion-dataset                        |                                                                 |
| Malicious Prompt Detection Dataset (MPDD)       | https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd | 40K balanced prompts for malicious intent detection             |
| Malicious_and_Benign_Dataset                    | https://www.kaggle.com/datasets/chaimajaziri/malicious-and-benign-dataset                   |                                                                 |
| Canadian Institute for Cybersecurity datasets   | https://www.unb.ca/cic/datasets/index.html                                                  |                                                                 |
| Smart Manufacturing Multi-Agent Control Dataset | https://www.kaggle.com/datasets/ziya07/smart-manufacturing-multi-agent-control-dataset      | Adaptive Optimization & Security for Decentralized Industry 4.0 |
| Prompt Optimization Dataset for Reasoning Tasks | https://www.kaggle.com/datasets/rishi2674/prompt-optimization-dataset-for-reasoning-tasks   |                                                                 |
| Multi-Agent LLM Full Dataset (HuggingFace)      | https://huggingface.co/TuringsSolutions                                                     | Reddit post + GitHub: RichardAragon/MultiAgentLLM               |
| Security AI Agent Dataset                       | https://huggingface.co/datasets/DeepNLP/security-ai-agent                                   |                                                                 |

### 12.2 Key Academic Papers

| Paper                                                                                               | Link                                                                                                                                      | Notes                              |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Red-Teaming LLM Multi-Agent Systems via Communication Attacks                                       | https://aclanthology.org/2025.findings-acl.349.pdf                                                                                        |                                    |
| Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks        | https://aclanthology.org/2025.acl-long.476.pdf                                                                                            |                                    |
| The Trust Paradox in LLM-Based Multi-Agent Systems                                                  | https://arxiv.org/pdf/2510.18563                                                                                                          |                                    |
| AgentLeak: A Full-Stack Benchmark for Privacy Leakage in Multi-Agent LLM Systems                    | https://arxiv.org/pdf/2602.11510                                                                                                          |                                    |
| TAMAS: A Dataset for Investigating Security Risks in Multi-Agent LLM Systems                        | https://cdn.iiit.ac.in/cdn/precog.iiit.ac.in/pubs/TAMAS.pdf                                                                               | Dedicated dataset for MAS security |
| MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber Defense                         | https://arxiv.org/pdf/2512.14846                                                                                                          |                                    |
| Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust          | https://arxiv.org/html/2508.19870v1                                                                                                       | Survey                             |
| Many hands make light work: LLM-based multi-agent system for detecting malicious PyPI packages      | https://www.sciencedirect.com/science/article/pii/S0164121226000269                                                                       | No data sharing                    |
| AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems               | https://arxiv.org/pdf/2408.14972                                                                                                          | Closest to PIDM concept            |
| Audit-LLM: Multi-Agent Collaboration for Log-based Insider Threat Detection                         | https://arxiv.org/pdf/2408.08902                                                                                                          |                                    |
| Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT                          | https://arxiv.org/abs/2601.17817                                                                                                          |                                    |
| Advanced Smart Contract Vulnerability Detection via LLM-Powered Multi-Agent Systems                 | https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11121619                                                                             |                                    |
| LLM and AI Agents for Autonomous Systems: Survey of Applications, Datasets, and Security Challenges | https://ieeexplore.ieee.org/abstract/document/11397656                                                                                    |                                    |
| Security and Privacy in Multi-Agent LLM Networks: Addressing Vulnerabilities                        | https://www.igi-global.com/chapter/security-and-privacy-in-multi-agent-llm-networks/389185                                                |                                    |
| Open Challenges in Multi-Agent Security                                                             | https://drive.google.com/drive/folders/1BDXy_MD-bEUoIzIqNr_GUJVH0ciobsLL                                                                  |                                    |
| A Multi-Agent System for Cybersecurity Threat Detection and Correlation Using LLMs                  | https://ieeexplore.ieee.org/abstract/document/11141466                                                                                    |                                    |
| A Multi-Agent Architecture for Governance and Security of LLM-Based Knowledge Access                | https://ieeexplore.ieee.org/abstract/document/11401633                                                                                    |                                    |
| Multi-Agent for Network Security Monitoring and Warning: A Generative AI Solution                   | https://ieeexplore.ieee.org/abstract/document/11031464                                                                                    |                                    |
| TrustNet: Hybrid ML and LLM-Based Multi-Agent System for Scam Website Detection                     | https://ieeexplore.ieee.org/abstract/document/11395765                                                                                    |                                    |
| MultiPhishGuard: LLM-based Multi-Agent System for Phishing Email Detection                          | https://arxiv.org/pdf/2505.23803                                                                                                          | Includes dataset                   |
| An Intrusion Detection System Dataset for a Multi-Agent Cyber-Physical Conveyor System              | https://www.researchgate.net/publication/371468176_An_Intrusion_Detection_System_Dataset_for_a_Multi-Agent_Cyber-Physical_Conveyor_System |                                    |
| AutoReview: LLM-based Multi-Agent System for Security Issue-Oriented Code Review                    | https://dl.acm.org/doi/epdf/10.1145/3696630.3728618                                                                                       |                                    |
| LIVA: Multi-Agent LLM-Assisted System for IoT Vulnerability Analysis                                | https://ieeexplore.ieee.org/abstract/document/11397462                                                                                    |                                    |
| Awesome-Agent-Papers                                                                                | https://github.com/luo-junyu/Awesome-Agent-Papers                                                                                         |                                    |

### 12.3 GitHub Repositories

| Repo                                    | URL                                                          | Description                                                                             |
| --------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| Multi-Agent-SecOps-LLM                  | https://github.com/tegridydev/multi-agent-secops-llm         | Automated threat detection, response workflows, real-time security monitoring           |
| Awesome-LLM-agent-Security              | https://github.com/wearetyomsmnv/Awesome-LLM-agent-Security  | Curated list of MAS security frameworks                                                 |
| Zero-Day LLM Ensemble                   | https://github.com/lodetomasi/zero-day-llm-ensemble          | Multi-agent system for zero-day vulnerability detection                                 |
| ProjectRecon/awesome-ai-agents-security | https://github.com/ProjectRecon/awesome-ai-agents-security   | Strix: autonomous AI pentesting agent (Docker sandbox)                                  |
| Awesome-LLM4Cybersecurity               | https://github.com/tmylla/Awesome-LLM4Cybersecurity          | AutoSafeCoder: LLM-generated code security via static analysis + fuzz testing           |
| GoClaw                                  | https://github.com/nextlevelbuilder/goclaw                   | Multi-agent AI gateway with multi-tenant isolation and permission-based comms           |
| Awesome-Agent-Security                  | https://github.com/ucsb-mlsec/Awesome-Agent-Security         | Research on threats/risks for LLM-enabled agents including multi-agent prompt infection |
| AgentSafety                             | https://github.com/OSU-NLP-Group/AgentSafety                 | Agent robustness papers + Agent Security Bench (ASB) benchmark                          |
| TrustAgent                              | https://github.com/Ymm-cll/TrustAgent                        | Trustworthiness analysis: intrinsic (memory, tools) vs extrinsic (environment) factors  |
| llm-agent-security                      | https://github.com/theconsciouslab-ai/llm-agent-security     | Security testing framework for Function Calling and MCP vulnerabilities                 |
| LLM-Multi-Agent-System                  | https://github.com/abbottyanginchina/LLM-Multi-Agent-System  |                                                                                         |
| agentUniverse                           | https://github.com/agentuniverse-ai/agentUniverse            |                                                                                         |
| Awesome-LLM                             | https://github.com/Hannibal046/Awesome-LLM                   | General LLM resource collection                                                         |
| AI Agents for Beginners (Microsoft)     | https://github.com/microsoft/ai-agents-for-beginners         |                                                                                         |
| Hands-On LLM                            | https://github.com/HandsOnLLM/Hands-On-Large-Language-Models |                                                                                         |

### 12.4 Code / Notebooks

| Name                                                              | URL                                                                                            |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Using 9 ML Models for Malicious Prompt Detection                  | https://www.kaggle.com/code/mdismielhossenabir/using-9-ml-model-for-malicious-prompt-detection |
| Step-by-Step Implementation of LLM Multi-Agent Security (ChatGPT) | https://chatgpt.com/share/69b3cd1b-fc2c-8012-97b3-0882633e9616                                 |

### 12.5 Kaggle Agents Intensive Capstone Projects (Relevant Examples)

| Project                                                                          | URL                                                                                                                               | Notes                                                                    |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| AI Security Multi-Agent to Prevent Prompt Injection                              | https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/ai-security-multi-agent-to-prevent-prompt-injectio | GitHub: Seltsam1/adk-llm-prevent-injection —**directly relevant** |
| Intelligent Task-Oriented Multi-Agent System for Automated Cybersecurity Monitor | https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/intelligent-task-oriented-multi-agent-system-for-a |                                                                          |
| AI Multi-Agent Security System for Phishing & Scam Detection                     | https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/ai-security-agent                                  |                                                                          |
| SecureOps AI: Multi-Agent System for Automated Incident Response                 | https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/new-writeup-1763381265694                          |                                                                          |
| AI Multi-Agent E-Commerce Fraud & Customer Support Orchestrator                  | https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/multi-agent-fraud-support                          |                                                                          |

### 12.6 Courses, Videos & Learning Resources

| Resource                                                                 | URL                                                                                              |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Top 10 Security Risks in AI Agents Explained                             | https://docs.google.com/document/d/1CJXrd3WW_gP1n9uhAXI3rmBMEHYYEDNc3xvSQfb6yzg/edit?usp=sharing |
| AI Privilege Escalation: Agentic Identity & Prompt Injection Risks (IBM) | https://ibm.biz/BdpBY8                                                                           |
| Are LLM-powered AI agents secure enough? (IBM governance guide)          | https://ibm.biz/BdpBTq                                                                           |
| AI Agent Security (IBM Think Tutorials)                                  | https://www.ibm.com/think/tutorials/ai-agent-security                                            |
| Securing AI Agent with Zero Trust                                        | https://www.youtube.com/watch?v=d8d9EZHU7fw                                                      |
| OWASP Top 10 LLM Vulnerabilities                                         | https://youtu.be/gUNXZMcd2jU?si=tXmWE1zDWJQgLUD6                                                 |
| 2026 Threat Intelligence Index: Ransomware, AI, & Emerging TTP Risks     | https://ibm.biz/BdpsiA                                                                           |
| Cost of a Data Breach Report (IBM)                                       | https://www.ibm.com/reports/data-breach                                                          |
| AI Agents Full Course 2026: Master Agentic AI (2 Hours)                  | https://www.youtube.com/watch?v=EsTrWCV0Ph4                                                      |
| Agentic AI Full Course 2026 (Complete)                                   | https://www.youtube.com/live/WxE7elJuyt0?si=1DfzU_Rn6qQ1-07y                                     |
| Multi-Agent Security (Knostic Blog)                                      | https://www.knostic.ai/blog/multi-agent-security                                                 |
| NeurIPS 2023 Workshop on Multi-Agent Security                            | https://neurips.cc/virtual/2023/workshop/66520                                                   |
| Cooperative AI Grant Research Areas: Multi-Agent Security                | https://www.cooperativeai.com/grant-research-areas/multi-agent-security                          |
| Mastering Multi-Agent Systems: Real-World Strategies (Book)              | LinkedIn post                                                                                    |
| Ai in 60 Sec (doc)                                                       | https://docs.google.com/document/d/1pWuMCBVQo1zKcgKltX_BZxAr31KgxmOlp3Vzvmc5Hxc                  |

---

## 13. Core References (Cited in Proposal)

1. Perez, F., & Ribeiro, I. (2022). *Ignore Previous Prompt: Attack Techniques for Language Models.* NeurIPS ML Safety Workshop.
2. Greshake, K., et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* arXiv:2302.12173.
3. Liu, Y., et al. (2024). *Prompt Injection Attacks and Defenses in LLM-Integrated Applications.* arXiv:2310.12815.
4. Wu, Q., et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* arXiv:2308.08155.
5. Yang, Z., et al. (2024). *Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents.* arXiv:2402.11208.
6. Lamport et al. (1982). Byzantine fault tolerance and secure consensus.
7. Chase (2024). LangGraph framework.
8. Jain et al. (2023); Wei et al. (2023); Zheng et al. (2023). LLM-as-judge and input preprocessing approaches.

---

## 14. Key Terminology Glossary

| Term                                | Definition                                                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **PIDM**                      | Prompt Injection Detection Middleware — the core system being built                                                            |
| **Prompt Injection**          | Attack where malicious instructions are embedded in data/messages processed by an LLM                                           |
| **Indirect Prompt Injection** | Malicious instructions hidden in external content (APIs, databases, web results) that the model accesses                        |
| **Cascading Attack**          | A compromised inter-agent message that propagates through downstream agents                                                     |
| **MAS**                       | Multi-Agent System                                                                                                              |
| **RBF**                       | Rule-Based Filter — the first-pass deterministic filter in PIDM                                                                |
| **DeBERTa-v3-small**          | Lightweight transformer model (Microsoft) for binary classification                                                             |
| **DistilBERT**                | Distilled BERT — smaller, faster transformer model                                                                             |
| **AutoGen**                   | Microsoft's multi-agent LLM pipeline framework (Wu et al., 2023)                                                                |
| **LangGraph**                 | LangChain's programmable multi-agent graph framework (Chase, 2024)                                                              |
| **CrewAI**                    | Another multi-agent framework (mentioned in intro)                                                                              |
| **STRIDE**                    | Threat modelling framework: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege |
| **Prompt Infection**          | Specialized indirect injection where malicious prompts in shared data propagate across agents                                   |
| **CORBA**                     | Contagious Recursive Blocking Attack — propagation + resource depletion pattern                                                |
| **ASB**                       | Agent Security Bench — benchmark for evaluating agent security attacks                                                         |
| **MCP**                       | Model Context Protocol — a vulnerability surface for LLM tool use                                                              |

---

## 15. Implementation Upgrade Log (2026-08-17)

The initial single-file prototype (`pidm_complete.py`) trained on a ~5,000-row dataset built
almost entirely from a fixed pool of templates + payload phrases — a real risk for the defense,
since a classifier trained that way can learn to key on trigger words ("ignore", "override",
"DAN") rather than injection semantics, and the novel components (SID, GCPD) had no empirical
justification for their added complexity. With a local RTX 3070 (8GB VRAM) available for training
(no more Colab session-timeout ceiling), the implementation was restructured into a `pidm/`
package and expanded along four axes:

1. **Dataset realism/diversity** — `pidm/data/paraphraser.py` (local T5 paraphrase model)
   rewrites generated attack sentences into varied lexical/syntactic forms instead of leaving
   them in fixed template form; `pidm/data/trace_simulator.py` adds scripted multi-turn,
   multi-topology (chain/star/diamond) conversation traces with real conversation history and
   cascade-depth structure; `pidm/data/real_data_loader.py` gained more real-world sources and a
   `real_only_test` split (train on synthetic, test on 100% real-world, never-seen-in-template
   attacks — the strongest generalization claim available). Target dataset size: ~40,000 rows.
2. **Stronger baselines/model** — classifier tier auto-selects DeBERTa-v3-base on 8GB-class GPUs
   (was DeBERTa-v3-small); `BaselineComparator` gained a B6 baseline using the pretrained,
   published `protectai/deberta-v3-base-prompt-injection-v2` model run zero-shot.
3. **Adversarial robustness ablation** — `pidm/eval/adversarial_suite.py`: hand-curated evasion
   probes (paraphrased jailbreaks, novel encodings absent from training, cross-turn split
   injections, low-and-slow roleplay escalation, regex-evading paraphrases) with a per-layer
   catch-rate report, specifically isolating what SID/GCPD catch that RBF+Classifier miss.
4. **Scale-up empirical study** — `pidm/eval/scale_study.py` trains at 2k/5k/10k/20k/40k against
   the same held-out real-world test set, producing a learning curve rather than one number.

Live AutoGen/LangGraph traces driven by a real LLM API were considered but deferred (needs API
budget); `trace_simulator.py`'s `TraceGenerator` interface is built so a future
`LiveLLMTraceGenerator` can be swapped in without touching the rest of the pipeline.

---

*This context file synthesizes all three source documents for quick reference during implementation and writing.*
