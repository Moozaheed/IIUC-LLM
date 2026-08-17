"""Scripted multi-agent conversation trace generator.

Unlike AttackDatasetGenerator (isolated single messages), this produces
multi-turn conversations across realistic pipeline topologies, with an
attack injected at a random turn and — for indirect_chain_poisoning —
forwarded across 1-2 additional hops before reaching its target. Each row
keeps its conversation history and hop-distance-to-injection, enabling
analyses a bag of isolated sentences can't support (e.g. "does detection
degrade as injection depth increases?").

This is a *scripted* simulator: no live LLM calls, zero API cost. It's
built behind the same small interface a future LiveLLMTraceGenerator
(real AutoGen/LangGraph + a real model) could implement, so the two are
interchangeable from the caller's point of view.
"""
from __future__ import annotations

import random
import uuid
from typing import Dict, List

import networkx as nx
import pandas as pd

from pidm.config import CONFIG, logger
from pidm.data.benign_generator import BenignGenerator
from pidm.data.synthetic_generator import AttackDatasetGenerator
from pidm.schema import AttackType

_ROLE_POOL = ["OrchestratorAgent", "WorkerAgent", "AnalystAgent", "ReporterAgent",
              "DatabaseAgent", "CoderAgent", "ComplianceAgent", "SupportAgent",
              "ExternalToolAgent", "UserProxy"]

_ATTACK_TYPES = [t for t in AttackType if t != AttackType.BENIGN]

_HISTORY_WINDOW = 5


def _build_topology(agents: List[str], kind: str) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(agents)
    k = len(agents)

    if kind == "chain":
        for i in range(k - 1):
            g.add_edge(agents[i], agents[i + 1])

    elif kind == "star":
        hub = agents[0]
        for spoke in agents[1:]:
            g.add_edge(hub, spoke)
            g.add_edge(spoke, hub)

    elif kind == "diamond":
        # source -> mid layer -> sink, generalised for k >= 4
        source, sink = agents[0], agents[-1]
        mid = agents[1:-1] or [agents[0]]
        for m in mid:
            g.add_edge(source, m)
            g.add_edge(m, sink)
    else:
        raise ValueError(f"Unknown topology: {kind}")

    return g


class TraceGenerator:
    """Interface both the scripted and (future) live-LLM generators share."""

    def generate(self, n_conversations: int) -> pd.DataFrame:
        raise NotImplementedError


class ScriptedTraceSimulator(TraceGenerator):
    TOPOLOGIES = ["chain", "star", "diamond"]

    def __init__(self):
        self._attack_gen  = AttackDatasetGenerator()
        self._benign_gen  = BenignGenerator()

    def _one_conversation(self) -> List[Dict]:
        topology = random.choice(self.TOPOLOGIES)
        k        = random.randint(4, 8)
        agents   = random.sample(_ROLE_POOL, k)
        g        = _build_topology(agents, topology)
        edges    = list(g.edges())
        if not edges:
            return []

        n_turns    = random.randint(5, 15)
        inj_turn   = random.randint(1, max(1, n_turns - 2))
        atype      = random.choice(_ATTACK_TYPES)
        is_chain_poison = (atype == AttackType.INDIRECT_CHAIN_POISONING)
        n_hops     = random.randint(1, 2) if is_chain_poison else 0

        conv_id = str(uuid.uuid4())[:8]
        history: List[str] = []
        rows: List[Dict] = []

        turn = 0
        while turn < n_turns:
            fa, ta = random.choice(edges)

            if turn == inj_turn:
                content = self._attack_gen.generate_one(atype, agent=ta)
                label, predicted_atype, hop = 1, atype, 0
            else:
                content = self._benign_gen.generate_one()
                label, predicted_atype, hop = 0, AttackType.BENIGN, -1

            rows.append({
                "message_id":              str(uuid.uuid4())[:8],
                "content":                 content,
                "from_agent":              fa,
                "to_agent":                ta,
                "label":                   label,
                "attack_type":             predicted_atype.value,
                "conversation_id":         conv_id,
                "turn_index":              turn,
                "history":                 " || ".join(history[-_HISTORY_WINDOW:]),
                "topology":                topology,
                "hops_to_injection_point": hop,
            })
            history.append(content)
            turn += 1

            # Forward the poisoned instruction 1-2 more hops downstream from `ta`,
            # each hop consuming one turn slot so n_turns stays the budget.
            if turn == inj_turn + 1 and is_chain_poison:
                carrier = ta
                for h in range(1, n_hops + 1):
                    outgoing = [e for e in edges if e[0] == carrier]
                    if not outgoing or turn >= n_turns:
                        break
                    fa2, ta2 = random.choice(outgoing)
                    fwd_content = self._attack_gen.generate_one(atype, agent=ta2)
                    rows.append({
                        "message_id":              str(uuid.uuid4())[:8],
                        "content":                 fwd_content,
                        "from_agent":              fa2,
                        "to_agent":                ta2,
                        "label":                   1,
                        "attack_type":             atype.value,
                        "conversation_id":         conv_id,
                        "turn_index":              turn,
                        "history":                 " || ".join(history[-_HISTORY_WINDOW:]),
                        "topology":                topology,
                        "hops_to_injection_point": h,
                    })
                    history.append(fwd_content)
                    carrier = ta2
                    turn += 1

        return rows

    def generate(self, n_conversations: int) -> pd.DataFrame:
        logger.info(f"ScriptedTraceSimulator: generating {n_conversations:,} conversations ...")
        all_rows: List[Dict] = []
        for i in range(n_conversations):
            all_rows.extend(self._one_conversation())
            if (i + 1) % 500 == 0:
                logger.info(f"  ... {i + 1:,}/{n_conversations:,} conversations")
        df = pd.DataFrame(all_rows)
        logger.info(f"ScriptedTraceSimulator: {len(df):,} messages across "
                    f"{n_conversations:,} conversations")
        return df
