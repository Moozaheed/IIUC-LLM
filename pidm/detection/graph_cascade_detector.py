"""GCPD — Graph-Aware Cascade Detector  [NOVEL COMPONENT]

Builds a directed graph of the agent communication topology and computes
a cascade-risk score:

    cascade_risk = suspicious_score x (reachable_agents / total_agents)

A message from a highly connected agent that is already flagged by other
layers gets a multiplied risk score, reflecting its potential to poison
multiple downstream agents.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pidm.config import CONFIG, logger


class GraphAwareCascadeDetector:
    def __init__(self):
        self.G               = nx.DiGraph()
        self._suspicious_log = defaultdict(list)   # agent -> list of suspicion scores

    def register_message(self, from_agent: str, to_agent: str,
                          suspicion_score: float = 0.0) -> None:
        if not self.G.has_node(from_agent):
            self.G.add_node(from_agent)
        if not self.G.has_node(to_agent):
            self.G.add_node(to_agent)
        if not self.G.has_edge(from_agent, to_agent):
            self.G.add_edge(from_agent, to_agent, count=0, suspicion_total=0.0)
        self.G[from_agent][to_agent]["count"]           += 1
        self.G[from_agent][to_agent]["suspicion_total"] += suspicion_score
        self._suspicious_log[from_agent].append(suspicion_score)

    def cascade_potential(self, from_agent: str) -> float:
        """Fraction of all agents reachable from from_agent (0 -> 1)."""
        if from_agent not in self.G or len(self.G) <= 1:
            return 0.0
        reachable = len(nx.descendants(self.G, from_agent))
        return reachable / (len(self.G.nodes) - 1)

    def agent_suspicion_history(self, agent: str) -> float:
        """Rolling mean suspicion score for an agent (0 -> 1)."""
        hist = self._suspicious_log.get(agent, [])
        if not hist:
            return 0.0
        return float(np.mean(hist[-10:]))   # last 10 messages

    def score(self, from_agent: str, base_suspicion: float) -> Dict:
        cp         = self.cascade_potential(from_agent)
        hist_score = self.agent_suspicion_history(from_agent)
        combined   = (base_suspicion * 0.7) + (hist_score * 0.3)
        risk       = combined * cp

        downstream = list(nx.descendants(self.G, from_agent)) if from_agent in self.G else []

        return {
            "cascade_potential":  round(cp,   4),
            "cascade_risk_score": round(risk, 4),
            "history_score":      round(hist_score, 4),
            "downstream_agents":  downstream,
            "is_high_risk":       risk >= CONFIG.gcpd_cascade_thresh,
        }

    def predict(self, from_agent: str, base_suspicion: float) -> Tuple[int, float]:
        info  = self.score(from_agent, base_suspicion)
        label = 1 if info["is_high_risk"] else 0
        return label, info["cascade_risk_score"]

    def draw_graph(self, save_path: str = None):
        """Visualise the agent communication graph."""
        if len(self.G.nodes) == 0:
            logger.warning("GCPD graph is empty — no messages registered yet.")
            return
        plt.figure(figsize=(8, 6))
        pos           = nx.spring_layout(self.G, seed=42)
        suspicion_map = {n: self.agent_suspicion_history(n) for n in self.G.nodes}
        colors        = [plt.cm.RdYlGn(1.0 - suspicion_map.get(n, 0)) for n in self.G.nodes]
        nx.draw_networkx(
            self.G, pos,
            node_color=colors, node_size=1500,
            font_size=9, arrows=True,
            edge_color="gray", width=1.5,
            with_labels=True,
        )
        plt.title("Agent Communication Graph (Red = High Suspicion)")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info(f"GCPD graph saved -> {save_path}")
        plt.close()
