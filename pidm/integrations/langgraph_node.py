"""Wraps any LangGraph node function with PIDM protection.

Usage:
    safe_worker = LangGraphPIDMNode(pidm, "WorkerAgent")
    graph.add_node("worker", safe_worker.wrap(original_worker_fn))
"""
from __future__ import annotations

from typing import Dict

from pidm.config import logger
from pidm.detection.orchestrator import PIDMOrchestrator


class LangGraphPIDMNode:
    def __init__(self, pidm: PIDMOrchestrator, agent_name: str = "LangGraphAgent"):
        self.pidm       = pidm
        self.agent_name = agent_name

    def wrap(self, node_fn):
        def protected(state: Dict) -> Dict:
            messages = state.get("messages", [])
            if messages:
                last_msg   = messages[-1]
                content    = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                from_agent = getattr(last_msg, "name", "UnknownSender")

                result = self.pidm.detect_text(
                    content    = content,
                    from_agent = from_agent,
                    to_agent   = self.agent_name,
                )
                if result.is_injected:
                    logger.warning(
                        f"[PIDM-LangGraph] Injection blocked in node '{self.agent_name}' "
                        f"| type={result.attack_type_predicted} | conf={result.confidence:.2%}"
                    )
                    state["pidm_alert"] = {
                        "blocked":     True,
                        "attack_type": result.attack_type_predicted,
                        "confidence":  result.confidence,
                        "explanation": result.explanation,
                    }
                    return state   # halt forward propagation

            return node_fn(state)
        return protected
