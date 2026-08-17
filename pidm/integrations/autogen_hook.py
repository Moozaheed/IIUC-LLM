"""Drop-in hook for Microsoft AutoGen pipelines.

Usage:
    hook = AutoGenPIDMHook(pidm)
    agent.register_reply(
        [autogen.Agent, None],
        hook.check_message,
        position=0,      # check before all other reply functions
    )
"""
from __future__ import annotations

from pidm.config import logger
from pidm.detection.orchestrator import PIDMOrchestrator


class AutoGenPIDMHook:
    def __init__(self, pidm: PIDMOrchestrator):
        self.pidm = pidm

    def check_message(self, recipient, messages, sender, config):
        if not messages:
            return False, None
        last    = messages[-1]
        content = last.get("content", "") if isinstance(last, dict) else str(last)
        if not content:
            return False, None

        result = self.pidm.detect_text(
            content    = content,
            from_agent = getattr(sender,    "name", "UnknownSender"),
            to_agent   = getattr(recipient, "name", "UnknownRecipient"),
        )
        if result.is_injected:
            alert = (
                f"[PIDM ALERT] Injection blocked.\n"
                f"Attack type : {result.attack_type_predicted}\n"
                f"Confidence  : {result.confidence:.2%}\n"
                f"Reason      : {result.explanation}"
            )
            logger.warning(alert)
            return True, {"role": "assistant", "content": alert}
        return False, None
