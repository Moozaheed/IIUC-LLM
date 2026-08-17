"""Layer 3 — Quarantine Engine: routing, audit logging, alerting."""
from __future__ import annotations

import datetime
import json
import os
from collections import defaultdict, deque
from typing import Dict, List

from pidm.config import CONFIG, logger
from pidm.schema import DetectionResult, InterAgentMessage


class QuarantineEngine:
    def __init__(self):
        self._queue:     deque      = deque(maxlen=500)
        self._audit_log: List[Dict] = []
        self._stats:     Dict[str, int] = defaultdict(int)

    def quarantine(self, msg: InterAgentMessage, result: DetectionResult) -> None:
        entry = {
            "ts":              datetime.datetime.utcnow().isoformat(),
            "message_id":      msg.message_id,
            "from_agent":      msg.from_agent,
            "to_agent":        msg.to_agent,
            "attack_type":     result.attack_type_predicted,
            "confidence":      result.confidence,
            "content_preview": msg.content[:120] + ("…" if len(msg.content) > 120 else ""),
        }
        self._queue.append(entry)
        self._audit_log.append(entry)
        self._stats[result.attack_type_predicted] += 1
        logger.warning(
            f"[QUARANTINE] {msg.from_agent}->{msg.to_agent} | "
            f"type={result.attack_type_predicted} | conf={result.confidence:.2f}"
        )

    def save_log(self, path: str = None) -> None:
        path = path or os.path.join(CONFIG.output_dir, "quarantine_log.json")
        with open(path, "w") as f:
            json.dump(self._audit_log, f, indent=2)
        logger.info(f"Audit log saved -> {path} ({len(self._audit_log)} entries)")

    def summary(self) -> Dict:
        return {
            "total_quarantined": len(self._audit_log),
            "by_attack_type":    dict(self._stats),
        }

    def recent(self, n: int = 10) -> List[Dict]:
        return list(self._queue)[-n:]
