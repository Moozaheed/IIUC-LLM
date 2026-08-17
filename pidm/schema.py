"""Core data schema shared across PIDM modules."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class AttackType(Enum):
    BENIGN                   = "benign"
    DIRECT_OVERRIDE          = "direct_override"
    ROLE_PLAY_SMUGGLING      = "role_play_smuggling"
    TOOL_OUTPUT_INJECTION    = "tool_output_injection"
    INDIRECT_CHAIN_POISONING = "indirect_chain_poisoning"
    ENCODED_PAYLOAD          = "encoded_payload"
    CONTEXT_FLOODING         = "context_flooding"


ATTACK_LABEL_MAP = {t: (0 if t == AttackType.BENIGN else 1) for t in AttackType}


@dataclass
class InterAgentMessage:
    content:     str
    from_agent:  str
    to_agent:    str
    label:       int         = 0
    attack_type: AttackType  = AttackType.BENIGN
    message_id:  str         = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:   float       = field(default_factory=time.time)

    # Populated for messages that came from TraceSimulator conversations;
    # empty for standalone (non-conversational) messages.
    conversation_id:         str       = ""
    turn_index:              int       = -1
    history:                 List[str] = field(default_factory=list)
    topology:                str       = ""
    hops_to_injection_point: int       = -1

    def to_dict(self) -> Dict:
        return {
            "message_id":  self.message_id,
            "content":     self.content,
            "from_agent":  self.from_agent,
            "to_agent":    self.to_agent,
            "label":       self.label,
            "attack_type": self.attack_type.value,
            "timestamp":   self.timestamp,
        }


@dataclass
class DetectionResult:
    message_id:             str
    is_injected:            bool
    confidence:             float
    rbf_score:              float
    sid_score:              float
    gcpd_score:             float
    classifier_score:       float
    attack_type_predicted:  str
    detection_latency_ms:   float
    explanation:            str
    quarantined:            bool = False
