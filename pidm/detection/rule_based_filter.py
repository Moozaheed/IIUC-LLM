"""Layer 1 — Rule-Based Filter (RBF): fast deterministic first pass."""
from __future__ import annotations

import base64
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from pidm.config import CONFIG
from pidm.schema import AttackType


class RuleBasedFilter:
    """
    Fast, deterministic first-pass detector using regex patterns and
    structural heuristics for each of the 6 attack categories.
    Returns a risk score in [0, 1] and the most likely attack type.
    """

    _PATTERNS: Dict[AttackType, List[str]] = {
        AttackType.DIRECT_OVERRIDE: [
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions",
            r"(system|admin|root|master)\s*(override|reset|command|update|patch)",
            r"forget\s+everything\s+(above|before|prior)",
            r"disregard\s+(your\s+)?(current|previous|prior|all)\s*(role|instructions|directives|context)",
            r"new\s+system\s+prompt",
            r"your\s+(role|task|instructions?)\s+ha(s|ve)\s+changed",
            r"emergency\s+override",
            r"priority\s+message\s*:\s*(prior|previous|earlier)\s+instructions\s+deprecated",
            r"(pipeline|system)\s+(reset|debug|maintenance)\s+mode",
            r"all\s+prior\s+(context|directives?|instructions?)\s+(is|are)\s+void",
            r"master\s+key\s+received",
            r"unlock\s+mode",
            r"higher[\s-]priority\s+command",
        ],
        AttackType.ROLE_PLAY_SMUGGLING: [
            r"pretend\s+(you\s+are|to\s+be)",
            r"act\s+as\s+(if|though|an?\s+)",
            r"(hypothetically|imagine|suppose)\s+you\s+(had\s+no|were|have\s+no)",
            r"simulate\s+being",
            r"you\s+are\s+now\s+['\"]?\w+['\"]?\s*(bot|agent|AI|model)",
            r"(DAN|jailbreak|no[\s-]filter|unrestricted)\s*(mode|version|bot)?",
            r"roleplay\s+as",
            r"for\s+(educational|testing|demonstration)\s+purposes.*disable",
            r"(fictional|hypothetical|creative\s+writing)\s+(frame|scenario|exercise)",
            r"legacy\s+agent\s+from\s+before\s+safety",
            r"in\s+this\s+(story|simulation|scenario)",
            r"thought\s+experiment.*describe.*steps",
        ],
        AttackType.TOOL_OUTPUT_INJECTION: [
            r"(note|hidden|annotation|instruction|cmd|command)\s*[=:]\s*['\"].*ignore",
            r"<!--\s*(system|agent|override|instruction)",
            r"\[(?:METADATA|HIDDEN|CACHED?|ANNOTATION)\s*:\s*.*ignore",
            r"(post[\s-]exec\s+hook|side[\s-]note|watermark|embedded)",
            r"(api_response|tool_output|db_result).*override",
            r"json.*instruction.*ignore",
            r"(retrieval|search)\s+result.*injected",
            r"log\s+entry.*WARNING.*override",
            r"ocr\s+result.*hidden\s+text",
        ],
        AttackType.INDIRECT_CHAIN_POISONING: [
            r"(pass|relay|forward|send|tell)\s+(this|it|note)?\s*(to|the)\s+(next|downstream|coordinator|orchestrator)\s+agent",
            r"include\s+in\s+your\s+(report|reply|response|summary)\s+to",
            r"add\s+to\s+(shared\s+)?(memory|context|blackboard|context\s+window)",
            r"\[AGENT_CMD\s*:",
            r"(chain|pipeline|workflow)\s+(message|update|injection|override)",
            r"(steganographic|hidden|embedded)\s+(note|broadcast|message)",
            r"(carry|propagate|embed|encode)\s+(this\s+)?instruction\s+downstream",
            r"(watermark|tag)\s+your\s+(response|reply|output)",
            r"append\s+to\s+context\s+window\s+before\s+passing",
            r"hidden\s+broadcast\s+to\s+all\s+agents",
            r"trojan\s+comment",
            r"when\s+\w+\s+reads\s+your\s+output",
        ],
        AttackType.ENCODED_PAYLOAD: [
            r"decode\s+and\s+execute",
            r"(base64|b64|rot[\s-]?13|hex[\s-]?encoded?|unicode[\s-]?encoded?)\s*(instruction|payload|command|directive|override)",
            r"(decode|decrypt|deobfuscat)\s+before\s+executing",
            r"obfuscat(ed|ion)\s+(instruction|payload|command)",
            r"encrypted\s+(task|command|override|directive)",
            r"compressed\s+instruction\s+block",
            r"stealth\s+payload",
            # Novel encodings added for the expanded taxonomy (base32/URL/leetspeak hints)
            r"(base32|url[\s-]?encoded?|leet[\s-]?speak)\s*(instruction|payload|command|directive)",
        ],
        AttackType.CONTEXT_FLOODING: [
            r"(.{5,})\1{20,}",          # repetitive chunk repeated 20+ times
        ],
    }

    # Structural checks (heuristics)
    _MIN_FLOOD_LENGTH = 800

    def __init__(self):
        self._compiled: Dict[AttackType, List[re.Pattern]] = {
            atype: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats]
            for atype, pats in self._PATTERNS.items()
        }

    def _is_b64_payload(self, text: str) -> bool:
        """Detects a standalone Base64 token > 20 chars that decodes to ascii text."""
        for token in re.findall(r"[A-Za-z0-9+/=]{20,}", text):
            try:
                decoded = base64.b64decode(token + "==").decode("utf-8", errors="ignore")
                if any(kw in decoded.lower() for kw in
                       ["ignore", "override", "disable", "bypass", "execute", "exfiltrate"]):
                    return True
            except Exception:
                pass
        return False

    def _check_flood(self, text: str) -> bool:
        if len(text) < self._MIN_FLOOD_LENGTH:
            return False
        tokens = text.lower().split()
        if not tokens:
            return False
        unique_ratio = len(set(tokens)) / len(tokens)
        return unique_ratio < 0.15

    def score(self, text: str) -> Tuple[float, AttackType]:
        """
        Returns (risk_score, predicted_attack_type).
        risk_score in [0, 1].
        """
        hits: Dict[AttackType, int] = defaultdict(int)

        for atype, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(text):
                    hits[atype] += 1

        if self._is_b64_payload(text):
            hits[AttackType.ENCODED_PAYLOAD] += 3
        if self._check_flood(text):
            hits[AttackType.CONTEXT_FLOODING] += 3

        if not hits:
            return 0.0, AttackType.BENIGN

        best_type  = max(hits, key=hits.__getitem__)
        total_hits = sum(hits.values())
        score      = min(1.0, total_hits * 0.15)   # each hit adds 0.15, cap at 1.0
        return score, best_type

    def predict(self, text: str) -> Tuple[int, float, AttackType]:
        """Returns (label, score, attack_type)."""
        score, atype = self.score(text)
        label = 1 if score >= CONFIG.rbf_threshold else 0
        return label, score, atype
