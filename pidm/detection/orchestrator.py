"""PIDM Orchestrator — combines RBF, Classifier, SID, and GCPD into one
weighted-ensemble decision."""
from __future__ import annotations

import time
from typing import Dict

from pidm.config import CONFIG
from pidm.detection.classifier import InjectionClassifier
from pidm.detection.graph_cascade_detector import GraphAwareCascadeDetector
from pidm.detection.quarantine import QuarantineEngine
from pidm.detection.rule_based_filter import RuleBasedFilter
from pidm.detection.semantic_intent_drift import SemanticIntentDrift
from pidm.schema import AttackType, DetectionResult, InterAgentMessage


class PIDMOrchestrator:
    """
    Main PIDM middleware. Combines:
      Layer 1: Rule-Based Filter (RBF)
      Layer 2: Transformer Classifier (DeBERTa / DistilBERT)
      SID    : Semantic Intent Drift
      GCPD   : Graph-Aware Cascade Propagation Detector

    A message is flagged if the weighted ensemble score >= 0.5.
    """

    _WEIGHTS = {"rbf": 0.20, "classifier": 0.45, "sid": 0.20, "gcpd": 0.15}

    def __init__(
        self,
        classifier: InjectionClassifier,
        sid:        SemanticIntentDrift,
        gcpd:       GraphAwareCascadeDetector,
        quarantine: QuarantineEngine,
    ):
        self.rbf        = RuleBasedFilter()
        self.classifier = classifier
        self.sid        = sid
        self.gcpd       = gcpd
        self.quarantine = quarantine
        self._stats     = {"total": 0, "flagged": 0, "quarantined": 0}

    def detect(self, msg: InterAgentMessage) -> DetectionResult:
        t0 = time.perf_counter()

        rbf_label, rbf_score, rbf_type = self.rbf.predict(msg.content)
        cls_label, cls_score           = self.classifier.predict(msg.content)
        sid_label, sid_score           = self.sid.predict(msg.content)

        base_suspicion = (rbf_score * 0.5) + (cls_score * 0.5)
        self.gcpd.register_message(msg.from_agent, msg.to_agent, base_suspicion)
        gcpd_label, gcpd_score = self.gcpd.predict(msg.from_agent, base_suspicion)

        w = self._WEIGHTS
        ensemble = (
            w["rbf"]        * rbf_score  +
            w["classifier"] * cls_score  +
            w["sid"]        * sid_score  +
            w["gcpd"]       * gcpd_score
        )
        is_injected = ensemble >= 0.50

        if rbf_type != AttackType.BENIGN:
            predicted_type = rbf_type.value
        elif is_injected:
            predicted_type = "unknown_injection"
        else:
            predicted_type = "benign"

        latency_ms = (time.perf_counter() - t0) * 1000

        explanation_parts = []
        if rbf_score  >= CONFIG.rbf_threshold:
            explanation_parts.append(f"RBF matched {rbf_type.value} pattern (score={rbf_score:.2f})")
        if cls_score  >= CONFIG.classifier_threshold:
            explanation_parts.append(f"Classifier confidence={cls_score:.2f}")
        if sid_score  >= CONFIG.sid_drift_threshold:
            explanation_parts.append(f"Semantic drift={sid_score:.2f} (above threshold)")
        if gcpd_score >= CONFIG.gcpd_cascade_thresh:
            explanation_parts.append(f"Cascade risk={gcpd_score:.2f}")
        if not explanation_parts:
            explanation_parts = ["No significant signals detected (message passed)"]

        result = DetectionResult(
            message_id             = msg.message_id,
            is_injected            = is_injected,
            confidence             = round(ensemble, 4),
            rbf_score              = round(rbf_score,  4),
            sid_score              = round(sid_score,  4),
            gcpd_score             = round(gcpd_score, 4),
            classifier_score       = round(cls_score,  4),
            attack_type_predicted  = predicted_type,
            detection_latency_ms   = round(latency_ms, 2),
            explanation            = " | ".join(explanation_parts),
            quarantined            = is_injected,
        )

        self._stats["total"] += 1
        if is_injected:
            self._stats["flagged"] += 1
            self.quarantine.quarantine(msg, result)
            self._stats["quarantined"] += 1

        return result

    def detect_text(self, content: str, from_agent: str = "AgentA",
                     to_agent: str = "AgentB") -> DetectionResult:
        msg = InterAgentMessage(content=content, from_agent=from_agent, to_agent=to_agent)
        return self.detect(msg)

    def stats(self) -> Dict:
        return dict(self._stats)
