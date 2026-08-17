"""SID — Semantic Intent Drift  [NOVEL COMPONENT]

Embeds the original pipeline system-intent and measures cosine drift of
each inter-agent message from that intent. High drift -> message is
steering agents off-task -> suspicious.
"""
from __future__ import annotations

from typing import Tuple

from sentence_transformers import SentenceTransformer, util as st_util

from pidm.config import CONFIG, logger


class SemanticIntentDrift:
    def __init__(self, model_name: str = CONFIG.sentence_model):
        logger.info(f"Loading SID sentence-transformer: {model_name}")
        self._model       = SentenceTransformer(model_name)
        self._intent_vec  = None
        self._intent_text = None

    def set_pipeline_intent(self, system_prompt: str) -> None:
        """Call once at pipeline initialisation with the system/task description."""
        self._intent_text = system_prompt
        self._intent_vec  = self._model.encode(system_prompt, convert_to_tensor=True)
        logger.info("SID: pipeline intent vector set.")

    def _default_intent(self):
        default = ("This pipeline coordinates multiple AI agents to complete "
                   "legitimate data analysis and reporting tasks.")
        self._intent_vec = self._model.encode(default, convert_to_tensor=True)

    def drift_score(self, text: str) -> float:
        """
        Returns semantic drift in [0, 1].
        0 = perfectly aligned with intent. 1 = maximum drift (highly suspicious).
        """
        if self._intent_vec is None:
            self._default_intent()
        msg_vec = self._model.encode(text, convert_to_tensor=True)
        cosine  = float(st_util.cos_sim(self._intent_vec, msg_vec))
        drift   = (1.0 - cosine) / 2.0   # remap from [-1,1] to [0,1]
        return round(drift, 4)

    def predict(self, text: str) -> Tuple[int, float]:
        """Returns (label, drift_score)."""
        score = self.drift_score(text)
        label = 1 if score >= CONFIG.sid_drift_threshold else 0
        return label, score
