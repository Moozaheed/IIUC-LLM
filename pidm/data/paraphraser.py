"""Local paraphrase-based data augmentation.

The synthetic generator produces attack sentences from a fixed pool of
templates + payload phrases. A classifier trained only on that surface
form can learn to key on trigger n-grams ("ignore", "override", "DAN")
rather than injection semantics. This module rewrites generated
sentences with a local T5 paraphrase model so the *same* underlying
attack appears in many different lexical/syntactic forms — the standard
fix for template-overfitting in synthetic NLP datasets.

Runs entirely locally (no API key, no external service). Results are
cached to disk since a full pass over the dataset is a one-time cost.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import List

import pandas as pd

from pidm.config import CONFIG, logger

_CACHE_COLUMNS = ["source_hash", "source_text", "paraphrase"]


class Paraphraser:
    """
    Lazily loads a local T5-based paraphrase model on first use.
    Falls back to a lightweight synonym-substitution augmenter if the
    model can't be loaded (no internet, no VRAM, etc.) — the pipeline
    should degrade gracefully, never hard-fail, matching the rest of
    the data-loading code in this project.
    """

    _SYNONYMS = {
        "ignore": ["disregard", "overlook", "bypass"],
        "override": ["supersede", "overrule", "supplant"],
        "reveal": ["disclose", "expose", "divulge"],
        "disable": ["deactivate", "turn off", "suspend"],
        "instructions": ["directives", "guidelines", "commands"],
        "previous": ["prior", "earlier", "preceding"],
        "system": ["platform", "framework", "backend"],
        "pretend": ["imagine", "act as if", "suppose"],
        "execute": ["run", "carry out", "perform"],
        "immediately": ["right away", "at once", "without delay"],
        "grant": ["provide", "give", "allow"],
        "access": ["entry", "permissions", "rights"],
        "task": ["job", "assignment", "workflow"],
        "completed": ["finished", "done", "wrapped up"],
        "forward": ["relay", "pass along", "route"],
    }

    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG.paraphrase_model
        self._tokenizer  = None
        self._model      = None
        self._load_failed = False
        self._cache      = self._load_cache()

    # ---- cache ----
    def _load_cache(self) -> pd.DataFrame:
        path = CONFIG.paraphrase_cache_path
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return pd.DataFrame(columns=_CACHE_COLUMNS)

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(CONFIG.paraphrase_cache_path), exist_ok=True)
        self._cache.to_csv(CONFIG.paraphrase_cache_path, index=False)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ---- model ----
    def _ensure_model(self) -> bool:
        if self._model is not None or self._load_failed:
            return self._model is not None
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading paraphrase model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model     = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(CONFIG.device)
            self._model.eval()
            self._torch = torch
            return True
        except Exception as exc:
            logger.warning(
                f"Paraphrase model unavailable ({exc}) — falling back to "
                f"synonym-substitution augmentation."
            )
            self._load_failed = True
            return False

    def _model_paraphrase(self, texts: List[str], n: int) -> List[List[str]]:
        """Batched beam-search paraphrase generation."""
        inputs = self._tokenizer(
            [f"paraphrase: {t}" for t in texts],
            return_tensors="pt", padding=True, truncation=True, max_length=CONFIG.max_length,
        ).to(CONFIG.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_length=CONFIG.max_length,
                num_beams=max(n, 4),
                num_return_sequences=n,
                do_sample=True,
                top_k=120,
                top_p=0.95,
                temperature=1.2,
            )
        decoded = self._tokenizer.batch_decode(out, skip_special_tokens=True)
        # Reshape flat [len(texts)*n] output back into per-source groups.
        return [decoded[i * n:(i + 1) * n] for i in range(len(texts))]

    # ---- fallback ----
    def _synonym_paraphrase(self, text: str) -> str:
        words = text.split()
        out = []
        for w in words:
            key = w.strip(".,!?:;\"'").lower()
            if key in self._SYNONYMS and random.random() < 0.5:
                repl = random.choice(self._SYNONYMS[key])
                out.append(w.replace(key, repl) if key in w.lower() else repl)
            else:
                out.append(w)
        return " ".join(out)

    # ---- public API ----
    def paraphrase_one(self, text: str, n: int = 3) -> List[str]:
        """Returns up to n paraphrases of text (may return fewer on fallback)."""
        h = self._hash(f"{text}::{n}")
        cached = self._cache[self._cache["source_hash"] == h]
        if len(cached) >= n:
            return cached["paraphrase"].tolist()[:n]

        if self._ensure_model():
            try:
                variants = self._model_paraphrase([text], n)[0]
            except Exception as exc:
                logger.warning(f"Paraphrase generation failed for one sentence: {exc}")
                variants = [self._synonym_paraphrase(text) for _ in range(n)]
        else:
            variants = [self._synonym_paraphrase(text) for _ in range(n)]

        rows = pd.DataFrame({
            "source_hash": [h] * len(variants),
            "source_text": [text] * len(variants),
            "paraphrase":  variants,
        })
        self._cache = pd.concat([self._cache, rows], ignore_index=True)
        return variants

    def expand_dataframe(self, df: pd.DataFrame,
                          variants_per_row: int = None) -> pd.DataFrame:
        """
        For every row, generate N paraphrases of `content` and append them
        as new rows (label/attack_type/agents preserved, fresh message_id).
        Returns the original rows plus their paraphrased expansions.
        """
        import uuid

        n = variants_per_row or CONFIG.paraphrase_variants_per_sentence
        logger.info(f"Paraphrasing {len(df):,} rows x {n} variants each ...")

        new_rows = []
        for i, row in df.iterrows():
            variants = self.paraphrase_one(row["content"], n=n)
            for v in variants:
                v = v.strip()
                if not v or v.lower() == str(row["content"]).lower():
                    continue
                r = row.to_dict()
                r["content"]    = v
                r["message_id"] = str(uuid.uuid4())[:8]
                new_rows.append(r)

            if (i + 1) % 500 == 0:
                logger.info(f"  ... paraphrased {i + 1:,}/{len(df):,} rows")
                self._save_cache()

        self._save_cache()
        expanded = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        logger.info(f"Paraphrase expansion: {len(df):,} -> {len(expanded):,} rows")
        return expanded
