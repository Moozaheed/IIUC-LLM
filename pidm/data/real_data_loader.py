"""Downloads and normalises publicly available prompt injection datasets
to mix with (or hold out against) AADG synthetic data.

This makes evaluation honest — the model is tested on real-world attacks
it has never seen in template form.

Sources (no API key required), each independently optional — a dead or
renamed HF dataset just gets skipped with a warning, never a hard failure:
  - deepset/prompt-injections
  - markusbayer/prompt-injection
  - JasperLS/prompt-injections
  - xTRam1/safe-guard-prompt-injection
  - jayavibhav/prompt-injection
  - GitHub AdvBench harmful_behaviors.csv
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from pidm.config import CONFIG, logger

_HF_SOURCES = [
    {"name": "deepset/prompt-injections",         "text_col": "text", "label_col": "label", "attack_type": "real_injection"},
    {"name": "markusbayer/prompt-injection",       "text_col": "text", "label_col": "label", "attack_type": "real_injection"},
    {"name": "JasperLS/prompt-injections",         "text_col": "text", "label_col": "label", "attack_type": "real_injection"},
    {"name": "xTRam1/safe-guard-prompt-injection", "text_col": "text", "label_col": "label", "attack_type": "real_injection"},
    {"name": "jayavibhav/prompt-injection",        "text_col": "text", "label_col": "label", "attack_type": "real_injection"},
]

_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)

_AGENT_PAIRS = [
    ("UserProxy",         "OrchestratorAgent"),
    ("OrchestratorAgent", "WorkerAgent"),
    ("WorkerAgent",       "ReporterAgent"),
    ("ExternalSource",    "OrchestratorAgent"),
    ("AnalystAgent",      "WorkerAgent"),
]

_COLS = ["message_id", "content", "from_agent", "to_agent", "label", "attack_type"]


class RealDataLoader:
    def _rand_pair(self) -> Tuple[str, str]:
        import random
        return random.choice(_AGENT_PAIRS)

    def _normalise(self, df: pd.DataFrame, text_col: str,
                    label_col: str, attack_type: str, source: str) -> pd.DataFrame:
        import uuid
        rows = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, "")).strip()
            if not text:
                continue
            label  = int(row[label_col]) if label_col in row else 1
            fa, ta = self._rand_pair()
            rows.append({
                "message_id":  str(uuid.uuid4())[:8],
                "content":     text,
                "from_agent":  fa,
                "to_agent":    ta,
                "label":       label,
                "attack_type": attack_type if label == 1 else "benign",
                "source":      source,
            })
        return pd.DataFrame(rows)

    def _load_hf(self) -> pd.DataFrame:
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            logger.warning("RealDataLoader: `datasets` not installed — skipping HF sources.")
            return pd.DataFrame()

        frames = []
        for src in _HF_SOURCES:
            try:
                logger.info(f"  Fetching HF dataset: {src['name']} ...")
                ds = hf_load(src["name"], split="train")
                df = ds.to_pandas()
                if len(df) > CONFIG.real_source_max_rows:
                    df = df.sample(CONFIG.real_source_max_rows, random_state=42).reset_index(drop=True)
                    logger.info(f"  {src['name']}: capped to {CONFIG.real_source_max_rows:,} rows "
                                f"(source had more — keeps one dataset from dominating the real-data pool)")
                if src["text_col"] not in df.columns:
                    logger.warning(f"  Column '{src['text_col']}' not found in {src['name']}. Skipping.")
                    continue
                if src["label_col"] in df.columns:
                    sample = df[src["label_col"]].iloc[0]
                    if str(sample).lower() in ["injection", "benign", "safe", "ham"]:
                        df[src["label_col"]] = df[src["label_col"]].apply(
                            lambda x: 0 if str(x).lower() in ["benign", "safe", "ham", "0"] else 1
                        )
                    else:
                        df[src["label_col"]] = df[src["label_col"]].astype(int)
                out = self._normalise(df, src["text_col"], src["label_col"],
                                       src["attack_type"], src["name"])
                frames.append(out)
                logger.info(f"  Loaded {len(out):,} rows from {src['name']}")
            except Exception as exc:
                logger.warning(f"  {src['name']} failed: {exc}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_advbench(self) -> pd.DataFrame:
        try:
            import urllib.request
            from io import StringIO
            logger.info("  Downloading AdvBench harmful_behaviors.csv ...")
            with urllib.request.urlopen(_ADVBENCH_URL, timeout=15) as r:
                raw = r.read().decode("utf-8")
            df  = pd.read_csv(StringIO(raw))
            col = "goal" if "goal" in df.columns else df.columns[0]
            df["label"] = 1
            out = self._normalise(df, col, "label", "real_harmful", "advbench")
            logger.info(f"  Loaded {len(out):,} rows from AdvBench")
            return out
        except Exception as exc:
            logger.warning(f"  AdvBench failed: {exc}")
            return pd.DataFrame()

    def load_all(self) -> pd.DataFrame:
        """Load every available real dataset and combine them."""
        logger.info("RealDataLoader: fetching real-world datasets ...")
        frames = []

        hf = self._load_hf()
        if not hf.empty:
            frames.append(hf)

        adv = self._load_advbench()
        if not adv.empty:
            frames.append(adv)

        if not frames:
            logger.warning("RealDataLoader: no real data available — evaluation "
                            "will use synthetic data only.")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        inj = int(combined["label"].sum())
        ben = len(combined) - inj
        logger.info(
            f"RealDataLoader: total {len(combined):,} rows loaded | "
            f"injected={inj:,} | benign={ben:,}"
        )
        return combined

    @staticmethod
    def merge(synthetic_df: pd.DataFrame, real_df: pd.DataFrame,
              real_ratio: float = 0.25) -> pd.DataFrame:
        """Blend synthetic and real data so real_ratio% of the final set is real."""
        if real_df.empty:
            logger.info("Merge: returning synthetic-only dataset.")
            return synthetic_df[_COLS].copy()

        n_syn       = len(synthetic_df)
        n_real_want = int(n_syn * real_ratio / (1.0 - real_ratio))
        n_real_take = min(n_real_want, len(real_df))
        real_sample = real_df.sample(n_real_take, random_state=42)

        for col in _COLS:
            if col not in real_sample.columns:
                real_sample[col] = "unknown"

        merged = pd.concat([synthetic_df[_COLS], real_sample[_COLS]], ignore_index=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

        real_pct = n_real_take / len(merged) * 100
        logger.info(f"Merged: {len(merged):,} rows total | synthetic={n_syn:,} | "
                    f"real={n_real_take:,} ({real_pct:.1f}%)")
        return merged

    @staticmethod
    def split_real_only_test(synthetic_df: pd.DataFrame,
                              real_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stronger generalization split: ALL synthetic (+ paraphrased + simulated
        trace) data goes to train/val; ALL real-world data becomes the test
        set, so evaluation only ever sees attacks the model has never
        encountered in template form. Falls back to a normal held-out slice
        of synthetic data if no real data is available.

        Returns (train_val_df, test_df).
        """
        train_val = synthetic_df[_COLS].copy().sample(frac=1, random_state=42).reset_index(drop=True)

        if real_df.empty:
            logger.warning("split_real_only_test: no real data — falling back to "
                            "a synthetic held-out slice as the test set.")
            n_test = int(len(train_val) * CONFIG.test_ratio)
            return train_val.iloc[:-n_test].reset_index(drop=True), \
                   train_val.iloc[-n_test:].reset_index(drop=True)

        test_df = real_df[_COLS].copy().sample(frac=1, random_state=42).reset_index(drop=True)
        if len(test_df) > CONFIG.real_test_max_size:
            # Stratified cap so the label balance survives the downsample —
            # an eval pass over the whole test set runs once per baseline
            # system, so an uncapped multi-hundred-thousand-row real pool
            # (a single oversized HF source can dwarf everything else) would
            # make every evaluation run impractically slow. Iterate groups
            # directly (not groupby().apply()) — apply() silently drops the
            # grouping column on recent pandas versions.
            frac  = CONFIG.real_test_max_size / len(test_df)
            parts = []
            for _, group in test_df.groupby("label"):
                n_take = max(1, min(len(group), round(len(group) * frac)))
                parts.append(group.sample(n_take, random_state=42))
            test_df = pd.concat(parts, ignore_index=True)
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_df = test_df.iloc[:CONFIG.real_test_max_size].reset_index(drop=True)
            logger.info(f"Real-only-test split: capped test set to {len(test_df):,} rows "
                        f"(real_test_max_size={CONFIG.real_test_max_size:,})")

        logger.info(f"Real-only-test split: train/val={len(train_val):,} (synthetic) | "
                    f"test={len(test_df):,} (100% real-world)")
        return train_val, test_df

    @staticmethod
    def dataset_statistics(df: pd.DataFrame) -> None:
        """Print a summary of the merged dataset (plotting handled by eval/evaluator.py)."""
        print("\n-- Dataset Statistics -------------------------------------")
        print(f"  Total rows    : {len(df):,}")
        print(f"  Injected (1)  : {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
        print(f"  Benign   (0)  : {(df['label']==0).sum():,}  ({(df['label']==0).mean()*100:.1f}%)")
        print("\n  Attack type distribution:")
        for atype, cnt in df["attack_type"].value_counts().items():
            bar = "#" * (cnt * 30 // max(len(df), 1))
            print(f"  {atype:<35} {cnt:>6}  {bar}")
        if "source" in df.columns:
            print("\n  Data source breakdown:")
            for src, cnt in df["source"].value_counts().items():
                print(f"  {str(src):<45} {cnt:>6}")
        print("-------------------------------------------------------------\n")
