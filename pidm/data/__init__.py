"""Dataset assembly: synthetic templates + paraphrase augmentation +
scripted multi-agent traces + real-world data, combined into the final
train/test pools.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from pidm.config import CONFIG, logger
from pidm.data.real_data_loader import RealDataLoader
from pidm.data.synthetic_generator import AttackDatasetGenerator
from pidm.data.trace_simulator import ScriptedTraceSimulator

__all__ = ["build_dataset", "AttackDatasetGenerator", "RealDataLoader", "ScriptedTraceSimulator"]

_COLS = ["message_id", "content", "from_agent", "to_agent", "label", "attack_type"]


def build_dataset(target_size: int = None,
                   use_paraphrase: bool = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds the full PIDM dataset and returns (train_val_df, test_df).

    Composition of the synthetic pool (target_size rows, default from
    CONFIG.dataset_size):
      ~65% paraphrase-expanded template sentences (AttackDatasetGenerator
           + Paraphraser) — breaks trigger-word overfitting.
      ~35% scripted multi-turn conversation traces (ScriptedTraceSimulator)
           — adds conversation history / cascade-depth structure.

    Real-world data (RealDataLoader) is held out entirely as the test set
    when CONFIG.real_only_test is True (default) — the strongest available
    generalization check: train on synthetic, test on real, never-seen
    attacks. Falls back to a blended split otherwise.
    """
    target_size = target_size or CONFIG.dataset_size
    use_paraphrase = CONFIG.use_paraphrase if use_paraphrase is None else use_paraphrase

    # Fast path: If pre-generated dataset already exists, load directly to skip hours of paraphrasing
    if CONFIG.dataset_path and os.path.isfile(CONFIG.dataset_path) and os.path.getsize(CONFIG.dataset_path) > 1000:
        logger.info(f"[build_dataset] Found pre-generated dataset at {CONFIG.dataset_path} — loading directly!")
        merged = pd.read_csv(CONFIG.dataset_path)
        from sklearn.model_selection import train_test_split
        train_val, test = train_test_split(
            merged, test_size=CONFIG.test_ratio,
            stratify=merged["label"], random_state=42,
        )
        train_val = train_val.reset_index(drop=True)
        test      = test.reset_index(drop=True)
        RealDataLoader.dataset_statistics(merged)
        logger.info(f"[build_dataset] pre-generated split | train_val={len(train_val):,} | test={len(test):,}")
        return train_val, test

    template_n = int(target_size * 0.65)
    base_n     = max(1, template_n // (1 + (CONFIG.paraphrase_variants_per_sentence if use_paraphrase else 0)))

    logger.info(f"[build_dataset] target={target_size:,} | "
                f"template base={base_n:,} (paraphrase={use_paraphrase}) | "
                f"remainder from trace simulator")

    gen           = AttackDatasetGenerator()
    df_templates  = gen.generate(n=base_n, paraphrase=use_paraphrase)

    remaining = max(0, target_size - len(df_templates))
    n_convos  = max(1, remaining // 10)   # ~10 messages/conversation on average
    sim       = ScriptedTraceSimulator()
    df_traces = sim.generate(n_convos)

    trace_cols_present = [c for c in _COLS if c in df_traces.columns]
    synthetic_pool = pd.concat(
        [df_templates[_COLS], df_traces[trace_cols_present].reindex(columns=_COLS, fill_value="")],
        ignore_index=True,
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"[build_dataset] synthetic pool assembled: {len(synthetic_pool):,} rows "
                f"({len(df_templates):,} template/paraphrase + {len(df_traces):,} trace)")

    loader  = RealDataLoader()
    df_real = loader.load_all()

    if CONFIG.real_only_test:
        train_val, test = RealDataLoader.split_real_only_test(synthetic_pool, df_real)
    else:
        from sklearn.model_selection import train_test_split
        merged = RealDataLoader.merge(synthetic_pool, df_real, real_ratio=CONFIG.real_ratio)
        train_val, test = train_test_split(
            merged, test_size=CONFIG.test_ratio,
            stratify=merged["label"], random_state=42,
        )
        train_val = train_val.reset_index(drop=True)
        test      = test.reset_index(drop=True)

    RealDataLoader.dataset_statistics(pd.concat([train_val, test], ignore_index=True))
    logger.info(f"[build_dataset] final split | train_val={len(train_val):,} | test={len(test):,}")
    return train_val, test
