"""Dataset-size scale-up study.

Trains the classifier at several dataset sizes drawn from the same pool
and evaluates each one against the *same* held-out test set, so the only
thing that varies between runs is how much training data PIDM saw. Turns
"we used a bigger dataset" into an actual reported result (a learning
curve) instead of just a bigger number.

Each size is checkpointed to its own directory and the running results
table is written to disk after every size, so a crash partway through a
multi-hour run doesn't lose the sizes already completed.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from pidm.config import CONFIG, logger
from pidm.detection.classifier import InjectionClassifier
from pidm.detection.graph_cascade_detector import GraphAwareCascadeDetector
from pidm.detection.orchestrator import PIDMOrchestrator
from pidm.detection.quarantine import QuarantineEngine
from pidm.detection.semantic_intent_drift import SemanticIntentDrift
from pidm.eval.evaluator import PIDMEvaluator

_DEFAULT_INTENT = (
    "This pipeline coordinates multiple AI agents to perform legitimate "
    "data analysis, report generation, and workflow automation tasks."
)

_RESULTS_CSV = os.path.join(CONFIG.scale_study_dir, "scale_study_results.csv")
_PLOT_PATH   = os.path.join(CONFIG.scale_study_dir, "scale_study_learning_curve.png")


class ScaleStudy:
    def __init__(self, train_val_pool: pd.DataFrame, test_df: pd.DataFrame):
        self.pool    = train_val_pool.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

    def run(self, sizes: List[int] = None) -> pd.DataFrame:
        sizes = sizes or CONFIG.scale_study_sizes
        results = []

        # Resume support: skip sizes already recorded from a prior partial run.
        done_sizes = set()
        if os.path.exists(_RESULTS_CSV):
            prior = pd.read_csv(_RESULTS_CSV)
            results = prior.to_dict("records")
            done_sizes = set(prior["train_size"].tolist())
            logger.info(f"ScaleStudy: resuming — {len(done_sizes)} sizes already completed.")

        for size in sizes:
            if size in done_sizes:
                logger.info(f"ScaleStudy: size={size:,} already done, skipping.")
                continue
            if size > len(self.pool):
                logger.warning(f"ScaleStudy: requested size={size:,} exceeds pool "
                                f"({len(self.pool):,}) — skipping.")
                continue

            logger.info(f"ScaleStudy: training at size={size:,} ...")
            subset = self.pool.sample(n=size, random_state=42).reset_index(drop=True)
            save_path = os.path.join(CONFIG.scale_study_dir, f"model_{size}")

            classifier = InjectionClassifier()
            classifier.train(subset, save_path=save_path)

            sid = SemanticIntentDrift()
            sid.set_pipeline_intent(_DEFAULT_INTENT)
            pidm = PIDMOrchestrator(classifier, sid, GraphAwareCascadeDetector(), QuarantineEngine())

            evaluator    = PIDMEvaluator(pidm, self.test_df)
            eval_results = evaluator.run_full()
            row = {"train_size": size, **eval_results["Full PIDM"],
                   "latency_p95_ms": eval_results["latency"]["p95_ms"]}
            results.append(row)

            pd.DataFrame(results).sort_values("train_size").to_csv(_RESULTS_CSV, index=False)
            logger.info(f"ScaleStudy: size={size:,} -> F1={row['F1']:.4f} FPR={row['FPR']:.4f} "
                        f"(checkpointed to {_RESULTS_CSV})")

        df = pd.DataFrame(results).sort_values("train_size").reset_index(drop=True)
        self._plot(df)
        return df

    @staticmethod
    def _plot(df: pd.DataFrame, save: bool = True) -> None:
        if df.empty:
            return
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(df["train_size"], df["F1"], "o-", color="#4C72B0", label="F1")
        ax1.plot(df["train_size"], df["Precision"], "o--", color="#55A868", label="Precision", alpha=0.7)
        ax1.plot(df["train_size"], df["Recall"], "o--", color="#8172B2", label="Recall", alpha=0.7)
        ax1.set_xlabel("Training set size (rows)")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(df["train_size"], df["FPR"], "s--", color="#C44E52", label="FPR")
        ax2.set_ylabel("False Positive Rate", color="#C44E52")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

        ax1.set_title("PIDM Scale-Up Study — Performance vs. Training Set Size\n"
                       "(evaluated on the same held-out real-world test set every run)")
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(_PLOT_PATH, dpi=150)
            logger.info(f"Scale-study learning curve saved -> {_PLOT_PATH}")
        plt.close(fig)
