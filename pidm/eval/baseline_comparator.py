"""Compares Full PIDM against 6 competing systems on the same held-out
test set. This is the key result table for the thesis.

B0 — No Protection             : always predicts benign (worst case)
B1 — Keyword Filter            : 24 hand-crafted attack keywords
B2 — TF-IDF + Logistic Reg.    : classical ML, no deep learning
B3 — RBF Only                  : our rule-based layer in isolation
B4 — Classifier Only (DeBERTa) : our transformer without SID/GCPD
B5 — Full PIDM (Ours)          : all four layers combined
B6 — protectai/deberta-v3-base-prompt-injection-v2 (zero-shot, pretrained)
     : a real published prompt-injection classifier, run with no
       fine-tuning on our data — the strongest "existing tool" comparison.
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from pidm.config import CONFIG, logger
from pidm.detection.orchestrator import PIDMOrchestrator
from pidm.detection.rule_based_filter import RuleBasedFilter

_PROTECTAI_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


class BaselineComparator:
    _KEYWORDS = [
        "ignore", "override", "bypass", "jailbreak", "exfiltrate",
        "disregard", "forget previous", "new instructions",
        "reveal system", "disable filter", "disable safety",
        "forget everything", "system prompt", "act as", "pretend you",
        "decode and execute", "base64", "admin command",
        "root command", "emergency override", "new system prompt",
        "all previous instructions", "role play", "dan mode",
    ]

    def __init__(self, train_df: pd.DataFrame):
        self._train_df = train_df
        self._rbf      = RuleBasedFilter()
        self._tfidf    = None
        self._fitted   = False
        self._protectai = None
        self._protectai_failed = False

    def _fit_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        logger.info("BaselineComparator: fitting TF-IDF + LR ...")
        self._tfidf = Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 3), max_features=30_000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)),
        ])
        self._tfidf.fit(self._train_df["content"].tolist(), self._train_df["label"].tolist())
        self._fitted = True

    # Some real-world/attack samples (context-flooding padding especially)
    # run to thousands of characters; a batched forward pass without a hard
    # cap can blow attention memory well past what an 8GB card has free
    # alongside the already-resident fine-tuned classifier + SID model.
    _PROTECTAI_BATCH_SIZE = 8
    _PROTECTAI_CHAR_CAP   = 2000

    def _load_protectai(self) -> bool:
        if self._protectai is not None or self._protectai_failed:
            return self._protectai is not None
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"BaselineComparator: loading pretrained {_PROTECTAI_MODEL} ...")
            if CONFIG.device == "cuda":
                import torch
                torch.cuda.empty_cache()
            self._protectai = hf_pipeline(
                "text-classification", model=_PROTECTAI_MODEL,
                device=0 if CONFIG.device == "cuda" else -1,
                truncation=True, max_length=256,
            )
            return True
        except Exception as exc:
            logger.warning(f"BaselineComparator: could not load {_PROTECTAI_MODEL} ({exc}) — "
                            f"skipping B6 baseline.")
            self._protectai_failed = True
            return False

    def _protectai_predict(self, texts) -> tuple:
        # Belt-and-suspenders char-level cap, independent of whatever the
        # pipeline's tokenizer truncation actually honors on this version.
        capped = [t[:self._PROTECTAI_CHAR_CAP] for t in texts]
        preds, probs = [], []
        batch_size = self._PROTECTAI_BATCH_SIZE
        i = 0
        while i < len(capped):
            chunk = capped[i:i + batch_size]
            try:
                outputs = list(self._protectai(chunk, batch_size=batch_size))
            except Exception as exc:  # CUDA OOM or similar — retry one-by-one, smaller
                logger.warning(f"BaselineComparator: B6 batch failed ({exc}) — "
                                f"retrying that chunk one item at a time.")
                if CONFIG.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()
                outputs = [self._protectai(t)[0] for t in chunk]
            for out in outputs:
                label_str = str(out["label"]).upper()
                is_inject = ("INJECTION" in label_str) or label_str in ("1", "LABEL_1")
                preds.append(1 if is_inject else 0)
                probs.append(out["score"] if is_inject else 1 - out["score"])
            i += batch_size
        return preds, probs

    @staticmethod
    def _metrics(y_true: list, y_pred: list, y_prob: list = None) -> Dict:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        try:
            auc = roc_auc_score(y_true, y_prob) if y_prob else 0.0
        except Exception:
            auc = 0.0
        return {
            "Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4),
            "Accuracy": round(acc, 4), "FPR": round(fpr, 4), "AUC-ROC": round(auc, 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        }

    def run(self, df_test: pd.DataFrame, pidm: "PIDMOrchestrator") -> pd.DataFrame:
        if not self._fitted:
            self._fit_tfidf()

        y_true      = df_test["label"].tolist()
        texts       = df_test["content"].tolist()
        agents_from = df_test["from_agent"].tolist()
        agents_to   = df_test["to_agent"].tolist()
        n = len(texts)

        logger.info(f"BaselineComparator: evaluating {n} test samples ...")
        results: Dict[str, Dict] = {}

        results["B0: No Protection"] = self._metrics(y_true, [0] * n)

        kw_preds = [1 if any(kw in t.lower() for kw in self._KEYWORDS) else 0 for t in texts]
        results["B1: Keyword Filter"] = self._metrics(y_true, kw_preds, [float(p) for p in kw_preds])

        tfidf_preds = self._tfidf.predict(texts).tolist()
        tfidf_probs = self._tfidf.predict_proba(texts)[:, 1].tolist()
        results["B2: TF-IDF + LR"] = self._metrics(y_true, tfidf_preds, tfidf_probs)

        rbf_preds, rbf_probs = [], []
        for t in texts:
            lbl, sc, _ = self._rbf.predict(t)
            rbf_preds.append(lbl); rbf_probs.append(sc)
        results["B3: RBF Only"] = self._metrics(y_true, rbf_preds, rbf_probs)

        logger.info("  Running Classifier-only baseline (batched) ...")
        pairs = pidm.classifier.predict_batch(texts, batch_size=64)
        cls_preds = [p[0] for p in pairs]
        cls_probs = [p[1] for p in pairs]
        results["B4: Classifier Only"] = self._metrics(y_true, cls_preds, cls_probs)

        logger.info("  Running Full PIDM ...")
        pidm_preds, pidm_probs = [], []
        for t, fa, ta in zip(texts, agents_from, agents_to):
            res = pidm.detect_text(t, fa, ta)
            pidm_preds.append(int(res.is_injected)); pidm_probs.append(res.confidence)
        results["B5: Full PIDM (Ours)"] = self._metrics(y_true, pidm_preds, pidm_probs)

        if self._load_protectai():
            logger.info(f"  Running B6: pretrained {_PROTECTAI_MODEL} (zero-shot) ...")
            pa_preds, pa_probs = self._protectai_predict(texts)
            results["B6: ProtectAI (pretrained)"] = self._metrics(y_true, pa_preds, pa_probs)

        df_result = pd.DataFrame(results).T
        df_result.index.name = "System"

        csv_path = os.path.join(CONFIG.output_dir, "baseline_comparison.csv")
        df_result.to_csv(csv_path)
        logger.info(f"Baseline CSV saved -> {csv_path}")

        self._print_table(df_result)
        self._plot_bar(df_result)
        self._plot_radar(df_result)

        return df_result

    @staticmethod
    def _print_table(df: pd.DataFrame) -> None:
        border = "=" * 90
        print(f"\n{border}")
        print("  BASELINE COMPARISON TABLE")
        total = int(df["TP"].iloc[0] + df["FP"].iloc[0] + df["FN"].iloc[0] + df["TN"].iloc[0])
        print(f"  Test set size: {total} samples")
        print(border)
        print(f"  {'System':<32} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'AUC':>7}  "
              f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print("  " + "-" * 86)
        for sys_name, row in df.iterrows():
            flag = "  <<<" if "PIDM" in str(sys_name) else ""
            print(f"  {sys_name:<32} {row['Precision']:>7.4f} {row['Recall']:>7.4f} "
                  f"{row['F1']:>7.4f} {row['FPR']:>7.4f} {row['AUC-ROC']:>7.4f}  "
                  f"{int(row['TP']):>5} {int(row['FP']):>5} {int(row['FN']):>5} {int(row['TN']):>5}{flag}")
        print(border + "\n")

    @staticmethod
    def _plot_bar(df: pd.DataFrame, save: bool = True) -> None:
        systems = [s.split(": ", 1)[-1] for s in df.index]
        metrics = ["Precision", "Recall", "F1", "FPR", "AUC-ROC"]
        x = np.arange(len(systems)); width = 0.14
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

        fig, ax = plt.subplots(figsize=(16, 7))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = df[metric].tolist()
            bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
            for j, bar in enumerate(bars):
                if "PIDM" in df.index[j]:
                    bar.set_edgecolor("gold"); bar.set_linewidth(2.5)

        ax.set_xticks(x + width * 2); ax.set_xticklabels(systems, fontsize=9, rotation=10, ha="right")
        ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=12)
        ax.set_title("PIDM vs Baselines — Precision, Recall, F1, FPR, AUC-ROC\n"
                     "(Gold border = our system, lower FPR is better)", fontsize=12)
        ax.legend(loc="upper left", fontsize=10); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "baseline_comparison.png")
            plt.savefig(path, dpi=150); logger.info(f"Baseline bar chart saved -> {path}")
        plt.close(fig)

    @staticmethod
    def _plot_radar(df: pd.DataFrame, save: bool = True) -> None:
        metrics   = ["Precision", "Recall", "F1", "AUC-ROC"]
        n_metrics = len(metrics)
        angles    = [n / n_metrics * 2 * np.pi for n in range(n_metrics)]
        angles   += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        colors = ["#aaaaaa", "#DD8452", "#4C72B0", "#55A868", "#8172B2", "#C44E52", "#2CA02C"]

        for i, (sys_name, row) in enumerate(df.iterrows()):
            vals  = [row[m] for m in metrics] + [row[metrics[0]]]
            lw    = 2.5 if "PIDM" in str(sys_name) else 1.2
            label = sys_name.split(": ", 1)[-1]
            ax.plot(angles, vals, "o-", linewidth=lw, color=colors[i % len(colors)], label=label)
            ax.fill(angles, vals, alpha=0.04, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Baseline Radar — PIDM vs All Systems", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "baseline_radar.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Baseline radar chart saved -> {path}")
        plt.close(fig)

    def improvement_summary(self, df_result: pd.DataFrame) -> None:
        if "B5: Full PIDM (Ours)" not in df_result.index:
            return
        pidm_row = df_result.loc["B5: Full PIDM (Ours)"]
        print("\n-- PIDM Improvement over Baselines -------------------------")
        print(f"  {'vs. System':<30}  {'dF1':>8}  {'dAUC':>8}  {'dFPR':>9}")
        print("  " + "-" * 60)
        for sys_name, row in df_result.iterrows():
            if "PIDM" in str(sys_name):
                continue
            d_f1  = pidm_row["F1"]      - row["F1"]
            d_auc = pidm_row["AUC-ROC"] - row["AUC-ROC"]
            d_fpr = row["FPR"]          - pidm_row["FPR"]   # reduction is good
            label = str(sys_name).split(": ", 1)[-1]
            print(f"  {label:<30}  {d_f1:>+8.4f}  {d_auc:>+8.4f}  {d_fpr:>+9.4f}")
        print("  (Positive values = PIDM is better)\n")
