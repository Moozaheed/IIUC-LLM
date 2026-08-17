"""Evaluation suite: per-layer metrics, ablation, confusion matrix, ROC,
per-attack-type breakdown, latency profiling, dataset-composition charts."""
from __future__ import annotations

import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from pidm.config import CONFIG, logger
from pidm.detection.orchestrator import PIDMOrchestrator
from pidm.detection.rule_based_filter import RuleBasedFilter


def plot_dataset_statistics(df: pd.DataFrame, save: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    label_counts = df["label"].value_counts()
    axes[0].pie(label_counts.values, labels=["Benign", "Injected"],
                colors=["#55A868", "#C44E52"], autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Label Balance")

    type_counts = df["attack_type"].value_counts()
    axes[1].barh(type_counts.index, type_counts.values, color="#4C72B0", alpha=0.85)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Attack Type Distribution")

    plt.suptitle("PIDM Dataset Statistics", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(CONFIG.output_dir, "dataset_statistics.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        logger.info(f"Dataset statistics chart saved -> {path}")
    plt.close(fig)


class PIDMEvaluator:
    def __init__(self, pidm: PIDMOrchestrator, df_test: pd.DataFrame):
        self.pidm    = pidm
        self.df_test = df_test.reset_index(drop=True)

    @staticmethod
    def _metrics(y_true, y_pred, y_prob=None) -> Dict:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        cm  = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(set(y_true)) > 1 else 0.0
        return {
            "Precision": round(p, 4), "Recall": round(r, 4),
            "F1":        round(f1, 4), "Accuracy": round(acc, 4),
            "FPR":       round(fpr, 4), "AUC-ROC": round(auc, 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        }

    def run_full(self) -> Dict:
        logger.info("Running full evaluation ...")
        rbf_obj = RuleBasedFilter()

        rbf_labels, rbf_scores   = [], []
        sid_labels, sid_scores   = [], []
        cls_labels, cls_scores   = [], []
        pidm_labels, pidm_scores = [], []
        latencies                = []
        true_labels              = self.df_test["label"].tolist()
        attack_types             = self.df_test["attack_type"].tolist()

        for _, row in self.df_test.iterrows():
            text = row["content"]

            rl, rs, _ = rbf_obj.predict(text)
            rbf_labels.append(rl); rbf_scores.append(rs)

            sl, ss = self.pidm.sid.predict(text)
            sid_labels.append(sl); sid_scores.append(ss)

            cl, cs = self.pidm.classifier.predict(text)
            cls_labels.append(cl); cls_scores.append(cs)

            t0  = time.perf_counter()
            res = self.pidm.detect_text(text, row["from_agent"], row["to_agent"])
            latencies.append((time.perf_counter() - t0) * 1000)
            pidm_labels.append(int(res.is_injected))
            pidm_scores.append(res.confidence)

        results = {
            "RBF Only":        self._metrics(true_labels, rbf_labels,  rbf_scores),
            "SID Only":        self._metrics(true_labels, sid_labels,  sid_scores),
            "Classifier Only": self._metrics(true_labels, cls_labels,  cls_scores),
            "Full PIDM":       self._metrics(true_labels, pidm_labels, pidm_scores),
            "latency": {
                "mean_ms":   round(float(np.mean(latencies)),   2),
                "median_ms": round(float(np.median(latencies)), 2),
                "p95_ms":    round(float(np.percentile(latencies, 95)), 2),
            },
        }

        type_results = {}
        df_tmp = self.df_test.copy()
        df_tmp["pred"]  = pidm_labels
        df_tmp["score"] = pidm_scores
        for atype in df_tmp["attack_type"].unique():
            sub = df_tmp[df_tmp["attack_type"] == atype]
            if len(sub) < 2:
                continue
            type_results[atype] = self._metrics(sub["label"].tolist(), sub["pred"].tolist(), sub["score"].tolist())
        results["per_attack_type"] = type_results

        self._true   = true_labels
        self._pidm_p = pidm_labels
        self._pidm_s = pidm_scores
        self._lat    = latencies
        self._rbf_p  = rbf_labels; self._rbf_s = rbf_scores
        self._sid_p  = sid_labels; self._sid_s = sid_scores
        self._cls_p  = cls_labels; self._cls_s = cls_scores
        self._types  = attack_types

        return results

    def plot_confusion_matrix(self, save: bool = True):
        cm = confusion_matrix(self._true, self._pidm_p)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Benign", "Injected"], yticklabels=["Benign", "Injected"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("PIDM Confusion Matrix (Full System)")
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "confusion_matrix.png")
            plt.savefig(path, dpi=150); logger.info(f"Saved -> {path}")
        plt.close(fig)

    def plot_roc(self, save: bool = True):
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, scores, labels in [
            ("RBF",        self._rbf_s,  self._true),
            ("SID",        self._sid_s,  self._true),
            ("Classifier", self._cls_s,  self._true),
            ("Full PIDM",  self._pidm_s, self._true),
        ]:
            if len(set(labels)) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            auc = roc_auc_score(labels, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — PIDM Layer Comparison")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "roc_curve.png")
            plt.savefig(path, dpi=150); logger.info(f"Saved -> {path}")
        plt.close(fig)

    def plot_ablation(self, results: Dict, save: bool = True):
        layers  = ["RBF Only", "SID Only", "Classifier Only", "Full PIDM"]
        metrics = ["Precision", "Recall", "F1", "FPR"]
        x = np.arange(len(layers)); width = 0.20
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = [results[l][metric] for l in layers]
            ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)

        ax.set_xticks(x + width * 1.5); ax.set_xticklabels(layers, fontsize=11)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
        ax.set_title("Ablation Study — PIDM Layer Comparison")
        ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "ablation_study.png")
            plt.savefig(path, dpi=150); logger.info(f"Saved -> {path}")
        plt.close(fig)

    def plot_per_attack_type(self, results: Dict, save: bool = True):
        type_res = results.get("per_attack_type", {})
        if not type_res:
            return
        types = list(type_res.keys())
        f1s   = [type_res[t]["F1"]  for t in types]
        fprs  = [type_res[t]["FPR"] for t in types]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(types))
        ax.bar(x - 0.2, f1s,  0.35, label="F1",  color="#4C72B0", alpha=0.85)
        ax.bar(x + 0.2, fprs, 0.35, label="FPR", color="#C44E52", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title("Full PIDM — Per-Attack-Type F1 and FPR"); ax.set_ylabel("Score")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "per_attack_type.png")
            plt.savefig(path, dpi=150); logger.info(f"Saved -> {path}")
        plt.close(fig)

    def plot_latency(self, save: bool = True):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(self._lat, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(self._lat),   color="red",    linestyle="--", label=f"Mean={np.mean(self._lat):.1f}ms")
        ax.axvline(np.median(self._lat), color="orange", linestyle="--", label=f"Median={np.median(self._lat):.1f}ms")
        ax.set_xlabel("Detection Latency (ms)"); ax.set_ylabel("Count")
        ax.set_title("PIDM Detection Latency Distribution")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            path = os.path.join(CONFIG.output_dir, "latency.png")
            plt.savefig(path, dpi=150); logger.info(f"Saved -> {path}")
        plt.close(fig)

    def print_report(self, results: Dict):
        print("\n" + "=" * 70)
        print("  PIDM EVALUATION REPORT")
        print("=" * 70)
        for layer in ["RBF Only", "SID Only", "Classifier Only", "Full PIDM"]:
            m = results[layer]
            print(f"\n  [{layer}]")
            print(f"    Precision : {m['Precision']:.4f}   Recall : {m['Recall']:.4f}")
            print(f"    F1-Score  : {m['F1']:.4f}   FPR    : {m['FPR']:.4f}")
            print(f"    Accuracy  : {m['Accuracy']:.4f}   AUC-ROC: {m['AUC-ROC']:.4f}")
            print(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
        lat = results["latency"]
        print(f"\n  [Latency]  mean={lat['mean_ms']}ms | median={lat['median_ms']}ms | p95={lat['p95_ms']}ms")
        print("\n  [Per-Attack-Type F1]")
        for atype, m in results.get("per_attack_type", {}).items():
            print(f"    {atype:<32} F1={m['F1']:.4f}  FPR={m['FPR']:.4f}")
        print("=" * 70 + "\n")
