"""Layer 2 — Transformer Classifier (DeBERTa-v3 / DistilBERT)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from pidm.config import CONFIG, logger


class _WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to reduce false negatives.

    Injection class (label=1) is upweighted so the model pays more for
    missing an attack than for a false alarm. Weight ratio 1.38:1.0 is
    derived from the 42/58 class split in the training pool (58/42 ≈ 1.38).
    **kwargs keeps the signature compatible with both transformers 4.x and
    5.x (which added num_items_in_batch to compute_loss).
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = torch.tensor([1.0, 1.38], device=outputs.logits.device,
                                dtype=outputs.logits.dtype)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                  weight=weights)
        return (loss, outputs) if return_outputs else loss


class _TokenizedDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Exclude token_type_ids — DeBERTa-v3 has type_vocab_size=0; passing
        # them triggers an out-of-range embedding lookup that silently corrupts
        # every forward pass and prevents the loss from decreasing.
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()
                if k != "token_type_ids"}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def _mixed_precision_kwargs() -> dict:
    """
    DeBERTa-v3's XSoftmax/StableDropout autograd ops produce NaN gradients
    under FP16 gradient scaling on Turing-class GPUs (T4, V100, RTX 20xx).
    BF16 avoids the scaler entirely and is safe; Ampere+ GPUs (RTX 30xx+,
    A100) support it natively. When BF16 is unavailable (Turing / CPU),
    fall back to FP32 — slower but always correct.
    """
    if CONFIG.device != "cuda":
        return {"fp16": False, "bf16": False}
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    # T4 / Turing: no BF16, FP16 kills DeBERTa gradients — use FP32.
    logger.info("BF16 not supported on this GPU; training in FP32 to avoid "
                "DeBERTa XSoftmax NaN overflow under FP16.")
    return {"fp16": False, "bf16": False}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}


class InjectionClassifier:
    """
    Fine-tunes DeBERTa-v3 (or DistilBERT on CPU) on the labeled
    inter-agent message dataset and provides inference.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG.classifier_model
        self.tokenizer   = None
        self.model       = None
        self._trained    = False

    def _tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=CONFIG.max_length,
            return_token_type_ids=False,   # DeBERTa-v3 type_vocab_size=0
        )

    def train(self, df: pd.DataFrame, save_path: str = None,
              num_epochs: int = None) -> Dict:
        """Fine-tune on the training split of df. Returns final eval metrics."""
        save_path = save_path or CONFIG.model_save_path
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        from sklearn.model_selection import train_test_split
        val_frac = CONFIG.val_ratio / (CONFIG.train_ratio + CONFIG.val_ratio)
        train_df, val_df = train_test_split(
            df, test_size=val_frac, stratify=df["label"], random_state=42
        )

        train_enc = self._tokenize(train_df["content"].tolist())
        val_enc   = self._tokenize(val_df["content"].tolist())

        train_ds = _TokenizedDataset(train_enc, train_df["label"].tolist())
        val_ds   = _TokenizedDataset(val_enc,   val_df["label"].tolist())

        # transformers renames/removes TrainingArguments fields across major
        # releases (evaluation_strategy -> eval_strategy, logging_dir dropped,
        # etc). Rather than chasing each rename, build the desired kwargs and
        # keep only the ones the installed version actually accepts.
        desired_args = {
            "output_dir":                   save_path,
            "num_train_epochs":             num_epochs or CONFIG.num_epochs,
            "per_device_train_batch_size":  CONFIG.batch_size,
            "per_device_eval_batch_size":   CONFIG.batch_size,
            "gradient_accumulation_steps":  CONFIG.gradient_accumulation_steps,
            "learning_rate":                CONFIG.learning_rate,
            "adam_epsilon":                 1e-6,   # DeBERTa-v3 requires this; default 1e-8 causes grad explosion
            "max_grad_norm":                1.0,
            "weight_decay":                 CONFIG.weight_decay,
            "warmup_ratio":                 0.10,   # 10% warmup is safer for DeBERTa-v3
            "warmup_steps":                 0,      # 0 = use warmup_ratio exclusively; avoids max() ambiguity
            "eval_strategy":                "epoch",
            "evaluation_strategy":          "epoch",   # older transformers name; filtered below
            "save_strategy":                "epoch",
            "load_best_model_at_end":       True,
            "metric_for_best_model":        "eval_f1",   # F1 not loss — loss can plateau while F1 climbs
            "greater_is_better":            True,
            "logging_dir":                  f"{save_path}/logs",
            "logging_steps":                50,
            "report_to":                    "none",
            "save_total_limit":             1,
            **_mixed_precision_kwargs(),
        }
        import inspect
        accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
        filtered_args = {k: v for k, v in desired_args.items() if k in accepted}
        dropped = set(desired_args) - set(filtered_args)
        if dropped:
            logger.info(f"TrainingArguments: this transformers version doesn't accept "
                        f"{sorted(dropped)} — skipping.")

        args = TrainingArguments(**filtered_args)

        # Seed everything before Trainer init so model weights, dropout, and
        # data shuffling inside the Trainer are all deterministic.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if CONFIG.device == "cuda":
            torch.cuda.manual_seed_all(42)

        # `tokenizer` was renamed to `processing_class` in newer transformers.
        desired_trainer_kwargs = {
            "model":           self.model,
            "args":            args,
            "train_dataset":   train_ds,
            "eval_dataset":    val_ds,
            "tokenizer":         self.tokenizer,   # older transformers name
            "processing_class":  self.tokenizer,   # newer transformers name
            "data_collator":   DataCollatorWithPadding(self.tokenizer),
            "compute_metrics": compute_metrics,
            "callbacks":       [EarlyStoppingCallback(early_stopping_patience=2)],
        }
        trainer_accepted = set(inspect.signature(_WeightedTrainer.__init__).parameters)
        trainer = _WeightedTrainer(**{k: v for k, v in desired_trainer_kwargs.items() if k in trainer_accepted})

        logger.info(f"Starting classifier training on {len(train_df):,} rows "
                    f"(batch={CONFIG.batch_size} x grad_accum={CONFIG.gradient_accumulation_steps}) ...")
        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.model.eval()
        self.model.to(CONFIG.device)
        self._trained = True
        logger.info(f"Classifier saved -> {save_path}")
        return metrics

    def load(self, path: str = None) -> None:
        path = path or CONFIG.model_save_path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Always load in FP32 for inference — DeBERTa attention overflows in FP16.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, torch_dtype=torch.float32
        )
        self.model.to(CONFIG.device)
        self.model.eval()
        self._trained = True
        logger.info(f"Classifier loaded from {path}")

    def predict(self, text: str) -> Tuple[int, float]:
        """Returns (label, probability_of_injection)."""
        if not self._trained:
            raise RuntimeError("Call .train() or .load() first.")
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=CONFIG.max_length, padding=True,
            return_token_type_ids=False,
        ).to(CONFIG.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        score = float(probs[1])
        if np.isnan(score) or np.isinf(score):
            score = 0.0
        label = 1 if score >= CONFIG.classifier_threshold else 0
        return label, score

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Tuple[int, float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=CONFIG.max_length, padding=True,
                return_token_type_ids=False,
            ).to(CONFIG.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            for s in probs:
                s = 0.0 if (np.isnan(s) or np.isinf(s)) else float(s)
                results.append((1 if s >= CONFIG.classifier_threshold else 0, s))
        return results
