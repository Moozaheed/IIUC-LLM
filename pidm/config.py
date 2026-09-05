"""Global PIDM configuration."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("PIDM")


def _gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


@dataclass
class PIDMConfig:
    # Model selection — resolved in __post_init__ based on available hardware.
    classifier_model: str = "microsoft/deberta-v3-base"
    sentence_model:   str = "sentence-transformers/all-MiniLM-L6-v2"
    paraphrase_model: str = "humarin/chatgpt_paraphraser_on_T5_base"

    # Training hyper-parameters (tuned per-device in __post_init__)
    num_epochs:                  int   = 3
    batch_size:                  int   = 16
    gradient_accumulation_steps: int   = 1
    learning_rate:                float = 3e-5
    max_length:                    int = 256
    warmup_steps:                  int = 0
    weight_decay:                float = 0.01

    # Detection thresholds (tune via ablation)
    rbf_threshold:        float = 0.40
    classifier_threshold: float = 0.50
    sid_drift_threshold:  float = 0.42
    gcpd_cascade_thresh:  float = 0.55

    # Dataset composition
    dataset_size:   int   = 40_000     # target final pool size
    train_ratio:    float = 0.70
    val_ratio:      float = 0.15
    test_ratio:     float = 0.15
    real_ratio:     float = 0.25       # used only when real_only_test is False
    real_only_test: bool  = False      # blend real data into training; True = zero-shot domain-shift ablation
    use_paraphrase: bool  = True       # paraphrase-augment generated sentences
    real_source_max_rows: int = 4_000  # cap per-source rows so one huge HF dataset can't dominate
    real_test_max_size:   int = 6_000  # cap the final real-only test set for tractable eval
    scale_study_sizes: List[int] = field(
        default_factory=lambda: [2_000, 5_000, 10_000, 20_000, 40_000]
    )

    # Paraphrase augmentation
    paraphrase_variants_per_payload:  int = 10   # payload phrase -> N paraphrases
    paraphrase_variants_per_sentence: int = 3    # full generated sentence -> N paraphrases
    paraphrase_batch_size:            int = 32

    # I/O paths
    output_dir:             str = "./pidm_output"
    dataset_path:           str = "./pidm_dataset.csv"
    paraphrase_cache_path:  str = "./pidm/data/paraphrase_cache.csv"
    model_save_path:        str = "./pidm_classifier"
    scale_study_dir:        str = "./pidm_output/scale_study"

    # Runtime
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.scale_study_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.paraphrase_cache_path), exist_ok=True)

        if self.device == "cpu":
            # No GPU at all — fall back to the lightest feasible model.
            self.classifier_model = "distilbert-base-uncased"
            self.num_epochs = 3
            self.batch_size = 8
            logger.warning("CPU detected — using DistilBERT, 3 epochs, batch_size=8.")
        else:
            vram = _gpu_vram_gb()
            if vram < 6.0:
                # Small/older GPU — deberta-v3-small is the safe default.
                self.classifier_model = "microsoft/deberta-v3-small"
                self.batch_size = 8
                self.gradient_accumulation_steps = 2
                logger.warning(
                    f"GPU VRAM={vram:.1f}GB (<6GB) — using DeBERTa-v3-small, "
                    f"batch_size=8, grad_accum=2."
                )
            elif vram < 10.0:
                # 8GB-class card (e.g. RTX 3070) — deberta-v3-base fits with fp16.
                self.classifier_model = "microsoft/deberta-v3-base"
                self.batch_size = 16
                self.gradient_accumulation_steps = 2
                logger.info(
                    f"GPU VRAM={vram:.1f}GB — using DeBERTa-v3-base, "
                    f"batch_size=16 (effective 32 via grad_accum=2), fp16."
                )
            else:
                self.classifier_model = "microsoft/deberta-v3-base"
                self.batch_size = 16
                self.gradient_accumulation_steps = 2
                logger.info(
                    f"GPU VRAM={vram:.1f}GB — using DeBERTa-v3-base, "
                    f"batch_size=16, grad_accum=2 (effective 32), fp32."
                )

        # Log every resolved value explicitly so the methods section of the
        # paper can be filled in directly from the run log — not inferred from code.
        logger.info(
            f"PIDMConfig RESOLVED | device={self.device} | model={self.classifier_model} | "
            f"batch_size={self.batch_size} | grad_accum={self.gradient_accumulation_steps} | "
            f"effective_batch={self.batch_size * self.gradient_accumulation_steps} | "
            f"epochs={self.num_epochs} | lr={self.learning_rate} | "
            f"fp16={'True' if self.device == 'cuda' else 'False'} | "
            f"max_seq_len={self.max_length} | dataset_size={self.dataset_size}"
        )


CONFIG = PIDMConfig()
