#!/usr/bin/env python3
"""
Local runner for the PIDM pipeline, tuned for a single desktop GPU
(developed against an RTX 3070, 8GB VRAM). No Colab session-timeout
ceiling here, so this defaults to the full 40k-row pipeline — use the
flags below for fast iteration runs instead.

Usage:
    python local_runner.py                          # full run
    python local_runner.py --dataset-size 2000 --epochs 1 --no-demo   # smoke test
    python local_runner.py --scale-study             # + multi-hour scale-up study
    python local_runner.py --no-real-only-test        # blend real data instead of holding it out
    python local_runner.py --work-dir D:\PIDM_work     # force where heavy artifacts land

Heavy artifacts (HF model/dataset cache, trained classifier checkpoints,
the built dataset, scale-study checkpoints) can add up to several GB and
are NOT auto-cleaned — by default they're written under the project
folder. If the project's drive has less than ~15GB free, this script
auto-redirects them to whichever local drive has the most free space
instead (skip with --work-dir to force a specific location).
"""
from __future__ import annotations

import argparse
import os
import shutil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIDM local runner (RTX 3070-tuned)")
    p.add_argument("--dataset-size", type=int, default=None,
                    help="Target dataset size (default: CONFIG.dataset_size, 40000)")
    p.add_argument("--epochs", type=int, default=None,
                    help="Override CONFIG.num_epochs (useful for smoke tests)")
    p.add_argument("--no-adversarial", action="store_true",
                    help="Skip the adversarial robustness suite")
    p.add_argument("--scale-study", action="store_true",
                    help="Also run the multi-hour dataset-size scale-up study")
    p.add_argument("--no-demo", action="store_true",
                    help="Don't launch the Gradio demo at the end")
    p.add_argument("--no-real-only-test", action="store_true",
                    help="Blend real-world data into train instead of holding it all out as test")
    p.add_argument("--no-paraphrase", action="store_true",
                    help="Skip paraphrase augmentation (faster, less diverse dataset)")
    p.add_argument("--work-dir", type=str, default=None,
                    help="Where to write HF cache / model checkpoints / dataset / outputs. "
                         "Auto-picked (roomiest local drive) if omitted.")
    return p.parse_args()


def _pick_work_dir(explicit: str) -> str:
    if explicit:
        return explicit

    cwd = os.getcwd()
    try:
        cwd_free_gb = shutil.disk_usage(cwd).free / (1024 ** 3)
    except OSError:
        cwd_free_gb = 0.0
    if cwd_free_gb >= 15:
        return os.path.join(cwd, ".pidm_work")

    best_drive, best_free_bytes = None, 0
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        drive = f"{letter}:\\"
        if not os.path.exists(drive):
            continue
        try:
            free = shutil.disk_usage(drive).free
        except OSError:
            continue
        if free > best_free_bytes:
            best_free_bytes, best_drive = free, letter

    if best_drive and best_free_bytes / (1024 ** 3) >= 15:
        chosen = f"{best_drive}:\\PIDM_work"
        print(f"[local_runner] Current drive has only {cwd_free_gb:.1f}GB free — "
              f"redirecting HF cache / checkpoints / dataset to {chosen}")
        return chosen

    return os.path.join(cwd, ".pidm_work")


def main() -> None:
    args = parse_args()

    work_dir = _pick_work_dir(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    # Must be set before transformers/datasets/huggingface_hub are imported
    # anywhere (including transitively through pidm.detection/pidm.data).
    os.environ.setdefault("HF_HOME", os.path.join(work_dir, "hf_cache"))

    from pidm.config import CONFIG

    CONFIG.output_dir           = os.path.join(work_dir, "pidm_output")
    CONFIG.dataset_path         = os.path.join(work_dir, "pidm_dataset.csv")
    CONFIG.paraphrase_cache_path = os.path.join(work_dir, "paraphrase_cache.csv")
    CONFIG.model_save_path      = os.path.join(work_dir, "pidm_classifier")
    CONFIG.scale_study_dir      = os.path.join(CONFIG.output_dir, "scale_study")
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    os.makedirs(CONFIG.scale_study_dir, exist_ok=True)
    print(f"[local_runner] Work dir: {work_dir}")

    if args.epochs is not None:
        CONFIG.num_epochs = args.epochs
    if args.no_real_only_test:
        CONFIG.real_only_test = False
    if args.no_paraphrase:
        CONFIG.use_paraphrase = False

    from pidm.main import main as pidm_main

    pidm_main(
        dataset_size    = args.dataset_size,
        run_adversarial = not args.no_adversarial,
        run_scale_study = args.scale_study,
        launch_demo     = not args.no_demo,
    )


if __name__ == "__main__":
    main()
