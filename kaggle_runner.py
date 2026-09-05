"""
PIDM Kaggle Runner
==================
Run this in a Kaggle notebook cell after uploading the repo as a dataset,
or let it clone from GitHub directly.

Step-by-step setup in Kaggle:
  1. Create a new Notebook on kaggle.com
  2. Settings (right sidebar) → Accelerator → GPU T4 x2  (or P100)
  3. Settings → Internet → ON  (required to download HuggingFace models)
  4. Upload this repo as a Kaggle Dataset, OR push to GitHub and let the
     script clone it (set GITHUB_REPO below)
  5. In a notebook cell, run:
       !python /kaggle/input/<your-dataset-slug>/kaggle_runner.py
     OR if cloning from GitHub:
       !python kaggle_runner.py

Outputs are saved to /kaggle/working/pidm_output/ and automatically
appear in the notebook's Output tab after the session ends.
"""
from __future__ import annotations

import os
import subprocess
import sys

# ── Configuration ────────────────────────────────────────────────────────────
# If you push the repo to GitHub, set this URL. Otherwise leave as None and
# upload the repo as a Kaggle dataset instead.
GITHUB_REPO   = "https://github.com/Moozaheed/IIUC-LLM.git"
GITHUB_BRANCH = "feedback-from-teacher"   # branch that contains the latest fixes

# Kaggle's fixed paths
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT   = "/kaggle/input"

# Where all heavy outputs go (visible in Kaggle's Output tab)
SAVE_DIR       = os.path.join(KAGGLE_WORKING, "pidm_output")
DATASET_CSV    = os.path.join(KAGGLE_WORKING, "pidm_dataset.csv")
MODEL_DIR      = os.path.join(KAGGLE_WORKING, "pidm_classifier")
HF_CACHE       = os.path.join(KAGGLE_WORKING, "hf_cache")
SCALE_DIR      = os.path.join(SAVE_DIR, "scale_study")
# ─────────────────────────────────────────────────────────────────────────────


def _is_kaggle() -> bool:
    return os.path.isdir(KAGGLE_WORKING)


def _install_deps() -> None:
    print("=" * 60)
    print("Installing dependencies ...")
    print("=" * 60)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "transformers==4.44.2", "datasets==2.20.0",
        "sentence-transformers==3.0.1", "networkx==3.3",
        "scikit-learn==1.5.1", "accelerate==0.33.0",
        "seaborn==0.13.2", "matplotlib==3.9.2",
        "pandas==2.2.2", "sentencepiece==0.2.0",
        "protobuf==5.27.3", "scipy==1.14.0",
        # Gradio skipped — share=True unreliable on Kaggle;
        # add manually if you need the demo.
    ])
    print("Dependencies installed.\n")


def _find_repo_root() -> str:
    """
    Returns the path that contains the pidm/ package.
    Search order:
      1. Current working directory (if running from inside the repo)
      2. Any /kaggle/input/<slug>/  folder that contains pidm/
      3. Clone from GITHUB_REPO into /kaggle/working/IIUC-LLM/
    """
    # Already inside the repo?
    if os.path.isdir(os.path.join(os.getcwd(), "pidm")):
        return os.getcwd()

    # Uploaded as a Kaggle dataset?
    if os.path.isdir(KAGGLE_INPUT):
        for slug in os.listdir(KAGGLE_INPUT):
            candidate = os.path.join(KAGGLE_INPUT, slug)
            if os.path.isdir(os.path.join(candidate, "pidm")):
                print(f"Found repo in Kaggle dataset: {candidate}")
                return candidate

    # Clone from GitHub (fallback when no dataset uploaded)
    if GITHUB_REPO:
        clone_dir = os.path.join(KAGGLE_WORKING, "IIUC-LLM")
        if not os.path.isdir(clone_dir):
            print(f"Cloning {GITHUB_REPO} (branch: {GITHUB_BRANCH}) ...")
            subprocess.check_call([
                "git", "clone",
                "--branch", GITHUB_BRANCH,
                "--depth", "1",
                GITHUB_REPO, clone_dir,
            ])
        elif os.path.isdir(os.path.join(clone_dir, ".git")):
            print("Repo already cloned — pulling latest ...")
            try:
                subprocess.check_call(["git", "-C", clone_dir, "pull"])
            except Exception as e:
                print(f"Git pull skipped: {e}")
        else:
            print(f"Using unpacked repo in {clone_dir}")
        return clone_dir

    raise RuntimeError(
        "Cannot find the PIDM repo. Either:\n"
        "  (a) Upload the repo as a Kaggle dataset, or\n"
        "  (b) Set GITHUB_REPO at the top of this file."
    )


def main() -> None:
    if not _is_kaggle():
        print("Warning: /kaggle/working not found. "
              "Running outside Kaggle — paths may differ.")

    _install_deps()

    repo_root = _find_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"Repo root: {repo_root}\n")

    # Point HuggingFace cache to working dir so models persist in output
    os.environ["HF_HOME"]            = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    # Apply config overrides before any PIDM imports
    from pidm.config import CONFIG

    CONFIG.output_dir            = SAVE_DIR
    CONFIG.dataset_path          = DATASET_CSV
    CONFIG.model_save_path       = MODEL_DIR
    CONFIG.paraphrase_cache_path = os.path.join(KAGGLE_WORKING, "paraphrase_cache.csv")
    CONFIG.scale_study_dir       = SCALE_DIR

    for d in [SAVE_DIR, MODEL_DIR, SCALE_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"Device : {CONFIG.device}")
    print(f"Model  : {CONFIG.classifier_model}")
    print(f"Outputs: {SAVE_DIR}\n")

    # Kaggle T4/P100 has 16GB VRAM — bump batch size for faster training
    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    if vram_gb >= 14:
        CONFIG.batch_size = 16
        CONFIG.gradient_accumulation_steps = 2
        print(f"Kaggle GPU detected ({vram_gb:.1f}GB) — using batch_size=16, grad_accum=2 for FP32 stability\n")

    from pidm.main import main as pidm_main

    pidm_main(
        run_adversarial = True,
        run_scale_study = False,   # enable with --scale-study locally; too slow for one Kaggle session
        launch_demo     = False,   # Gradio share=True unreliable on Kaggle
    )

    # Print all output files for easy reference
    print("\n" + "=" * 60)
    print("OUTPUT FILES (download from Kaggle Output tab):")
    print("=" * 60)
    for f in sorted(os.listdir(SAVE_DIR)):
        fpath = os.path.join(SAVE_DIR, f)
        if os.path.isfile(fpath):
            print(f"  {f:<45} {os.path.getsize(fpath) / 1024:>8.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
