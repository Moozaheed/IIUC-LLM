"""
PIDM Colab Runner

Run this file in Google Colab (Runtime -> T4 GPU) in a fresh cell:

    !python colab_runner.py

It clones/pulls the repo, installs dependencies, mounts Drive for
persistent output storage, and runs the full pipeline via pidm.main.
Session-length limits on Colab mean the multi-hour scale-up study is
skipped by default here — run that locally (see local_runner.py) or in
a Colab Pro background session instead.
"""
from __future__ import annotations

import os
import subprocess
import sys

REPO_URL  = "https://github.com/Moozaheed/IIUC-LLM.git"
REPO_DIR  = "/content/IIUC-LLM"

print("Installing packages ...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2", "datasets==2.20.0", "sentence-transformers==3.0.1",
    "networkx==3.3", "scikit-learn==1.5.1", "gradio==4.42.0", "torch",
    "accelerate==0.33.0", "seaborn==0.13.2", "matplotlib==3.9.2",
    "pandas==2.2.2", "sentencepiece==0.2.0", "protobuf==5.27.3",
    "scipy==1.14.0",
])

try:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/PIDM_Thesis"
    IN_COLAB = True
except Exception:
    SAVE_DIR = "./PIDM_Thesis"
    IN_COLAB = False

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory: {SAVE_DIR}\n")

if IN_COLAB:
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, ".")

from pidm.config import CONFIG  # noqa: E402

CONFIG.output_dir      = f"{SAVE_DIR}/pidm_output"
CONFIG.dataset_path    = f"{SAVE_DIR}/pidm_dataset.csv"
CONFIG.model_save_path = f"{SAVE_DIR}/pidm_classifier"
CONFIG.scale_study_dir = f"{SAVE_DIR}/pidm_output/scale_study"
os.makedirs(CONFIG.output_dir, exist_ok=True)
os.makedirs(CONFIG.scale_study_dir, exist_ok=True)

print(f"Device : {CONFIG.device}")
print(f"Model  : {CONFIG.classifier_model}")

from pidm.main import main  # noqa: E402

main(run_adversarial=True, run_scale_study=False, launch_demo=True)

print(f"\nAll outputs saved to: {SAVE_DIR}")
