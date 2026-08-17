# Google Colab — Step-by-Step Run Guide
## PIDM: Prompt Injection Detection Middleware

The pipeline now lives in the `pidm/` package (see [README.md](README.md) for the layout) instead
of a single script, so the Colab flow is a single entrypoint call rather than 10 separate cells.

---

## STEP 1 — Open Google Colab

Go to: **https://colab.research.google.com**
- Sign in with your Google account
- Click **"New notebook"**

---

## STEP 2 — Set Runtime to GPU (CRITICAL)

Without this, training takes hours instead of tens of minutes.

1. Click **Runtime** (top menu)
2. Click **"Change runtime type"**
3. Under **Hardware accelerator** → select **T4 GPU**
4. Click **Save**

You will see a green indicator "Connected to T4" in the top-right corner.

---

## STEP 3 — Push the repo to GitHub first (one-time, from your own machine)

`colab_runner.py` clones the repo rather than requiring file-by-file uploads. Before your first
Colab run:
1. Update `REPO_URL` at the top of [colab_runner.py](colab_runner.py) to your actual GitHub URL.
2. `git init && git add . && git commit -m "PIDM v2"` and push to GitHub (see repo root — it isn't
   initialised as a git repo yet).

If you'd rather not push to GitHub, upload the whole `pidm/` folder plus `colab_runner.py` to
`/content/` via the Colab file browser and skip the git-clone step (comment it out in
`colab_runner.py` — the `else` branch already handles running from the current directory).

---

## STEP 4 — Run

Create one cell and run it:
```python
!python colab_runner.py
```

**Wait for package install (~2-3 min)**, then Drive will ask you to authorize mounting — click
"Connect to Google Drive" and allow access. Everything after that runs unattended:

1. Builds the ~40,000-row dataset (paraphrase-augmented synthetic attacks + scripted multi-agent
   conversation traces + real-world data held out as the test set)
2. Fine-tunes the classifier (DeBERTa-v3-base/small, auto-selected — T4 has 16GB, so base fits
   comfortably)
3. Runs the full ablation evaluation, the 6-system baseline comparison, and the adversarial
   robustness suite
4. Runs the pipeline simulation and saves the GCPD agent graph
5. Launches the Gradio demo with a public `https://xxxxx.gradio.live` link — share that with your
   supervisor for a live demo

The multi-hour dataset-size scale-up study is **not** run on Colab by default (session limits) —
run it locally instead: `python local_runner.py --scale-study` (see README's local Quick Start).

---

## STEP 5 — Download All Outputs

Your Google Drive folder `PIDM_Thesis/` will have:

```
PIDM_Thesis/
├── pidm_output/
│   ├── confusion_matrix.png       <- for thesis Chapter 7
│   ├── roc_curve.png              <- for thesis Chapter 7
│   ├── ablation_study.png         <- for thesis Chapter 7
│   ├── per_attack_type.png        <- for thesis Chapter 7
│   ├── latency.png                <- for thesis Chapter 7
│   ├── agent_graph.png            <- for thesis Chapter 7
│   ├── baseline_comparison.png/csv
│   ├── baseline_radar.png
│   ├── dataset_statistics.png
│   ├── scale_study/               <- only if you ran it locally and synced the folder up
│   └── quarantine_log.json        <- audit trail
├── pidm_dataset.csv               <- your dataset (release this!)
└── pidm_classifier/               <- trained model files
```

---

## Expected Timeline (T4 GPU, ~40k rows)

| Stage | Time |
|---|---|
| Package install | 2-3 min |
| Dataset build (paraphrase + traces + real data) | 10-20 min (paraphrase pass is the slow part; cached after first run) |
| Classifier training | 30-50 min |
| Evaluation + baselines + adversarial suite | 5-10 min |
| Pipeline simulation + Gradio launch | ~1 min |
| **Total** | **~1-1.5 hours** |

---

## If the Session Times Out During Training

Colab free tier disconnects after ~90 minutes of inactivity.
1. Reconnect runtime, re-run `!python colab_runner.py`
2. The paraphrase cache (`pidm/data/paraphrase_cache.csv`) and any saved classifier checkpoint
   under `PIDM_Thesis/pidm_classifier/` are reused automatically if present on Drive, so a second
   run doesn't redo finished work from scratch.

---

## Tips

- **Keep the browser tab active** during training (move the mouse occasionally, or use Colab Pro).
- **Colab Pro** gives longer sessions and faster GPUs — useful for the paraphrase pass on the full
  40k dataset the first time you build it.
- Prefer running the heavy, multi-hour parts (full 40k training, scale study, adversarial suite)
  **locally** on the RTX 3070 via `local_runner.py` — no session-timeout risk, and outputs land
  directly in your project folder instead of needing a Drive round-trip.
