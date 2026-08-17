"""Main PIDM pipeline: build dataset -> train classifier -> evaluate ->
compare baselines -> (optional) adversarial suite -> (optional) scale
study -> pipeline simulation -> Gradio demo.
"""
from __future__ import annotations

import os

import pandas as pd

from pidm.config import CONFIG, logger
from pidm.data import build_dataset
from pidm.detection.classifier import InjectionClassifier
from pidm.detection.graph_cascade_detector import GraphAwareCascadeDetector
from pidm.detection.orchestrator import PIDMOrchestrator
from pidm.detection.quarantine import QuarantineEngine
from pidm.detection.semantic_intent_drift import SemanticIntentDrift
from pidm.eval.adversarial_suite import AdversarialSuite
from pidm.eval.baseline_comparator import BaselineComparator
from pidm.eval.evaluator import PIDMEvaluator, plot_dataset_statistics
from pidm.eval.scale_study import ScaleStudy
from pidm.sim.pipeline_simulator import SyntheticPipelineSimulator

_DEFAULT_INTENT = (
    "This pipeline coordinates multiple AI agents to perform legitimate "
    "data analysis, report generation, and workflow automation tasks."
)


def main(
    dataset_size:    int  = None,
    run_adversarial: bool = True,
    run_scale_study: bool = False,
    launch_demo:     bool = True,
) -> None:
    print("\n" + "=" * 70)
    print("  PIDM — Prompt Injection Detection Middleware")
    print("  International Islamic University Chittagong (IIUC)")
    print("=" * 70 + "\n")

    # -- Step 1: Build dataset (synthetic + paraphrase + traces + real) --
    print("[1/8] Building dataset (synthetic + paraphrase + traces + real-world) ...")
    df_train_val, df_test = build_dataset(target_size=dataset_size)
    plot_dataset_statistics(pd.concat([df_train_val, df_test], ignore_index=True))
    print(f"   train_val={len(df_train_val):,} | test={len(df_test):,} "
          f"(real_only_test={CONFIG.real_only_test})")

    # -- Step 2: Train classifier --
    print("\n[2/8] Training classifier ...")
    classifier = InjectionClassifier()
    classifier.train(df_train_val)

    # -- Step 3: Assemble PIDM --
    print("\n[3/8] Initialising SID, GCPD, Quarantine, Orchestrator ...")
    sid = SemanticIntentDrift()
    sid.set_pipeline_intent(_DEFAULT_INTENT)
    gcpd       = GraphAwareCascadeDetector()
    quarantine = QuarantineEngine()
    pidm       = PIDMOrchestrator(classifier, sid, gcpd, quarantine)

    # -- Step 4: Ablation evaluation --
    print("\n[4/8] Running PIDM ablation evaluation ...")
    evaluator = PIDMEvaluator(pidm, df_test)
    results   = evaluator.run_full()
    evaluator.print_report(results)
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc()
    evaluator.plot_ablation(results)
    evaluator.plot_per_attack_type(results)
    evaluator.plot_latency()

    # -- Step 5: Baseline comparison --
    print("\n[5/8] Running baseline comparison ...")
    comparator  = BaselineComparator(train_df=df_train_val)
    df_baseline = comparator.run(df_test, pidm)
    comparator.improvement_summary(df_baseline)

    # -- Step 6: Adversarial robustness suite --
    if run_adversarial:
        print("\n[6/8] Running adversarial robustness suite ...")
        AdversarialSuite(pidm).run()
    else:
        print("\n[6/8] Skipping adversarial suite (run_adversarial=False)")

    # -- Step 7: Scale-up study (optional — multi-hour on a single GPU) --
    if run_scale_study:
        print("\n[7/8] Running dataset-size scale-up study ...")
        ScaleStudy(df_train_val, df_test).run()
    else:
        print("\n[7/8] Skipping scale-up study (run_scale_study=False) — "
              "run separately via ScaleStudy for a multi-hour background job.")

    # -- Step 7b: Pipeline simulation & GCPD graph --
    print("\n[7b/8] Running synthetic pipeline simulation ...\n")
    sim = SyntheticPipelineSimulator()
    sim.run(pidm)
    quarantine.save_log()
    gcpd.draw_graph(save_path=os.path.join(CONFIG.output_dir, "agent_graph.png"))
    print("\nQuarantine summary:", quarantine.summary())

    # -- Step 8: Gradio demo --
    if launch_demo:
        print("\n[8/8] Launching Gradio demo ...")
        from pidm.demo.gradio_app import build_gradio_demo
        demo = build_gradio_demo(pidm)
        if demo:
            demo.launch(share=True, debug=False)
        else:
            print("   Gradio unavailable — install with: pip install gradio")
    else:
        print("\n[8/8] Skipping Gradio demo (launch_demo=False)")

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {CONFIG.output_dir}")
    if os.path.isdir(CONFIG.output_dir):
        for f in sorted(os.listdir(CONFIG.output_dir)):
            fpath = os.path.join(CONFIG.output_dir, f)
            if os.path.isfile(fpath):
                print(f"    {f:<40} {os.path.getsize(fpath)/1024:>8.1f} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
