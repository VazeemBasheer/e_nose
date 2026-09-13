"""
evaluate_test.py
================
Final evaluation of the champion model on the held-out test split. This is
meant to run ONCE, after model selection is fully locked in from CV alone
(train_baseline.py -> tune_models.py -> champion_model.py). Looking at the
test set more than once and adjusting anything in response turns it into a
second validation set and invalidates the whole point of holding it out.

Guardrail: if data/processed/test_evaluation.json already exists, this
script refuses to overwrite it unless run with --force. If you're re-running
because you changed the model/features/pipeline, that's a signal you're
tuning against the test set — go back to CV, not here.

Inputs (relative to project root):
  models/champion_model.joblib
  data/processed/champion_model.json      (for context/reporting only)
  data/interim/feature_test_selected.parquet

Outputs:
  data/processed/test_evaluation.json         — final metrics + per-class report
  reports/figures/test_confusion_matrix.png
  reports/figures/test_roc_curves.png
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent  # src/evaluate_test.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
for d in (PROCESSED_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["air", "control", "copd", "smokers"]  # must match train_baseline.py / tune_models.py

ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]


def plot_confusion(y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Champion model — held-out TEST confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_proba, out_path):
    y_bin = label_binarize(y_true, classes=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, cls in enumerate(CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Champion model — held-out TEST ROC (one-vs-rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing test_evaluation.json. Only use this "
                              "if you're SURE this isn't a second look after changing "
                              "anything in response to a prior test result.")
    args = parser.parse_args()

    result_path = PROCESSED_DIR / "test_evaluation.json"
    if result_path.exists() and not args.force:
        existing = json.loads(result_path.read_text())
        raise SystemExit(
            f"\ntest_evaluation.json already exists (evaluated {existing.get('evaluated_at')}, "
            f"model={existing.get('champion_model')} [{existing.get('champion_variant')}]).\n"
            "Refusing to overwrite: the test split should be evaluated ONCE. If you changed "
            "the model, features, or pipeline after seeing that result, that change was "
            "informed by the test set — go back and re-select via CV instead.\n"
            "If you're certain this is a legitimate first-time run (e.g. the pipeline was "
            "reset from scratch and this file is stale), re-run with --force.\n"
        )

    champion_path = MODELS_DIR / "champion_model.joblib"
    champion_info_path = PROCESSED_DIR / "champion_model.json"
    test_path = INTERIM_DIR / "feature_test_selected.parquet"
    for p in (champion_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run the pipeline through champion_model.py first.")

    champion_info = json.loads(champion_info_path.read_text()) if champion_info_path.exists() else {}
    model = joblib.load(champion_path)

    df = pd.read_parquet(test_path)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X_test = df[feature_cols]
    y_test = df["group"].map({cls: i for i, cls in enumerate(CLASSES)}).to_numpy()

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    sensitivity = recall_score(y_test, pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    y_bin = label_binarize(y_test, classes=range(len(CLASSES)))
    present = y_bin.sum(axis=0) > 0
    try:
        auroc = roc_auc_score(y_bin[:, present], proba[:, present], average="macro", multi_class="ovr")
    except ValueError:
        auroc = float("nan")

    class_report = classification_report(
        y_test, pred, labels=range(len(CLASSES)), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )

    plot_confusion(y_test, pred, FIGURES_DIR / "test_confusion_matrix.png")
    plot_roc(y_test, proba, FIGURES_DIR / "test_roc_curves.png")

    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "champion_model": champion_info.get("model", "unknown"),
        "champion_variant": champion_info.get("variant", "unknown"),
        "n_test_samples": len(df),
        "test_class_distribution": df["group"].value_counts().to_dict(),
        "metrics": {
            "sensitivity_macro": sensitivity,
            "macro_f1": macro_f1,
            "auroc_macro_ovr": auroc,
        },
        "per_class_report": class_report,
        "cv_estimate_for_comparison": champion_info.get("metrics", {}),
    }
    result_path.write_text(json.dumps(result, indent=2))

    print(f"=== FINAL TEST EVALUATION ({champion_info.get('model', 'unknown')} "
          f"[{champion_info.get('variant', 'unknown')}]) ===")
    print(f"n_test_samples: {len(df)}  class distribution: {df['group'].value_counts().to_dict()}")
    print(f"sensitivity_macro: {sensitivity:.3f}")
    print(f"macro_f1:          {macro_f1:.3f}")
    print(f"auroc_macro_ovr:   {auroc:.3f}")
    print("\nCompare against the CV estimate that selected this model "
          f"(data/processed/champion_model.json): {champion_info.get('metrics', {})}")
    print("A test result meaningfully worse than the CV estimate suggests the CV process "
          "was optimistic (e.g. AIR/ambient samples being trivially easy, as flagged earlier) "
          "— a test result close to or better than CV is the reassuring outcome.")
    print(f"\nSaved test_evaluation.json, test_confusion_matrix.png, test_roc_curves.png.")


if __name__ == "__main__":
    main()