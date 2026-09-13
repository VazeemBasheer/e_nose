"""
champion_model.py
==================
Compares every model/variant evaluated so far (baseline fixed-defaults from
train_baseline.py, tuned from tune_models.py) and selects a single champion
for test-set evaluation.

Selection rule follows the project's stated evaluation priority (from the
original spec): clinical sensitivity (recall) first, AUROC second, macro F1
third — NOT a weighted average of the three. This is a strict lexicographic
ranking: a model only wins on AUROC if it's tied (to 3 decimal places) with
the leader on sensitivity, and so on. Ties are broken by lower std across
folds (a model that's inconsistent across folds is a worse bet at this
sample size even with the same mean).

This does NOT touch the test split — selection is based entirely on the
train-split CV results already produced. The champion is only a candidate
for final test evaluation, not a claim about real-world performance yet.

Inputs (relative to project root):
  data/processed/cv_results.csv                 (from train_baseline.py)
  data/processed/tuning_nested_cv_results.csv    (from tune_models.py, optional
                                                   — script still runs without it)

Outputs:
  data/processed/model_comparison.csv   — every model/variant, mean+-std, rank
  data/processed/champion_model.json    — selected model, variant, metrics, rationale
  models/champion_model.joblib          — copy of the winning fitted model
"""

import json
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # src/champion_model.py -> project root
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

METRICS = ["sensitivity_macro", "auroc_macro_ovr", "macro_f1"]  # priority order, in this order
TIE_DECIMALS = 3  # metrics within this many decimals are treated as tied for ranking purposes


def load_results():
    frames = []

    baseline_path = PROCESSED_DIR / "cv_results.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline["variant"] = "baseline"
        frames.append(baseline)

    tuned_path = PROCESSED_DIR / "tuning_nested_cv_results.csv"
    if tuned_path.exists():
        tuned = pd.read_csv(tuned_path)
        tuned["variant"] = "tuned"
        frames.append(tuned)

    if not frames:
        raise FileNotFoundError(
            "Neither cv_results.csv nor tuning_nested_cv_results.csv found — "
            "run train_baseline.py (and optionally tune_models.py) first."
        )

    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["model", "variant"])[METRICS]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = results.groupby(["model", "variant"]).size()
    return summary.reset_index()


def rank_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Lexicographic sort by METRICS in priority order (rounded for tie
    handling), then by lower std on the top-priority metric as final
    tiebreak."""
    sort_cols = []
    ascending = []
    for metric in METRICS:
        rounded_col = f"{metric}_rounded"
        summary[rounded_col] = summary[f"{metric}_mean"].round(TIE_DECIMALS)
        sort_cols.append(rounded_col)
        ascending.append(False)
    sort_cols.append(f"{METRICS[0]}_std")
    ascending.append(True)  # lower std on the top-priority metric wins ties

    ranked = summary.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked = ranked.drop(columns=[f"{m}_rounded" for m in METRICS])
    return ranked


def find_model_file(model: str, variant: str) -> Path:
    filename = f"{model}_tuned.joblib" if variant == "tuned" else f"{model}.joblib"
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected fitted model at {path} for champion {model} ({variant}) "
            "but it's missing — re-run train_baseline.py / tune_models.py."
        )
    return path


def main():
    results = load_results()
    summary = summarize(results)
    ranked = rank_models(summary)

    ranked.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    print("=== Model comparison (ranked by sensitivity > AUROC > macro F1) ===")
    print(ranked.to_string(index=False))

    champion = ranked.iloc[0]
    champion_model, champion_variant = champion["model"], champion["variant"]

    source_path = find_model_file(champion_model, champion_variant)
    champion_path = MODELS_DIR / "champion_model.joblib"
    shutil.copy(source_path, champion_path)

    champion_info = {
        "model": champion_model,
        "variant": champion_variant,
        "source_file": source_path.name,
        "metrics": {m: {"mean": champion[f"{m}_mean"], "std": champion[f"{m}_std"]} for m in METRICS},
        "n_folds": int(champion["n_folds"]),
        "selection_rule": (
            "Strict priority: sensitivity_macro > auroc_macro_ovr > macro_f1 "
            f"(tied within {TIE_DECIMALS} decimals), then lower std on "
            "sensitivity_macro as final tiebreak. Based on train-split CV "
            "only — not yet evaluated on the held-out test split."
        ),
        "runner_up": {
            "model": ranked.iloc[1]["model"] if len(ranked) > 1 else None,
            "variant": ranked.iloc[1]["variant"] if len(ranked) > 1 else None,
        } if len(ranked) > 1 else None,
    }
    with open(PROCESSED_DIR / "champion_model.json", "w") as f:
        json.dump(champion_info, f, indent=2)

    print(f"\nChampion: {champion_model} ({champion_variant})")
    print(f"  sensitivity_macro: {champion['sensitivity_macro_mean']:.3f} +/- {champion['sensitivity_macro_std']:.3f}")
    print(f"  auroc_macro_ovr:   {champion['auroc_macro_ovr_mean']:.3f} +/- {champion['auroc_macro_ovr_std']:.3f}")
    print(f"  macro_f1:          {champion['macro_f1_mean']:.3f} +/- {champion['macro_f1_std']:.3f}")
    print(f"\nSaved model_comparison.csv, champion_model.json, and "
          f"models/champion_model.joblib (copied from {source_path.name}).")
    print("Reminder: champion is a train-CV-based candidate. Evaluate ONCE on "
          "the held-out test split before treating this as a final result.")


if __name__ == "__main__":
    main()