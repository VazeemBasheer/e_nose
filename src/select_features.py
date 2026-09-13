"""
select_features.py
===================
Prunes redundant/collinear features flagged in eda.py's correlation section.
Selection criteria are FIT ON TRAIN ONLY, then applied identically to test —
fitting on test would leak information into the "held-out" set.

Steps:
  1. Manual drop: per-sensor `_min`, `_max`, `_std` columns. EDA showed these
     are near-exact duplicates of other columns already kept:
       r0 == min (r=1.00), peak_value == max (r=1.00),
       peak_amplitude/decay_rate ~= std (r=0.99) for several sensors.
  2. Near-zero-variance drop (fit on train): kills dead-signal columns, e.g.
     TGS800_decay_rate/rise_rate, TGS813_decay_rate (flagged in EDA section A).
  3. Greedy pairwise-correlation pruning (fit on train, threshold configurable):
     catches remaining redundancy the manual rule didn't name explicitly,
     e.g. cross-sensor collinearity like MQ137 vs MQ138 (up to r=1.00 on
     peak_amplitude). For each correlated pair, the later column (by column
     order) is dropped and the earlier one kept.

Inputs (data/interim/, relative to project root):
  feature_train.parquet, feature_test.parquet

Outputs:
  data/interim/feature_train_selected.parquet
  data/interim/feature_test_selected.parquet
  data/processed/selected_features.csv   (kept feature list + drop reasons for
                                           everything removed)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # src/select_features.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]

MANUAL_DROP_SUFFIXES = ("_min", "_max", "_std")
VARIANCE_THRESHOLD = 1e-8
CORR_THRESHOLD = 0.95


def numeric_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLS]


def manual_drop(cols: list) -> tuple:
    """Return (kept, dropped_with_reason) after stripping known-redundant suffixes."""
    dropped = {c: f"manual: redundant with r0/peak_value/decay_rate (suffix {c[c.rfind('_'):]})"
               for c in cols if c.endswith(MANUAL_DROP_SUFFIXES)}
    kept = [c for c in cols if c not in dropped]
    return kept, dropped


def variance_drop(train_df: pd.DataFrame, cols: list) -> tuple:
    variances = train_df[cols].var()
    low_var = variances[variances < VARIANCE_THRESHOLD].index.tolist()
    dropped = {c: f"near-zero variance on train (var={variances[c]:.2e})" for c in low_var}
    kept = [c for c in cols if c not in dropped]
    return kept, dropped


def correlation_prune(train_df: pd.DataFrame, cols: list, threshold: float) -> tuple:
    corr = train_df[cols].corr().abs()
    dropped = {}
    kept = []
    for col in cols:
        if col in dropped:
            continue
        kept.append(col)
        # anything strongly correlated with this kept column, appearing later, gets dropped
        correlated_with_col = corr.index[(corr[col] > threshold) & (corr.index != col)]
        for other in correlated_with_col:
            if other in cols and other not in kept and other not in dropped:
                r = corr.loc[col, other]
                dropped[other] = f"corr={r:.2f} with kept column '{col}' (threshold {threshold})"
    return kept, dropped


def main():
    train_path = INTERIM_DIR / "feature_train.parquet"
    test_path = INTERIM_DIR / "feature_test.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run ingest.py, preprocess.py, split.py first.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path) if test_path.exists() else None

    all_cols = numeric_feature_cols(train_df)
    log.info(f"Starting numeric feature columns: {len(all_cols)}")

    all_drop_reasons = {}

    cols, dropped = manual_drop(all_cols)
    all_drop_reasons.update(dropped)
    log.info(f"After manual _min/_max/_std drop: {len(cols)} columns ({len(dropped)} dropped)")

    cols, dropped = variance_drop(train_df, cols)
    all_drop_reasons.update(dropped)
    log.info(f"After near-zero-variance drop: {len(cols)} columns ({len(dropped)} dropped)")

    cols, dropped = correlation_prune(train_df, cols, CORR_THRESHOLD)
    all_drop_reasons.update(dropped)
    log.info(f"After correlation pruning (>{CORR_THRESHOLD}): {len(cols)} columns ({len(dropped)} dropped)")

    id_cols_present = [c for c in ID_COLS if c in train_df.columns]
    final_cols = id_cols_present + cols

    train_selected = train_df[final_cols]
    train_selected.to_parquet(INTERIM_DIR / "feature_train_selected.parquet", index=False)

    if test_df is not None:
        test_selected = test_df[[c for c in final_cols if c in test_df.columns]]
        test_selected.to_parquet(INTERIM_DIR / "feature_test_selected.parquet", index=False)
        log.info(f"Saved feature_train_selected.parquet ({train_selected.shape}) "
            f"and feature_test_selected.parquet ({test_selected.shape})")
    else:
        log.info(f"Saved feature_train_selected.parquet ({train_selected.shape}) "
            f"— no feature_test.parquet found, test file skipped")

    # record what was kept vs dropped, and why, for the report trail
    summary_rows = [{"feature": c, "status": "kept", "reason": ""} for c in cols]
    summary_rows += [{"feature": c, "status": "dropped", "reason": r} for c, r in all_drop_reasons.items()]
    summary_df = pd.DataFrame(summary_rows).sort_values(["status", "feature"])
    summary_df.to_csv(PROCESSED_DIR / "selected_features.csv", index=False)

    log.info(f"\nFinal: {len(cols)} feature columns kept, {len(all_drop_reasons)} dropped "
        f"(of {len(all_cols)} original)")
    log.info(f"Selection log saved to data/processed/selected_features.csv")


if __name__ == "__main__":
    main()