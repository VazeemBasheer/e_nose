"""
split.py
========
Creates a held-out test set from feature.parquet, respecting subject-level
grouping where it's known, so a subject's replicate measurements never end
up split across train and test.

Split strategy per group:
  - COPD, CONTROL: subject-level split (GroupShuffleSplit on subject_id) —
    both replicates of a subject stay together.
  - SMOKERS, AIR: no reliable subject_id (per Phase 1), so these are split
    at the sample level instead. Documented caveat, not a workaround to
    hide.

The held-out test set is meant for FINAL evaluation only. Model selection /
hyperparameter tuning should use GroupKFold or Leave-One-Subject-Out on the
train split, not this test set.

Output: adds a "split" column (train/test) to the feature table and writes:
  - feature_train.parquet
  - feature_test.parquet
(Both are also derivable by filtering feature_split.parquet on "split", also
written for convenience.)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # src/split.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = INTERIM_DIR / "feature.parquet"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# groups with a reliable subject_id to split on vs. groups split at sample level
SUBJECT_LEVEL_GROUPS = {"copd", "control"}
SAMPLE_LEVEL_GROUPS = {"smokers", "air"}


def split_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["subject_id"]))
    df = df.copy()
    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df


def split_sample_level(df: pd.DataFrame) -> pd.DataFrame:
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    df = df.copy()
    df["split"] = "train"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    return df


def main():
    features = pd.read_parquet(INPUT_PATH)

    parts = []
    for group in features["group"].unique():
        g_df = features[features["group"] == group].reset_index(drop=True)

        if group in SUBJECT_LEVEL_GROUPS:
            g_df = split_subject_level(g_df)
            n_subj_train = g_df.loc[g_df["split"] == "train", "subject_id"].nunique()
            n_subj_test = g_df.loc[g_df["split"] == "test", "subject_id"].nunique()
            log.info(
                f"[{group}] subject-level split: "
                f"{n_subj_train} subjects/train, {n_subj_test} subjects/test "
                f"({(g_df['split']=='train').sum()} samples/train, "
                f"{(g_df['split']=='test').sum()} samples/test)"
            )
        elif group in SAMPLE_LEVEL_GROUPS:
            g_df = split_sample_level(g_df)
            log.info(
                f"[{group}] sample-level split (no subject_id available): "
                f"{(g_df['split']=='train').sum()} samples/train, "
                f"{(g_df['split']=='test').sum()} samples/test"
            )
        else:
            raise ValueError(f"Unhandled group '{group}' — add it to one of the split sets above")

        parts.append(g_df)

    result = pd.concat(parts, ignore_index=True)

    # sanity check: no subject appears in both splits (for the subject-level groups)
    for group in SUBJECT_LEVEL_GROUPS:
        g_df = result[result["group"] == group]
        train_subj = set(g_df.loc[g_df["split"] == "train", "subject_id"])
        test_subj = set(g_df.loc[g_df["split"] == "test", "subject_id"])
        overlap = train_subj & test_subj
        assert not overlap, f"[{group}] subject leakage across split: {overlap}"

    result.to_parquet(INTERIM_DIR / "feature_split.parquet", index=False)
    result[result["split"] == "train"].drop(columns="split").to_parquet(
        INTERIM_DIR / "feature_train.parquet", index=False
    )
    result[result["split"] == "test"].drop(columns="split").to_parquet(
        INTERIM_DIR / "feature_test.parquet", index=False
    )

    log.info(f"\nSaved feature_split.parquet, feature_train.parquet, feature_test.parquet")
    log.info(f"Overall: {result.shape[0]} samples -> "
              f"{(result['split']=='train').sum()} train / {(result['split']=='test').sum()} test")


if __name__ == "__main__":
    main()