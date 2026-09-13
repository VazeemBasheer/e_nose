"""
tune_models.py
==============
Hyperparameter tuning for the two models worth tuning (xgboost,
logistic_regression — see baseline CV results; linear_svm stays at its
fixed-default baseline for now, its instability didn't look like a
C-tuning problem).

Uses NESTED cross-validation so the reported performance is honest:
  - Outer GroupKFold (N_OUTER_SPLITS): held out for evaluation only, never
    touched by the search.
  - Inner GroupKFold (N_INNER_SPLITS), built fresh from each outer fold's
    training groups: hyperparameter search happens here.
Searching hyperparameters on the same folds used to report final metrics
would leak information into the result — same failure mode flagged for the
near-perfect baseline AUROC.

Group key ("cv_group") matches train_baseline.py: subject-based for
COPD/Control, unique per-sample for Smokers/Air (no reliable subject
linkage — Phase 1 decision).

Scoring metric for the search: macro recall (sensitivity) — matches the
project's stated evaluation priority (clinical sensitivity over raw
accuracy).

After nested CV reports the honest estimate, a FINAL search is run on the
full train split (inner GroupKFold only, no outer holdout — there's nothing
left to hold out from) to pick the hyperparameters actually used going
forward, and refits each model on the full train split with those params.

Inputs (data/interim/, relative to project root):
  feature_train_selected.parquet

Outputs:
  data/processed/tuning_nested_cv_results.csv   — outer-fold metrics per model
  data/processed/tuning_best_params_per_fold.csv — best params chosen per outer fold
                                                     (check stability across folds)
  data/processed/tuned_hyperparameters.json      — final chosen params per model
  models/<model>_tuned.joblib                    — refit on full train split
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier
import joblib

# cosmetic only: sklearn 1.8 warns that `penalty` is deprecated in favor of
# `l1_ratio` alone, but still honors it correctly — safe to silence.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # src/tune_models.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
for d in (PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3
RANDOM_STATE = 42
CLASSES = ["air", "control", "copd", "smokers"]  # sorted, fixed order

ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]

SEARCH_SCORING = "recall_macro"  # matches project's stated sensitivity-first priority


def build_cv_groups(df: pd.DataFrame) -> np.ndarray:
    groups = []
    for _, row in df.iterrows():
        if row["group"] in ("copd", "control") and pd.notna(row["subject_id"]):
            groups.append(f"{row['group']}_subj{int(row['subject_id'])}")
        else:
            groups.append(f"{row['group']}_sample{int(row['sample_id'])}")
    return np.array(groups)


def get_search_spaces():
    """(pipeline, param_distributions, search_type) per model."""
    xgb_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE)),
    ])
    xgb_params = {
        "clf__max_depth": [2, 3, 4],
        "clf__n_estimators": [50, 100, 150],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
    }

    logreg_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=5000, random_state=RANDOM_STATE,
        )),
    ])
    logreg_params = {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    }

    return {
        "xgboost": (xgb_pipeline, xgb_params, "random"),
        "logistic_regression": (logreg_pipeline, logreg_params, "grid"),
    }


def make_search(pipeline, params, search_type, cv_splits):
    if search_type == "random":
        return RandomizedSearchCV(
            pipeline, params, n_iter=25, scoring=SEARCH_SCORING,
            cv=cv_splits, random_state=RANDOM_STATE, n_jobs=-1,
        )
    return GridSearchCV(pipeline, params, scoring=SEARCH_SCORING, cv=cv_splits, n_jobs=-1)


def inner_splits(X_outer, y_outer, groups_outer, n_splits):
    n_splits = min(n_splits, len(set(groups_outer)))
    return list(GroupKFold(n_splits=n_splits).split(X_outer, y_outer, groups=groups_outer))


def evaluate(y_true, y_pred, y_proba):
    sensitivity = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        y_bin = label_binarize(y_true, classes=range(len(CLASSES)))
        present = y_bin.sum(axis=0) > 0
        auroc = roc_auc_score(y_bin[:, present], y_proba[:, present], average="macro", multi_class="ovr")
    except ValueError:
        auroc = np.nan
    return sensitivity, macro_f1, auroc


def nested_cv(X, y_enc, groups, model_name, pipeline, params, search_type):
    outer = GroupKFold(n_splits=N_OUTER_SPLITS)
    fold_rows = []
    best_params_rows = []

    for fold_i, (train_idx, val_idx) in enumerate(outer.split(X, y_enc, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]
        groups_train = groups[train_idx]

        cv_splits = inner_splits(X_train, y_train, groups_train, N_INNER_SPLITS)
        search = make_search(pipeline, params, search_type, cv_splits)
        search.fit(X_train, y_train)

        pred = search.predict(X_val)
        proba = search.predict_proba(X_val)
        sensitivity, macro_f1, auroc = evaluate(y_val, pred, proba)

        fold_rows.append({
            "model": model_name, "fold": fold_i, "n_val": len(val_idx),
            "sensitivity_macro": sensitivity, "macro_f1": macro_f1, "auroc_macro_ovr": auroc,
        })
        best_params_rows.append({"model": model_name, "fold": fold_i, **search.best_params_})

        log.info(f"[{model_name}] outer fold {fold_i}: n_val={len(val_idx)} "
                 f"sensitivity={sensitivity:.3f} macro_f1={macro_f1:.3f} auroc={auroc:.3f} "
                 f"best_params={search.best_params_}")

    return fold_rows, best_params_rows


def final_tune_on_full_train(X, y_enc, groups, model_name, pipeline, params, search_type):
    cv_splits = inner_splits(X, y_enc, groups, N_INNER_SPLITS)
    search = make_search(pipeline, params, search_type, cv_splits)
    search.fit(X, y_enc)
    log.info(f"[{model_name}] final params chosen on full train split: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def main():
    train_path = INTERIM_DIR / "feature_train_selected.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run select_features.py first.")

    df = pd.read_parquet(train_path)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    X = df[feature_cols]
    y = df["group"]
    y_enc = y.map({cls: i for i, cls in enumerate(CLASSES)}).to_numpy()
    groups = build_cv_groups(df)

    log.info(f"Train samples: {len(df)}, features: {len(feature_cols)}, "
             f"unique CV groups: {len(set(groups))}\n")

    all_fold_rows = []
    all_best_params_rows = []
    final_params_by_model = {}

    for model_name, (pipeline, params, search_type) in get_search_spaces().items():
        log.info(f"=== {model_name}: nested CV (honest performance estimate) ===")
        fold_rows, best_params_rows = nested_cv(X, y_enc, groups, model_name, pipeline, params, search_type)
        all_fold_rows.extend(fold_rows)
        all_best_params_rows.extend(best_params_rows)

        log.info(f"\n=== {model_name}: final search on full train split ===")
        final_model, final_params = final_tune_on_full_train(
            X, y_enc, groups, model_name, pipeline, params, search_type
        )
        final_params_by_model[model_name] = final_params
        joblib.dump(final_model, MODELS_DIR / f"{model_name}_tuned.joblib")
        log.info("")

    nested_results = pd.DataFrame(all_fold_rows)
    nested_results.to_csv(PROCESSED_DIR / "tuning_nested_cv_results.csv", index=False)

    best_params_df = pd.DataFrame(all_best_params_rows)
    best_params_df.to_csv(PROCESSED_DIR / "tuning_best_params_per_fold.csv", index=False)

    with open(PROCESSED_DIR / "tuned_hyperparameters.json", "w") as f:
        json.dump(final_params_by_model, f, indent=2)

    summary = (
        nested_results.groupby("model")[["sensitivity_macro", "macro_f1", "auroc_macro_ovr"]]
        .agg(["mean", "std"])
    )
    log.info("=== Nested CV Summary (mean +/- std across outer folds) ===")
    log.info(summary.to_string())
    log.info(f"\nCompare against data/processed/cv_summary.csv (fixed-default baseline) "
              "to see whether tuning actually helped.")
    log.info("Check tuning_best_params_per_fold.csv for stability — if best params vary "
              "wildly across outer folds, the search is likely overfitting to fold noise "
              "at this sample size, not finding a robust setting.")
    log.info(f"\nSaved tuning_nested_cv_results.csv, tuning_best_params_per_fold.csv, "
              f"tuned_hyperparameters.json, and *_tuned.joblib models.")


if __name__ == "__main__":
    main()