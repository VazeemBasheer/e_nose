"""
train_baseline.py
==================
Trains and evaluates 3 baseline classifiers on the group label
(copd / control / smokers / air) using group-aware cross-validation on the
training split only (data/interim/feature_train_selected.parquet). The held-
out test split is NOT touched here — this script is for model selection.

Models (fixed, reasonable defaults — tune later once we know which is worth it):
  1. Logistic Regression (elastic net, multinomial) — interpretable, cheap to
     deploy on an MCU as a dot-product + softmax.
  2. XGBoost (shallow, regularized) — captures nonlinear sensor interactions.
  3. Linear-kernel SVM — strong classical performer at small-n/high-p tabular
     problems; linear kernel keeps edge export cheap (no kernel trick to bake
     into fixed-size C code).

Cross-validation:
  GroupKFold, N_SPLITS folds. Group key ("cv_group") is subject-based for
  COPD/Control (so a subject's 2 replicate measurements never split across
  folds) and a unique per-sample key for Smokers/Air (no reliable subject
  linkage — see Phase 1 notes), so those samples fold independently.
  NOTE: GroupKFold here is not stratified by class — with uneven group sizes
  per class, fold class balance can vary. Acceptable for a first baseline
  pass; revisit if fold-to-fold metric variance looks driven by this.

Metrics per fold: macro recall (sensitivity), macro F1, macro AUROC (OvR).
Missing values are median-imputed and features standardized INSIDE each
fold's pipeline (fit on that fold's train portion only) to avoid leakage.

Inputs (data/interim/, relative to project root):
  feature_train_selected.parquet

Outputs:
  data/processed/cv_results.csv   — per-fold, per-model metrics
  data/processed/cv_summary.csv   — mean +/- std per model, across folds
  reports/figures/confusion_matrix_<model>.png   — from out-of-fold predictions
  reports/figures/roc_curves_<model>.png         — one-vs-rest per class
  reports/figures/xgboost_feature_importance.png
  models/<model>.joblib           — each model refit on the FULL train split
                                     (for later evaluation against the test
                                     split, once model selection is done)
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent  # src/train_baseline.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
for d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42
CLASSES = ["air", "control", "copd", "smokers"]  # sorted, fixed order for reproducible metrics/plots

ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]


def build_cv_groups(df: pd.DataFrame) -> np.ndarray:
    """Subject-based group key for COPD/Control; unique per-sample key for
    Smokers/Air (no reliable subject linkage — Phase 1 decision)."""
    groups = []
    for _, row in df.iterrows():
        if row["group"] in ("copd", "control") and pd.notna(row["subject_id"]):
            groups.append(f"{row['group']}_subj{int(row['subject_id'])}")
        else:
            groups.append(f"{row['group']}_sample{int(row['sample_id'])}")
    return np.array(groups)


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", l1_ratio=0.5, C=1.0, solver="saga",
                max_iter=5000, random_state=RANDOM_STATE,
            )),
        ]),
        "xgboost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                eval_metric="mlogloss", random_state=RANDOM_STATE,
            )),
        ]),
        "linear_svm": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def run_cv(X: pd.DataFrame, y_enc: np.ndarray, groups: np.ndarray, model_name: str, pipeline: Pipeline):
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    oof_pred = np.full(len(y_enc), -1)
    oof_proba = np.zeros((len(y_enc), len(CLASSES)))

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y_enc, groups=groups)):
        pipe = pipeline
        pipe.fit(X.iloc[train_idx], y_enc[train_idx])
        pred = pipe.predict(X.iloc[val_idx])
        proba = pipe.predict_proba(X.iloc[val_idx])

        oof_pred[val_idx] = pred
        oof_proba[val_idx] = proba

        sensitivity = recall_score(y_enc[val_idx], pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_enc[val_idx], pred, average="macro", zero_division=0)
        try:
            y_val_bin = label_binarize(y_enc[val_idx], classes=range(len(CLASSES)))
            present = y_val_bin.sum(axis=0) > 0
            auroc = roc_auc_score(y_val_bin[:, present], proba[:, present], average="macro", multi_class="ovr")
        except ValueError:
            auroc = np.nan

        fold_rows.append({
            "model": model_name, "fold": fold_i, "n_val": len(val_idx),
            "sensitivity_macro": sensitivity, "macro_f1": macro_f1, "auroc_macro_ovr": auroc,
        })
        log.info(f"[{model_name}] fold {fold_i}: n_val={len(val_idx)} "
                 f"sensitivity={sensitivity:.3f} macro_f1={macro_f1:.3f} auroc={auroc:.3f}")

    return fold_rows, oof_pred, oof_proba


def plot_confusion(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} — out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, y_proba, model_name):
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
    ax.set_title(f"{model_name} — one-vs-rest ROC (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_curves_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_importance(pipe: Pipeline, feature_names: list):
    clf = pipe.named_steps["clf"]
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, hue=importances.index,
                palette="viridis", legend=False)
    ax.set_title("XGBoost feature importance (top 20, full train refit)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xgboost_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
             f"unique CV groups: {len(set(groups))}, folds: {N_SPLITS}")
    log.info(f"Class distribution: {y.value_counts().to_dict()}\n")

    all_fold_rows = []
    for model_name, pipeline in get_models().items():
        log.info(f"=== {model_name} ===")
        fold_rows, oof_pred, oof_proba = run_cv(X, y_enc, groups, model_name, pipeline)
        all_fold_rows.extend(fold_rows)

        plot_confusion(y_enc, oof_pred, model_name)
        plot_roc(y_enc, oof_proba, model_name)

        # refit on the FULL train split for later test-set evaluation
        final_pipeline = get_models()[model_name]
        final_pipeline.fit(X, y_enc)
        joblib.dump(final_pipeline, MODELS_DIR / f"{model_name}.joblib")

        if model_name == "xgboost":
            plot_xgb_importance(final_pipeline, feature_cols)

        log.info("")

    cv_results = pd.DataFrame(all_fold_rows)
    cv_results.to_csv(PROCESSED_DIR / "cv_results.csv", index=False)

    cv_summary = (
        cv_results.groupby("model")[["sensitivity_macro", "macro_f1", "auroc_macro_ovr"]]
        .agg(["mean", "std"])
    )
    cv_summary.to_csv(PROCESSED_DIR / "cv_summary.csv")

    log.info("=== CV Summary (mean +/- std across folds) ===")
    log.info(cv_summary.to_string())
    log.info(f"\nSaved cv_results.csv, cv_summary.csv, confusion/ROC figures, "
             f"and {len(get_models())} fitted models to models/")


if __name__ == "__main__":
    main()