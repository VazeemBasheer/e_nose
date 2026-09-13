"""
eda.py
======
Exploratory data analysis — data quality checks and plots — run STRICTLY on
the training split (data/interim/feature_train.parquet), per the project
decision to keep EDA train-only and avoid leaking test-set patterns into
modeling decisions.

Covers:
  A. Data quality: shape/class balance (samples vs subjects), missingness,
     duplicate rows, near-zero-variance columns, descriptive stats.
  B. Raw time-series sanity plots: a few overlaid raw curves per sensor per
     group (restricted to train samples), to visually confirm the transient
     shape looks physically sane before trusting engineered features.
  C. Replicate consistency: for COPD/Control (which have paired
     measurements), replicate-1 vs replicate-2 scatter per core feature.
  D. Univariate distributions: boxplots of core features by group, per
     sensor.
  E. Correlation heatmap across all engineered features.
  F. PCA projection colored by group (exploratory only — median-imputed,
     standardized).
  G. Demographic confounding checks: age/gender by group, feature-vs-age
     scatter for the groups that have demographics (COPD, Control).
  H. Non-parametric group comparison tests (Kruskal-Wallis across all
     groups, per core feature) — small-n, so treat as a screening signal
     rather than confirmatory.

Inputs (relative to project root, run from src/):
  data/interim/feature_train.parquet
  data/interim/raw_long_<group>.parquet

Outputs:
  reports/eda_report.md
  reports/figures/*.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent  # src/eda.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["copd", "control", "smokers", "air"]
SENSOR_NAMES = ["SP3", "MQ3", "TGS822", "MQ138", "MQ137", "TGS813", "TGS800", "MQ135"]
CORE_FEATURES = ["baseline_ratio", "peak_amplitude", "auc", "decay_rate"]
ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]

sns.set_theme(style="whitegrid")
report = []  # list of markdown strings, joined and written at the end


def log(md_line: str = ""):
    report.append(md_line)
    print(md_line)


def save_fig(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(BASE_DIR)


# --------------------------------------------------------------------------
# A. Data quality
# --------------------------------------------------------------------------
def section_data_quality(df: pd.DataFrame):
    log("## A. Data Quality\n")

    counts = df.groupby("group").size()
    subj_counts = (
        df[df["subject_id"].notna()]
        .groupby("group")["subject_id"].nunique()
    )
    log("**Samples vs. unique subjects per group (train split):**\n")
    log("| group | samples | unique subjects |")
    log("|---|---|---|")
    for g in GROUPS:
        n_samp = counts.get(g, 0)
        n_subj = subj_counts.get(g, np.nan)
        n_subj_str = "n/a (no subject linkage)" if pd.isna(n_subj) else str(int(n_subj))
        log(f"| {g} | {n_samp} | {n_subj_str} |")
    log("")

    n_dupe_rows = df.duplicated(subset=[c for c in df.columns if c not in ID_COLS]).sum()
    log(f"**Fully duplicate feature rows (excluding id columns):** {n_dupe_rows}\n")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        log("**Missing values (train split):**\n")
        log("| column | n missing |")
        log("|---|---|")
        for col, n in missing.items():
            log(f"| {col} | {n} |")
        log("\n*Expected: subject_id/replicate/gender/age null for smokers+air "
            "(no subject linkage); severity_level_est/main_cause null outside COPD.*\n")
    else:
        log("**Missing values:** none.\n")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(
        ["sample_id", "subject_id", "replicate", "age"]
    )
    variances = df[numeric_cols].var()
    near_zero = variances[variances < 1e-8].index.tolist()
    log(f"**Near-zero-variance feature columns (train split):** "
        f"{near_zero if near_zero else 'none'}\n")

    desc = df[numeric_cols].describe().T[["mean", "std", "min", "max"]]
    desc_path = REPORTS_DIR / "feature_describe_train.csv"
    desc.to_csv(desc_path)
    log(f"Full descriptive stats table written to `{desc_path.relative_to(BASE_DIR)}`.\n")


# --------------------------------------------------------------------------
# B. Raw time-series sanity plots
# --------------------------------------------------------------------------
def section_raw_timeseries(feature_train: pd.DataFrame, n_examples: int = 4):
    log("## B. Raw Time-Series Sanity Plots\n")

    for group in GROUPS:
        raw_path = INTERIM_DIR / f"raw_long_{group}.parquet"
        if not raw_path.exists():
            log(f"*(skipped {group}: {raw_path.name} not found)*\n")
            continue

        train_sample_ids = feature_train.loc[feature_train["group"] == group, "sample_id"].unique()
        if len(train_sample_ids) == 0:
            continue

        raw = pd.read_parquet(raw_path)
        raw = raw[raw["sample_id"].isin(train_sample_ids)]

        example_ids = sorted(train_sample_ids)[:n_examples]

        fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
        for ax, sensor in zip(axes.flat, SENSOR_NAMES):
            for sid in example_ids:
                curve = raw[(raw["sample_id"] == sid) & (raw["sensor"] == sensor)].sort_values("t")
                ax.plot(curve["t"], curve["value"], alpha=0.8, linewidth=1, label=f"sample {sid}")
            ax.set_title(sensor, fontsize=10)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{group.upper()} — raw sensor curves (train samples, first {n_examples} shown)")
        fig.tight_layout()
        fig_path = save_fig(fig, f"raw_curves_{group}")
        log(f"- `{group}`: `{fig_path}`")
    log("")


# --------------------------------------------------------------------------
# C. Replicate consistency (COPD, Control only — the groups with pairing)
# --------------------------------------------------------------------------
def section_replicate_consistency(df: pd.DataFrame):
    log("## C. Replicate Consistency (COPD / Control)\n")

    for group in ["copd", "control"]:
        g = df[df["group"] == group]
        if g["subject_id"].isna().all():
            continue

        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        for ax, sensor in zip(axes.flat, SENSOR_NAMES):
            col = f"{sensor}_baseline_ratio"
            if col not in g.columns:
                continue
            pivoted = g.pivot(index="subject_id", columns="replicate", values=col)
            if 1.0 not in pivoted.columns or 2.0 not in pivoted.columns:
                continue
            ax.scatter(pivoted[1.0], pivoted[2.0], alpha=0.8)
            lims = [
                min(pivoted[1.0].min(), pivoted[2.0].min()),
                max(pivoted[1.0].max(), pivoted[2.0].max()),
            ]
            ax.plot(lims, lims, "--", color="gray", linewidth=1)
            ax.set_title(f"{sensor}_baseline_ratio", fontsize=9)
            ax.set_xlabel("replicate 1")
            ax.set_ylabel("replicate 2")
        fig.suptitle(f"{group.upper()} — replicate 1 vs 2 (baseline_ratio), train subjects only")
        fig.tight_layout()
        fig_path = save_fig(fig, f"replicate_consistency_{group}")
        log(f"- `{group}`: `{fig_path}` (points near the diagonal = consistent replicates)")
    log("")


# --------------------------------------------------------------------------
# D. Univariate distributions of core features, by group
# --------------------------------------------------------------------------
def section_univariate(df: pd.DataFrame):
    log("## D. Univariate Distributions (core features, by group)\n")

    fig, axes = plt.subplots(len(SENSOR_NAMES), len(CORE_FEATURES), figsize=(18, 28))
    for i, sensor in enumerate(SENSOR_NAMES):
        for j, feat in enumerate(CORE_FEATURES):
            col = f"{sensor}_{feat}"
            ax = axes[i, j]
            if col in df.columns:
                sns.boxplot(data=df, x="group", y=col, hue="group", ax=ax,
                            order=GROUPS, palette="Set2", legend=False)
                ax.set_title(col, fontsize=8)
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.axis("off")
    fig.tight_layout()
    fig_path = save_fig(fig, "core_feature_boxplots")
    log(f"- All sensors x core features: `{fig_path}`\n")


# --------------------------------------------------------------------------
# E. Correlation heatmap
# --------------------------------------------------------------------------
def section_correlation(df: pd.DataFrame):
    log("## E. Feature Correlation Heatmap\n")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["sample_id", "subject_id", "replicate", "age"]]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=True, yticklabels=True, ax=ax,
                cbar_kws={"shrink": 0.6})
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    fig_path = save_fig(fig, "feature_correlation_heatmap")
    log(f"- `{fig_path}`\n")

    high_corr = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .pipe(lambda s: s[s.abs() > 0.9])
        .sort_values(key=abs, ascending=False)
    )
    if len(high_corr):
        log(f"**{len(high_corr)} feature pairs with |correlation| > 0.9** "
            "(candidates for redundancy reduction before modeling):\n")
        log("| feature 1 | feature 2 | corr |")
        log("|---|---|---|")
        for (f1, f2), v in high_corr.head(20).items():
            log(f"| {f1} | {f2} | {v:.2f} |")
        if len(high_corr) > 20:
            log(f"| ... | ... | ({len(high_corr) - 20} more) |")
        log("")


# --------------------------------------------------------------------------
# F. PCA projection
# --------------------------------------------------------------------------
def section_pca(df: pd.DataFrame):
    log("## F. PCA Projection (exploratory)\n")
    log("*Median-imputed, standardized. Train split only. For a quick visual "
        "read of class separability — not a modeling result.*\n")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["sample_id", "subject_id", "replicate", "age"]]
    X = df[numeric_cols].to_numpy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for group, color in zip(GROUPS, sns.color_palette("Set2", len(GROUPS))):
        mask = df["group"] == group
        ax.scatter(coords[mask, 0], coords[mask, 1], label=group, alpha=0.8, color=color)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.legend()
    ax.set_title("PCA of engineered features, colored by group")
    fig_path = save_fig(fig, "pca_projection")
    log(f"- `{fig_path}` (PC1+PC2 explain {sum(explained)*100:.1f}% of variance)\n")


# --------------------------------------------------------------------------
# G. Demographic confounding
# --------------------------------------------------------------------------
def section_demographics(df: pd.DataFrame):
    log("## G. Demographic Confounding Checks\n")

    demo_df = df[df["age"].notna()]
    if len(demo_df):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=demo_df, x="group", y="age", hue="group", ax=ax,
                    palette="Set2", legend=False)
        ax.set_title("Age by group (COPD/Control only — no age linkage for smokers/air)")
        fig_path = save_fig(fig, "age_by_group")
        log(f"- `{fig_path}`\n")

        gender_counts = demo_df.groupby(["group", "gender"]).size().unstack(fill_value=0)
        log("**Gender counts by group:**\n")
        log(gender_counts.to_markdown())
        log("")

        fig, axes = plt.subplots(1, len(CORE_FEATURES), figsize=(20, 4))
        for ax, feat in zip(axes, CORE_FEATURES):
            col = f"SP3_{feat}"
            if col not in demo_df.columns:
                ax.axis("off")
                continue
            sns.scatterplot(data=demo_df, x="age", y=col, hue="group", ax=ax,
                             palette="Set2", legend=(ax is axes[0]))
            ax.set_title(col, fontsize=9)
        fig.suptitle("Feature vs. age, colored by group (SP3 sensor shown as example)")
        fig.tight_layout()
        fig_path = save_fig(fig, "feature_vs_age")
        log(f"- `{fig_path}`\n")
        log("*If a feature's group separation tracks age rather than diverging "
            "within age-matched subjects, treat it as a possible age confound "
            "rather than a clean biomarker.*\n")
    else:
        log("*No age data available in this split.*\n")


# --------------------------------------------------------------------------
# H. Non-parametric group comparison
# --------------------------------------------------------------------------
def section_group_tests(df: pd.DataFrame):
    log("## H. Group Comparison (Kruskal-Wallis, screening only)\n")
    log("*Small n per group — treat p-values as a screening signal, not "
        "confirmatory evidence.*\n")

    log("| feature | H-statistic | p-value |")
    log("|---|---|---|")
    for sensor in SENSOR_NAMES:
        for feat in CORE_FEATURES:
            col = f"{sensor}_{feat}"
            if col not in df.columns:
                continue
            groups_data = [df.loc[df["group"] == g, col].dropna() for g in GROUPS]
            groups_data = [g for g in groups_data if len(g) > 0]
            if len(groups_data) < 2:
                continue
            try:
                h_stat, p_val = stats.kruskal(*groups_data)
                log(f"| {col} | {h_stat:.2f} | {p_val:.4f} |")
            except ValueError:
                continue
    log("")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    feature_train_path = INTERIM_DIR / "feature_train.parquet"
    if not feature_train_path.exists():
        raise FileNotFoundError(
            f"{feature_train_path} not found — run ingest.py, preprocess.py, "
            "then split.py first."
        )
    df = pd.read_parquet(feature_train_path)

    log("# EDA Report (train split only)\n")
    log(f"Loaded `{feature_train_path.relative_to(BASE_DIR)}`: "
        f"{df.shape[0]} samples x {df.shape[1]} columns.\n")

    section_data_quality(df)
    section_raw_timeseries(df)
    section_replicate_consistency(df)
    section_univariate(df)
    section_correlation(df)
    section_pca(df)
    section_demographics(df)
    section_group_tests(df)

    report_path = REPORTS_DIR / "eda_report.md"
    report_path.write_text("\n".join(report))
    print(f"\nSaved {report_path}")


if __name__ == "__main__":
    main()