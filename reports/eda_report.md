# EDA Report (train split only)

Loaded `data\interim\feature_train.parquet`: 68 samples x 104 columns.

## A. Data Quality

**Samples vs. unique subjects per group (train split):**

| group | samples | unique subjects |
|---|---|---|
| copd | 32 | 16 |
| control | 16 | 8 |
| smokers | 12 | n/a (no subject linkage) |
| air | 8 | n/a (no subject linkage) |

**Fully duplicate feature rows (excluding id columns):** 0

**Missing values (train split):**

| column | n missing |
|---|---|
| subject_id | 20 |
| replicate | 20 |
| gender | 20 |
| age | 20 |
| MQ3_rise_rate | 1 |
| SP3_decay_rate | 1 |
| severity_level_est | 36 |
| main_cause | 36 |

*Expected: subject_id/replicate/gender/age null for smokers+air (no subject linkage); severity_level_est/main_cause null outside COPD.*

**Near-zero-variance feature columns (train split):** ['TGS800_decay_rate', 'TGS800_rise_rate', 'TGS813_decay_rate']

Full descriptive stats table written to `reports\feature_describe_train.csv`.

## B. Raw Time-Series Sanity Plots

- `copd`: `reports\figures\raw_curves_copd.png`
- `control`: `reports\figures\raw_curves_control.png`
- `smokers`: `reports\figures\raw_curves_smokers.png`
- `air`: `reports\figures\raw_curves_air.png`

## C. Replicate Consistency (COPD / Control)

- `copd`: `reports\figures\replicate_consistency_copd.png` (points near the diagonal = consistent replicates)
- `control`: `reports\figures\replicate_consistency_control.png` (points near the diagonal = consistent replicates)

## D. Univariate Distributions (core features, by group)

- All sensors x core features: `reports\figures\core_feature_boxplots.png`

## E. Feature Correlation Heatmap

- `reports\figures\feature_correlation_heatmap.png`

**265 feature pairs with |correlation| > 0.9** (candidates for redundancy reduction before modeling):

| feature 1 | feature 2 | corr |
|---|---|---|
| SP3_peak_value | SP3_max | 1.00 |
| TGS800_peak_value | TGS800_max | 1.00 |
| TGS813_peak_value | TGS813_max | 1.00 |
| TGS800_r0 | TGS800_min | 1.00 |
| TGS822_peak_value | TGS822_max | 1.00 |
| TGS822_r0 | TGS822_min | 1.00 |
| SP3_r0 | SP3_min | 1.00 |
| MQ135_peak_value | MQ135_max | 1.00 |
| MQ137_peak_value | MQ137_max | 1.00 |
| MQ137_peak_amplitude | MQ138_peak_amplitude | 1.00 |
| MQ138_peak_value | MQ138_max | 0.99 |
| TGS813_r0 | TGS813_min | 0.99 |
| MQ137_std | MQ138_std | 0.99 |
| MQ137_r0 | MQ137_min | 0.99 |
| MQ135_r0 | MQ135_min | 0.99 |
| MQ137_auc | MQ138_auc | 0.99 |
| MQ138_r0 | MQ138_min | 0.99 |
| TGS813_decay_rate | TGS813_std | 0.99 |
| SP3_peak_amplitude | SP3_std | 0.99 |
| MQ137_min | MQ138_min | 0.99 |
| ... | ... | (245 more) |

## F. PCA Projection (exploratory)

*Median-imputed, standardized. Train split only. For a quick visual read of class separability — not a modeling result.*

- `reports\figures\pca_projection.png` (PC1+PC2 explain 75.1% of variance)

## G. Demographic Confounding Checks

- `reports\figures\age_by_group.png`

**Gender counts by group:**

| group   |   F |   M |
|:--------|----:|----:|
| control |   6 |  10 |
| copd    |  16 |  16 |

- `reports\figures\feature_vs_age.png`

*If a feature's group separation tracks age rather than diverging within age-matched subjects, treat it as a possible age confound rather than a clean biomarker.*

## H. Group Comparison (Kruskal-Wallis, screening only)

*Small n per group — treat p-values as a screening signal, not confirmatory evidence.*

| feature | H-statistic | p-value |
|---|---|---|
| SP3_baseline_ratio | 53.52 | 0.0000 |
| SP3_peak_amplitude | 40.50 | 0.0000 |
| SP3_auc | 48.53 | 0.0000 |
| SP3_decay_rate | 45.50 | 0.0000 |
| MQ3_baseline_ratio | 40.47 | 0.0000 |
| MQ3_peak_amplitude | 44.46 | 0.0000 |
| MQ3_auc | 50.39 | 0.0000 |
| MQ3_decay_rate | 41.84 | 0.0000 |
| TGS822_baseline_ratio | 52.37 | 0.0000 |
| TGS822_peak_amplitude | 43.92 | 0.0000 |
| TGS822_auc | 46.47 | 0.0000 |
| TGS822_decay_rate | 45.19 | 0.0000 |
| MQ138_baseline_ratio | 42.16 | 0.0000 |
| MQ138_peak_amplitude | 50.79 | 0.0000 |
| MQ138_auc | 54.88 | 0.0000 |
| MQ138_decay_rate | 44.60 | 0.0000 |
| MQ137_baseline_ratio | 42.39 | 0.0000 |
| MQ137_peak_amplitude | 49.85 | 0.0000 |
| MQ137_auc | 55.56 | 0.0000 |
| MQ137_decay_rate | 37.29 | 0.0000 |
| TGS813_baseline_ratio | 46.14 | 0.0000 |
| TGS813_peak_amplitude | 42.45 | 0.0000 |
| TGS813_auc | 52.10 | 0.0000 |
| TGS813_decay_rate | 44.71 | 0.0000 |
| TGS800_baseline_ratio | 55.22 | 0.0000 |
| TGS800_peak_amplitude | 36.85 | 0.0000 |
| TGS800_auc | 41.80 | 0.0000 |
| TGS800_decay_rate | 39.16 | 0.0000 |
| MQ135_baseline_ratio | 42.41 | 0.0000 |
| MQ135_peak_amplitude | 37.79 | 0.0000 |
| MQ135_auc | 40.52 | 0.0000 |
| MQ135_decay_rate | 40.50 | 0.0000 |
