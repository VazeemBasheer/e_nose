"""
clean.py
=============
End-to-end preprocessing for the e-nose COPD / Smokers / Control / Air breath
dataset (Durán Acevedo et al., 2021, Data in Brief; Mendeley Data DOI
10.17632/h5pcn99zw4).

Pipeline:
  1. Load raw tab-separated sensor files + demographics table.
  2. Clean: fix known row-count mismatch, check for duplicate/constant
     columns, validate value ranges, standardize demographic fields.
  3. Reshape each file's repeating 8-column sensor blocks into tidy
     long-format time series and attach subject/replicate/group identity.
  4. Extract per-sample, per-sensor engineered features (baseline ratio,
     peak amplitude, AUC, rise/decay rate, summary stats).
  5. Merge demographics + group label onto the feature table.
  6. Write a single analysis-ready file: feature.parquet

Run:
    python clean.py
Inputs expected in data/raw/ (relative to project root):
    AIR.txt, CONTROL.txt, COPD.txt, SMOKERS.txt,
    General_data_from_the_dataset.txt
Output (folder convention: all parquet -> data/interim/):
    data/interim/feature.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # src/preprocess.py -> project root
INPUT_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_NAMES = ["SP3", "MQ3", "TGS822", "MQ138", "MQ137", "TGS813", "TGS800", "MQ135"]
N_SENSORS = 8
EXPECTED_N_TIMEPOINTS = 4000
BASELINE_FRAC = 0.05  # first 5% of timepoints define the R0 baseline

GROUP_FILES = {
    "copd": "COPD.txt",
    "control": "CONTROL.txt",
    "smokers": "SMOKERS.txt",
    "air": "AIR.txt",
}

# subjects per group / replicate measurements per subject, where reliably known
GROUP_META = {
    "copd": {"n_subjects": 20, "reps_per_subject": 2},
    "control": {"n_subjects": 10, "reps_per_subject": 2},
    "smokers": {"n_subjects": None, "reps_per_subject": None},  # unresolved, see README
    "air": {"n_subjects": None, "reps_per_subject": None},      # no subjects (ambient)
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_raw_sensor_file(group: str) -> pd.DataFrame:
    path = INPUT_DIR / GROUP_FILES[group]
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] % N_SENSORS != 0:
        raise ValueError(
            f"{group}: {df.shape[1]} columns not divisible by {N_SENSORS} sensors"
        )
    return df


def load_demographics():
    path = INPUT_DIR / "General_data_from_the_dataset.txt"
    raw = pd.read_csv(path, sep="\t")

    copd = (
        raw[["COPD_patient", "Gender", "Age", "Severity_level_(estimated)", "Main_cause"]]
        .dropna(subset=["COPD_patient"])
        .rename(columns={
            "COPD_patient": "subject_id", "Gender": "gender", "Age": "age",
            "Severity_level_(estimated)": "severity_level_est", "Main_cause": "main_cause",
        })
    )
    copd["subject_id"] = copd["subject_id"].astype(int)

    smokers = (
        raw[["SMOKERS_patient", "Gender.1", "Age.1", "Span_of_habit_(years)", "Relevant notes"]]
        .dropna(subset=["SMOKERS_patient"])
        .rename(columns={
            "SMOKERS_patient": "subject_id", "Gender.1": "gender", "Age.1": "age",
            "Span_of_habit_(years)": "years_smoking", "Relevant notes": "notes",
        })
    )
    smokers["subject_id"] = smokers["subject_id"].astype(int)

    control = (
        raw[["CONTROL_patient", "Gender.2", "Age.2"]]
        .dropna(subset=["CONTROL_patient"])
        .rename(columns={"CONTROL_patient": "subject_id", "Gender.2": "gender", "Age.2": "age"})
    )
    control["subject_id"] = control["subject_id"].astype(int)
    control["age"] = control["age"].astype(int)

    for d in (copd, smokers, control):
        d["gender"] = d["gender"].astype(str).str.strip().str.upper()

    return copd, smokers, control


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_raw_sensor_file(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Row/column-level cleaning on one raw sensor file."""
    n_rows_before = df.shape[0]

    # known issue: AIR.txt has extra rows past the documented 4000 timepoints
    if n_rows_before > EXPECTED_N_TIMEPOINTS:
        log.info(
            f"[{group}] trimming {n_rows_before - EXPECTED_N_TIMEPOINTS} extra "
            f"row(s) beyond the expected {EXPECTED_N_TIMEPOINTS} timepoints"
        )
        df = df.iloc[:EXPECTED_N_TIMEPOINTS].reset_index(drop=True)
    elif n_rows_before < EXPECTED_N_TIMEPOINTS:
        log.warning(
            f"[{group}] only {n_rows_before} rows, fewer than the expected "
            f"{EXPECTED_N_TIMEPOINTS} - leaving as-is, downstream windows may "
            "need to handle variable length"
        )

    # missing values
    n_missing = df.isnull().sum().sum()
    if n_missing:
        log.warning(f"[{group}] {n_missing} missing values found -> forward/back fill")
        df = df.ffill().bfill()

    # duplicate columns (would indicate a copy/paste error in the raw export)
    dup_cols = df.T[df.T.duplicated()].T.columns.tolist()
    if dup_cols:
        log.warning(f"[{group}] {len(dup_cols)} duplicate column(s) found at positions {dup_cols}")

    # constant (dead-sensor) columns
    constant_cols = df.columns[df.nunique() <= 1].tolist()
    if constant_cols:
        log.warning(f"[{group}] {len(constant_cols)} constant column(s) at positions {constant_cols}")

    # implausible negative sensor readings (MOS sensor responses should be >= 0)
    n_negative = (df < 0).sum().sum()
    if n_negative:
        log.warning(f"[{group}] {n_negative} negative values found -> clipping to 0")
        df = df.clip(lower=0)

    return df


def clean_demographics(copd, smokers, control):
    for name, d in [("copd", copd), ("smokers", smokers), ("control", control)]:
        before = len(d)
        d.drop_duplicates(subset="subject_id", keep="first", inplace=True)
        if len(d) != before:
            log.warning(f"[{name}] dropped {before - len(d)} duplicate subject_id row(s)")
    return copd, smokers, control


# --------------------------------------------------------------------------
# 3. Reshape to long format
# --------------------------------------------------------------------------
def reshape_to_long(df: pd.DataFrame, group: str) -> pd.DataFrame:
    n_t, n_cols = df.shape
    n_samples = n_cols // N_SENSORS
    arr = df.to_numpy(dtype=np.float32).reshape(n_t, n_samples * N_SENSORS)

    meta = GROUP_META[group]
    n_subjects, reps = meta["n_subjects"], meta["reps_per_subject"]
    if n_subjects is not None and reps is not None and n_subjects * reps == n_samples:
        subject_id = np.repeat(np.arange(1, n_subjects + 1), reps)
        replicate = np.tile(np.arange(1, reps + 1), n_subjects)
    else:
        # unresolved subject linkage (smokers) or not applicable (air):
        # treat every sample as independent rather than guess a grouping
        subject_id = np.full(n_samples, np.nan)
        replicate = np.full(n_samples, np.nan)

    sample_meta = pd.DataFrame({
        "sample_id": np.arange(n_samples),
        "subject_id": subject_id,
        "replicate": replicate,
    })

    long_df = pd.DataFrame(arr)
    long_df["t"] = np.arange(n_t)
    long_df = long_df.melt(id_vars="t", var_name="col", value_name="value")
    long_df["sample_id"] = long_df["col"] // N_SENSORS
    long_df["sensor"] = (long_df["col"] % N_SENSORS).map(dict(enumerate(SENSOR_NAMES)))
    long_df = long_df.drop(columns="col").merge(sample_meta, on="sample_id", how="left")
    long_df["group"] = group

    return long_df[["group", "sample_id", "subject_id", "replicate", "sensor", "t", "value"]]


# --------------------------------------------------------------------------
# 4. Feature extraction
# --------------------------------------------------------------------------
def extract_features(long_df: pd.DataFrame) -> pd.DataFrame:
    n_t = long_df["t"].max() + 1
    n_baseline = max(1, int(n_t * BASELINE_FRAC))

    rows = []
    for (sample_id, sensor), g in long_df.groupby(["sample_id", "sensor"], sort=False):
        g = g.sort_values("t")
        v = g["value"].to_numpy()
        t = g["t"].to_numpy()

        r0 = v[:n_baseline].mean()
        dev = v - r0
        peak_idx = int(np.argmax(np.abs(dev)))
        peak_value = v[peak_idx]
        peak_amplitude = abs(dev[peak_idx])
        peak_t = t[peak_idx]

        baseline_ratio = peak_value / r0 if r0 != 0 else np.nan
        auc = np.trapezoid(dev, t)

        t_end, end_value = t[-1], v[-1]
        rise_rate = peak_amplitude / peak_t if peak_t > 0 else np.nan
        decay_span = t_end - peak_t
        decay_rate = abs(peak_value - end_value) / decay_span if decay_span > 0 else np.nan

        rows.append({
            "sample_id": sample_id, "sensor": sensor,
            "r0": r0, "peak_value": peak_value, "peak_amplitude": peak_amplitude,
            "peak_t": peak_t, "baseline_ratio": baseline_ratio, "auc": auc,
            "rise_rate": rise_rate, "decay_rate": decay_rate,
            "mean": v.mean(), "std": v.std(), "min": v.min(), "max": v.max(),
        })

    feat_long = pd.DataFrame(rows)
    feat_wide = feat_long.pivot(index="sample_id", columns="sensor")
    feat_wide.columns = [f"{sensor}_{feat}" for feat, sensor in feat_wide.columns]
    return feat_wide.reset_index()


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    copd_demo, smokers_demo, control_demo = load_demographics()
    copd_demo, smokers_demo, control_demo = clean_demographics(copd_demo, smokers_demo, control_demo)
    demo_by_group = {"copd": copd_demo, "control": control_demo}  # smokers/air: no reliable per-sample link

    all_feature_tables = []
    for group in ["copd", "control", "smokers", "air"]:
        log.info(f"\n=== {group.upper()} ===")
        raw = load_raw_sensor_file(group)
        clean = clean_raw_sensor_file(raw, group)
        long_df = reshape_to_long(clean, group)

        feats = extract_features(long_df)
        feats.insert(0, "group", group)

        sample_meta = long_df[["sample_id", "subject_id", "replicate"]].drop_duplicates()
        feats = feats.merge(sample_meta, on="sample_id", how="left")

        if group in demo_by_group:
            feats = feats.merge(demo_by_group[group], on="subject_id", how="left")

        log.info(f"[{group}] {feats.shape[0]} samples x {feats.shape[1]} columns")
        all_feature_tables.append(feats)

    features = pd.concat(all_feature_tables, ignore_index=True)

    id_cols = [c for c in ["group", "sample_id", "subject_id", "replicate", "gender", "age"] if c in features.columns]
    other_cols = [c for c in features.columns if c not in id_cols]
    features = features[id_cols + other_cols]

    out_path = INTERIM_DIR / "feature.parquet"
    features.to_parquet(out_path, index=False)

    log.info(f"\nSaved {out_path}  shape={features.shape}")
    log.info(f"Samples per group:\n{features.groupby('group').size().to_string()}")
    log.info(f"Remaining nulls (expected for smokers/air subject fields, and COPD-only columns):\n"
              f"{features.isnull().sum()[features.isnull().sum() > 0].to_string()}")


if __name__ == "__main__":
    main()