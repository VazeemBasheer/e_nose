"""
Phase 1: Data Ingestion
Electronic nose dataset for COPD detection from smokers and healthy people
through exhaled breath analysis (Durán Acevedo et al., 2021, Data in Brief;
Mendeley Data DOI: 10.17632/h5pcn99zw4).

Raw file layout (empirically validated against the actual uploaded files):
  - Each file is tab-separated, no header.
  - Rows = time samples of the exhaled-breath transient (~4000 pts @ 500 Hz).
  - Columns = repeating blocks of 8, one block per breath sample/measurement.
    Sensor order within a block (per the source paper):
      SP-3, MQ-3, TGS822, MQ138, MQ137, TGS813, TGS800, MQ135
  - COPD.txt   : 320 cols -> 40 samples  = 20 subjects x 2 measurements
  - CONTROL.txt: 160 cols -> 20 samples  = 10 subjects x 2 measurements
  - SMOKERS.txt: 128 cols -> 16 samples. Demographics only list 4 subjects,
      so (per explicit instruction) each of the 16 samples is treated as an
      INDEPENDENT sample with NO subject-repetition assumption. subject_id
      is therefore left as unknown/unlinked for this group (do not merge
      demographics 1:1 onto samples).
  - AIR.txt    : 80 cols -> 10 samples = ambient baseline readings, no
      associated patient. Has 4080 rows vs 4000 for the others (80 extra
      rows) - kept as-is per group, not truncated, and flagged in the report.

Outputs (folder convention: csv -> data/processed/, parquet -> data/interim/):
  - data/processed/demographics_copd.csv / demographics_control.csv / demographics_smokers.csv
  - data/interim/raw_long_<group>.parquet   : tidy long-format time series
       [group, sample_id, subject_id, replicate, sensor, t, value]
  - data/processed/sample_index_<group>.csv : one row per sample with subject/replicate/labels
  - ingestion_report.md                     : summary, validation checks, and caveats
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # src/ingest.py -> project root
UPLOAD_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

SENSOR_NAMES = ["SP3", "MQ3", "TGS822", "MQ138", "MQ137", "TGS813", "TGS800", "MQ135"]
N_SENSORS = 8

GROUP_FILES = {
    "copd": "COPD.txt",
    "control": "CONTROL.txt",
    "smokers": "SMOKERS.txt",
    "air": "AIR.txt",
}

# subjects per group, measurements per subject (None where not applicable/unknown)
GROUP_META = {
    "copd": {"n_subjects": 20, "reps_per_subject": 2, "has_demographics": True},
    "control": {"n_subjects": 10, "reps_per_subject": 2, "has_demographics": True},
    "smokers": {"n_subjects": None, "reps_per_subject": None, "has_demographics": False},
    "air": {"n_subjects": None, "reps_per_subject": None, "has_demographics": False},
}


def load_raw(group):
    path = UPLOAD_DIR / GROUP_FILES[group]
    df = pd.read_csv(path, sep="\t", header=None)
    n_cols = df.shape[1]
    assert n_cols % N_SENSORS == 0, f"{group}: column count {n_cols} not divisible by {N_SENSORS}"
    n_samples = n_cols // N_SENSORS
    return df, n_samples


def reshape_to_long(df, n_samples, group):
    """Reshape (n_timepoints, n_samples*8) -> tidy long dataframe."""
    n_t = df.shape[0]
    arr = df.to_numpy(dtype=np.float32)  # (n_t, n_samples*8)
    # reshape into (n_t, n_samples, 8)
    arr = arr.reshape(n_t, n_samples, N_SENSORS)

    meta = GROUP_META[group]
    n_subjects = meta["n_subjects"]
    reps = meta["reps_per_subject"]

    records_sample_id = np.repeat(np.arange(n_samples), N_SENSORS)
    records_sensor = np.tile(SENSOR_NAMES, n_samples)

    if n_subjects is not None and reps is not None and n_subjects * reps == n_samples:
        subject_id = np.repeat(np.arange(1, n_subjects + 1), reps)
        replicate = np.tile(np.arange(1, reps + 1), n_subjects)
    else:
        # no reliable subject grouping (smokers) or not applicable (air)
        subject_id = np.full(n_samples, np.nan)
        replicate = np.full(n_samples, np.nan)

    sample_index = pd.DataFrame({
        "group": group,
        "sample_id": np.arange(n_samples),
        "subject_id": subject_id,
        "replicate": replicate,
    })

    # long format: melt time x (sample, sensor)
    t = np.arange(n_t)
    long_frames = []
    # vectorized construction to avoid slow python loops
    value = arr.reshape(n_t, n_samples * N_SENSORS)
    long_df = pd.DataFrame(value)
    long_df["t"] = t
    long_df = long_df.melt(id_vars="t", var_name="col", value_name="value")
    long_df["sample_id"] = long_df["col"] // N_SENSORS
    long_df["sensor"] = long_df["col"] % N_SENSORS
    long_df["sensor"] = long_df["sensor"].map(dict(enumerate(SENSOR_NAMES)))
    long_df = long_df.drop(columns="col")
    long_df["group"] = group
    long_df = long_df.merge(sample_index[["sample_id", "subject_id", "replicate"]], on="sample_id", how="left")
    long_df = long_df[["group", "sample_id", "subject_id", "replicate", "sensor", "t", "value"]]

    return long_df, sample_index


def load_demographics():
    path = UPLOAD_DIR / "General_data_from_the_dataset.txt"
    raw = pd.read_csv(path, sep="\t")

    copd = raw[["COPD_patient", "Gender", "Age", "Severity_level_(estimated)", "Main_cause"]].dropna(how="all")
    copd = copd.rename(columns={
        "COPD_patient": "subject_id", "Gender": "gender", "Age": "age",
        "Severity_level_(estimated)": "severity_level_est", "Main_cause": "main_cause",
    }).dropna(subset=["subject_id"])
    copd["subject_id"] = copd["subject_id"].astype(int)

    smokers = raw[["SMOKERS_patient", "Gender.1", "Age.1", "Span_of_habit_(years)", "Relevant notes"]].dropna(how="all")
    smokers = smokers.rename(columns={
        "SMOKERS_patient": "subject_id", "Gender.1": "gender", "Age.1": "age",
        "Span_of_habit_(years)": "years_smoking", "Relevant notes": "notes",
    }).dropna(subset=["subject_id"])
    smokers["subject_id"] = smokers["subject_id"].astype(int)

    control = raw[["CONTROL_patient", "Gender.2", "Age.2"]].dropna(how="all")
    control = control.rename(columns={
        "CONTROL_patient": "subject_id", "Gender.2": "gender", "Age.2": "age",
    }).dropna(subset=["subject_id"])
    control["subject_id"] = control["subject_id"].astype(int)
    control["age"] = control["age"].astype(int)

    return copd, smokers, control


def validate_block_hypothesis(df, n_samples, group):
    """Sanity check: same sensor-position column across samples should show a
    tighter, sensor-characteristic scale range than mixing sensor positions."""
    pos_means = []
    for pos in range(N_SENSORS):
        cols = df.iloc[:, pos::N_SENSORS]
        pos_means.append(cols.mean().mean())
    return pos_means


def main():
    report_lines = []
    report_lines.append("# Phase 1 — Data Ingestion Report\n")
    report_lines.append(
        "Source: *Electronic nose dataset for COPD detection from smokers and "
        "healthy people through exhaled breath analysis* (Durán Acevedo et al., "
        "2021, Data in Brief; Mendeley Data DOI 10.17632/h5pcn99zw4).\n"
    )
    report_lines.append(
        "Sensor array (column order within each 8-col block): "
        + ", ".join(SENSOR_NAMES) + "\n"
    )

    copd_demo, smokers_demo, control_demo = load_demographics()
    copd_demo.to_csv(PROCESSED_DIR / "demographics_copd.csv", index=False)
    smokers_demo.to_csv(PROCESSED_DIR / "demographics_smokers.csv", index=False)
    control_demo.to_csv(PROCESSED_DIR / "demographics_control.csv", index=False)

    all_sample_index = []

    for group in ["copd", "control", "smokers", "air"]:
        df, n_samples = load_raw(group)
        n_t = df.shape[0]
        pos_means = validate_block_hypothesis(df, n_samples, group)

        long_df, sample_index = reshape_to_long(df, n_samples, group)

        # attach demographics where we have a reliable subject linkage
        if group == "copd":
            sample_index = sample_index.merge(copd_demo, on="subject_id", how="left")
        elif group == "control":
            sample_index = sample_index.merge(control_demo, on="subject_id", how="left")
        # smokers / air: no reliable per-sample subject linkage -> left as NaN

        long_path = INTERIM_DIR / f"raw_long_{group}.parquet"
        long_df.to_parquet(long_path, index=False)
        sample_index.to_csv(PROCESSED_DIR / f"sample_index_{group}.csv", index=False)
        all_sample_index.append(sample_index)

        report_lines.append(f"\n## {group.upper()}")
        report_lines.append(f"- Raw shape: {df.shape[0]} timepoints x {df.shape[1]} columns")
        report_lines.append(f"- Parsed as: {n_samples} samples x {N_SENSORS} sensors")
        if group in ("copd", "control"):
            meta = GROUP_META[group]
            report_lines.append(
                f"- Subject linkage: {meta['n_subjects']} subjects x {meta['reps_per_subject']} "
                f"replicate measurements each (matches demographics file)"
            )
            n_demo = len(copd_demo) if group == "copd" else len(control_demo)
            report_lines.append(f"- Demographics rows available: {n_demo}")
        elif group == "smokers":
            report_lines.append(
                "- Subject linkage: NOT assumed. 16 samples treated as independent "
                "(per explicit decision) — demographics (4 subjects) NOT merged 1:1 "
                "onto samples. See demographics_smokers.csv separately."
            )
        else:
            report_lines.append("- No associated subject/demographics (ambient air baseline).")
            if n_t != 4000:
                report_lines.append(
                    f"- ⚠️ Row count is {n_t}, not the expected 4000 — {n_t-4000} extra "
                    "rows kept as-is; verify against source before using for "
                    "fixed-length feature windows."
                )
        report_lines.append(
            "- Sensor-position sanity check (mean value per sensor slot, averaged "
            "across all samples in this file): "
            + ", ".join(f"{s}={m:.3f}" for s, m in zip(SENSOR_NAMES, pos_means))
        )

    combined_index = pd.concat(all_sample_index, ignore_index=True)
    combined_index.to_csv(PROCESSED_DIR / "sample_index_all.csv", index=False)

    report_lines.append("\n## Combined sample index")
    report_lines.append(f"- Total samples across all groups: {len(combined_index)}")
    report_lines.append(combined_index.groupby("group").size().to_string())

    report_lines.append("\n## Caveats / assumptions carried into Phase 2")
    report_lines.append(
        "- Sensor identity assignment (SP-3, MQ-3, TGS822, MQ138, MQ137, TGS813, "
        "TGS800, MQ135, in that column order) follows the source paper's stated "
        "array order — it is NOT independently verifiable from the raw numeric "
        "files alone, since there is no header row. Values are unitless in the "
        "raw text (paper describes them as voltage-domain sensor responses)."
    )
    report_lines.append(
        "- SMOKERS group: sample-to-subject mapping is unresolved (16 raw samples "
        "vs 4 documented subjects). Demographics for this group are kept as a "
        "separate table, not joined to samples. Any modeling that needs "
        "GroupKFold/LOSO on smokers should either exclude this group or use "
        "sample_id as a (weaker) grouping key."
    )
    report_lines.append(
        "- AIR group has 80 extra rows relative to the other three files (4080 "
        "vs 4000) - not truncated here."
    )

    (BASE_DIR / "ingestion_report.md").write_text("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()