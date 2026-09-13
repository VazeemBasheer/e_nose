"""
app.py
======
Streamlit interface for the champion model. Predicts group membership
(COPD / Control / Smokers / Air) from the engineered e-nose sensor features
the model was actually trained on — NOT asthma phenotype (see project notes:
that was a different, never-built hypothetical from the original spec).

The model consumes the 50 engineered features selected by select_features.py
(baseline_ratio, auc, rise_rate, decay_rate, etc. per sensor) — not raw
sensor time-series directly. This app lets you either:
  - Enter feature values manually (grouped by sensor, pre-filled with
    train-set medians as a sane starting point), or
  - Auto-fill from a real train or held-out test sample, for demo purposes.

IMPORTANT CAVEATS surfaced in the UI itself, not just here:
  - This is a research prototype, not a diagnostic device.
  - The champion model's near-perfect test score (see evaluate_test.py) is
    on only 18 held-out samples — treat any single prediction's confidence
    with the same skepticism, not as a clinical-grade probability.
  - Predicting "air" for real breath input, or vice versa, would indicate
    something wrong with how features were computed upstream, not a subtle
    model error — the air/breath distinction should be the easiest part of
    this problem by a wide margin.

Run:
    streamlit run app.py

Inputs (relative to project root):
  models/champion_model.joblib
  data/processed/champion_model.json           (for display only)
  data/interim/feature_train_selected.parquet   (feature schema + medians)
  data/interim/feature_test_selected.parquet    (optional, for the "load a
                                                   real test sample" demo button)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent  # src/app.py -> project root
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

CLASSES = ["air", "control", "copd", "smokers"]  # must match training encoding order
ID_COLS = ["group", "sample_id", "subject_id", "replicate", "gender", "age",
           "severity_level_est", "main_cause", "years_smoking", "notes"]
SENSOR_NAMES = ["SP3", "MQ3", "TGS822", "MQ138", "MQ137", "TGS813", "TGS800", "MQ135"]


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "champion_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_champion_info():
    path = PROCESSED_DIR / "champion_model.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_reference_data():
    train_path = INTERIM_DIR / "feature_train_selected.parquet"
    test_path = INTERIM_DIR / "feature_test_selected.parquet"
    train_df = pd.read_parquet(train_path) if train_path.exists() else None
    test_df = pd.read_parquet(test_path) if test_path.exists() else None
    return train_df, test_df


def feature_columns_from(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ID_COLS]


def group_features_by_sensor(feature_cols: list) -> dict:
    grouped = {sensor: [] for sensor in SENSOR_NAMES}
    other = []
    for col in feature_cols:
        matched = False
        for sensor in SENSOR_NAMES:
            if col.startswith(f"{sensor}_"):
                grouped[sensor].append(col)
                matched = True
                break
        if not matched:
            other.append(col)
    if other:
        grouped["other"] = other
    return {k: v for k, v in grouped.items() if v}


def main():
    st.set_page_config(page_title="E-nose group classifier", layout="wide")
    st.title("E-nose breath sensor — group classifier")
    st.caption(
        "Research prototype. Predicts dataset group membership (COPD / Control / "
        "Smokers / Air), NOT a clinical diagnosis and NOT asthma phenotype. "
        "See README / project notes for what this model actually is."
    )

    model = load_model()
    if model is None:
        st.error(
            f"No model found at `models/champion_model.joblib`. Run the pipeline "
            "through champion_model.py first."
        )
        return

    champion_info = load_champion_info()
    train_df, test_df = load_reference_data()
    if train_df is None:
        st.error("No `data/interim/feature_train_selected.parquet` found — run the pipeline first.")
        return

    feature_cols = feature_columns_from(train_df)
    medians = train_df[feature_cols].median()
    mins = train_df[feature_cols].min()
    maxs = train_df[feature_cols].max()
    grouped = group_features_by_sensor(feature_cols)

    with st.sidebar:
        st.header("Model info")
        st.write(f"**Model:** {champion_info.get('model', 'unknown')} "
                 f"({champion_info.get('variant', 'unknown')})")
        cv_metrics = champion_info.get("metrics", {})
        if cv_metrics:
            st.write("**CV performance (train split):**")
            for m, v in cv_metrics.items():
                st.write(f"- {m}: {v.get('mean', float('nan')):.3f} ± {v.get('std', float('nan')):.3f}")
        st.divider()

        st.header("Autofill (demo only)")
        st.caption(
            "Loads a real sample's feature values for convenience. Doesn't affect "
            "the model — purely fills the form below."
        )
        autofill_source = st.radio("Source", ["Train-set median (default)", "Random train sample", "Random test sample"])

        if st.button("Apply autofill"):
            if autofill_source == "Random train sample":
                row = train_df.sample(1).iloc[0]
                st.session_state["_autofill_row"] = row
                st.session_state["_autofill_label"] = f"train sample, true group = {row['group']}"
            elif autofill_source == "Random test sample" and test_df is not None:
                row = test_df.sample(1).iloc[0]
                st.session_state["_autofill_row"] = row
                st.session_state["_autofill_label"] = f"held-out test sample, true group = {row['group']}"
            else:
                st.session_state["_autofill_row"] = None
                st.session_state["_autofill_label"] = "train-set medians"
            for col in feature_cols:
                fill_row = st.session_state.get("_autofill_row")
                st.session_state[f"feat_{col}"] = (
                    float(fill_row[col]) if fill_row is not None else float(medians[col])
                )

        if "_autofill_label" in st.session_state:
            st.info(f"Form filled from: {st.session_state['_autofill_label']}")

    st.subheader("Sensor feature values")
    st.caption(
        "Grouped by sensor. Only the features that survived correlation/variance "
        "pruning (select_features.py) are shown — the model doesn't use the rest."
    )

    values = {}
    tabs = st.tabs(list(grouped.keys()))
    for tab, sensor in zip(tabs, grouped.keys()):
        with tab:
            cols_per_row = 3
            sensor_features = grouped[sensor]
            for i in range(0, len(sensor_features), cols_per_row):
                row_features = sensor_features[i:i + cols_per_row]
                cols = st.columns(len(row_features))
                for col_widget, feat in zip(cols, row_features):
                    default = st.session_state.get(f"feat_{feat}", float(medians[feat]))
                    with col_widget:
                        values[feat] = st.number_input(
                            feat,
                            value=default,
                            min_value=float(mins[feat]) if np.isfinite(mins[feat]) else None,
                            max_value=float(maxs[feat]) if np.isfinite(maxs[feat]) else None,
                            key=f"feat_{feat}",
                            format="%.4f",
                        )

    st.divider()

    if st.button("Predict", type="primary"):
        X = pd.DataFrame([values])[feature_cols]  # enforce training column order
        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        pred_label = CLASSES[pred_idx]

        st.subheader(f"Prediction: `{pred_label}`")

        proba_df = pd.DataFrame({"group": CLASSES, "probability": proba}).set_index("group")
        st.bar_chart(proba_df)

        top_prob = proba[pred_idx]
        if top_prob < 0.6:
            st.warning(
                f"Top class probability is only {top_prob:.2f} — the model isn't confident. "
                "Treat this prediction as unreliable rather than a close call."
            )

        st.caption(
            "Reminder: even a high-confidence prediction here reflects performance on "
            "this study's ~86 samples, not validated real-world diagnostic accuracy. "
            "See reports/eda_report.md and data/processed/test_evaluation.json for caveats."
        )


if __name__ == "__main__":
    main()