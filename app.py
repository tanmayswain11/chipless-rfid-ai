# ============================================================
# CHIPLESS RFID MULTIPARAMETER SENSING DASHBOARD
# File: app.py
# ============================================================
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)
from ml.inference import MultiparameterPredictor
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CRFID Multiparameter Sensing",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
SAVED_MODELS = ROOT / "saved_models"
# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .dashboard-title {
        font-size: 30px;
        font-weight: 750;
        color: #172554;
    }
    .dashboard-subtitle {
        color: #64748b;
        font-size: 14px;
    }
    .metric-card {
        background-color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #172554;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# SESSION STATE
# ============================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_curve" not in st.session_state:
    st.session_state.prediction_curve = None
# ============================================================
# LOAD INFERENCE ENGINE
# ============================================================
@st.cache_resource
def load_predictor():
    return MultiparameterPredictor(
        project_root=ROOT
    )
# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_frequency_column(df):
    for column in df.columns:
        name = str(column).lower()
        if "freq" in name:
            return column
    return df.columns[0]
def find_magnitude_columns(df, frequency_column):
    return [
        column
        for column in df.columns
        if column != frequency_column
        and pd.api.types.is_numeric_dtype(df[column])
    ]
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )
    actual = actual[valid]
    predicted = predicted[valid]
    if len(actual) == 0:
        raise ValueError("No valid actual-predicted pairs.")
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    # R2 requires at least two observations
    if len(actual) >= 2:
        r2 = r2_score(actual, predicted)
        explained_variance = explained_variance_score(
            actual, predicted
        )
    else:
        r2 = np.nan
        explained_variance = np.nan
    return {
        "Samples": len(actual),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Median Absolute Error": median_absolute_error(
            actual, predicted
        ),
        "Explained Variance": explained_variance,
    }
def find_metric_columns(df):
    aliases = {
        "MAE": ["mae", "mean_absolute_error"],
        "RMSE": ["rmse", "root_mean_squared_error"],
        "MSE": ["mse", "mean_squared_error"],
        "R2": ["r2", "r2_score", "r2_oof"],
    }
    result = {}
    for metric, candidates in aliases.items():
        for column in df.columns:
            normalized = (
                str(column)
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            if any(
                normalized == candidate
                or normalized.endswith("_" + candidate)
                for candidate in candidates
            ):
                result[metric] = column
                break
    return result
def find_prediction_columns(df):
    actual_column = None
    predicted_column = None
    for column in df.columns:
        name = str(column).lower().replace(" ", "_")
        if actual_column is None and (
            "actual" in name or "true" in name or "ground_truth" in name
        ):
            actual_column = column
        if predicted_column is None and (
            "predicted" in name or "prediction" in name or "y_pred" in name
        ):
            predicted_column = column
    return actual_column, predicted_column
def load_evaluation_files():
    if not PROCESSED_DATA.exists():
        return []
    files = []
    for path in PROCESSED_DATA.glob("*.csv"):
        if "comparison" in path.name.lower():
            files.append(path)
    return files
def display_metric_cards(metrics):
    columns = st.columns(4)
    items = [
        ("MAE", metrics["MAE"]),
        ("RMSE", metrics["RMSE"]),
        ("MSE", metrics["MSE"]),
        ("R²", metrics["R2"]),
    ]
    for col, (name, value) in zip(columns, items):
        with col:
            if pd.isna(value):
                display_value = "N/A"
            else:
                display_value = f"{value:.4f}"
            st.metric(
                label=name,
                value=display_value
            )
def show_model_evaluation_file(path):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return
    st.markdown(f"**File:** `{path.name}`")
    if df.empty:
        st.warning("This evaluation file is empty.")
        return
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    metric_columns = find_metric_columns(df)
    if metric_columns:
        chart_data = pd.DataFrame()
        for metric, column in metric_columns.items():
            chart_data[metric] = pd.to_numeric(
                df[column],
                errors="coerce"
            )
        if "Model" in df.columns:
            chart_data.index = df["Model"].astype(str)
        elif "model_name" in df.columns:
            chart_data.index = df["model_name"].astype(str)
        else:
            chart_data.index = [
                f"Model {i + 1}"
                for i in range(len(chart_data))
            ]
        st.markdown("**Model comparison**")
        st.bar_chart(
            chart_data[[
                col for col in ["MAE", "RMSE"]
                if col in chart_data.columns
            ]],
            use_container_width=True
        )
        if "R2" in chart_data.columns:
            st.markdown("**R² comparison**")
            st.bar_chart(
                chart_data[["R2"]],
                use_container_width=True
            )
    else:
        st.info(
            "No recognized MAE, RMSE, MSE or R² columns "
            "were found in this file. Showing the saved table only."
        )
    actual_col, predicted_col = find_prediction_columns(df)
    if actual_col is not None and predicted_col is not None:
        actual = pd.to_numeric(
            df[actual_col],
            errors="coerce"
        )
        predicted = pd.to_numeric(
            df[predicted_col],
            errors="coerce"
        )
        valid = actual.notna() & predicted.notna()
        if valid.sum() >= 2:
            metrics = calculate_metrics(
                actual[valid],
                predicted[valid]
            )
            st.markdown("**Metrics from actual-predicted pairs**")
            display_metric_cards(metrics)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(
                actual[valid],
                predicted[valid],
                alpha=0.8
            )
            lower = min(
                actual[valid].min(),
                predicted[valid].min()
            )
            upper = max(
                actual[valid].max(),
                predicted[valid].max()
            )
            ax.plot(
                [lower, upper],
                [lower, upper],
                linestyle="--"
            )
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            plt.close(fig)
            residuals = (
                actual[valid] - predicted[valid]
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(
                predicted[valid],
                residuals,
                alpha=0.8
            )
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            ax.set_title("Residual Analysis")
            st.pyplot(fig)
            plt.close(fig)
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📡 CRFID Dashboard")
st.sidebar.caption(
    "Chipless RFID Multiparameter Sensing System"
)
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Inference",
        "RF Signal Analysis",
        "Model Evaluation",
        "Dataset Explorer",
        "Model Information",
    ]
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **Developed for**
    Chipless RFID Multiparameter Sensing
    **Parameters**
    - Temperature
    - Water Volume
    - Relative Humidity
    """
)
# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown(
        '<div class="dashboard-title">'
        'CRFID Multiparameter Sensing Dashboard'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="dashboard-subtitle">'
        'Machine Learning Based RF Sensor Analysis and Prediction'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()
    try:
        predictor = load_predictor()
        st.success("All three trained models loaded successfully.")
    except Exception as exc:
        predictor = None
        st.error(f"Model loading failed: {exc}")
    st.subheader("Model Performance Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🌡️ Temperature")
        st.success("Model Loaded")
        st.markdown("**Extra Trees Regressor**")
        st.metric("Training Curves", "10")
        st.metric("MAE", "9.493 °C")
        st.metric("RMSE", "11.108 °C")
        st.metric("R²", "0.820")
    with col2:
        st.markdown("### 💧 Water Volume")
        st.success("Model Loaded")
        st.markdown("**SVR RBF**")
        st.metric("Training Curves", "5")
        st.metric("MAE", "7.638 mL")
        st.metric("RMSE", "8.784 mL")
        st.metric("R²", "-0.543")
        st.warning("Exploratory performance is poor.")
    with col3:
        st.markdown("### 💦 Relative Humidity")
        st.success("Model Loaded")
        st.markdown("**Extra Trees Regressor**")
        st.metric("Training Curves", "3")
        st.warning("Labels are assumed.")
        st.error("No independent validation metrics.")
    st.divider()
    st.subheader("Latest Prediction")
    result = st.session_state.prediction_result
    if result is not None:
        predictions = result["predictions"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Temperature",
            f'{predictions["temperature"]["prediction"]:.3f} °C'
        )
        c2.metric(
            "Water Volume",
            f'{predictions["water_volume"]["prediction"]:.3f} mL'
        )
        c3.metric(
            "Relative Humidity",
            f'{predictions["relative_humidity"]["prediction"]:.3f} %'
        )
    else:
        st.info("No predictions available. Run Live Inference first.")
    st.divider()
    st.subheader("Evaluation Files")
    evaluation_files = load_evaluation_files()
    if evaluation_files:
        for path in evaluation_files:
            st.write(f"📄 {path.name}")
    else:
        st.info("No model comparison CSV files found.")
# ============================================================
# PAGE 2: LIVE INFERENCE
# ============================================================
elif page == "Live Inference":
    st.title("Live Multiparameter Inference")
    st.write(
        "Upload an RF measurement CSV containing frequency "
        "and magnitude response columns."
    )
    try:
        predictor = load_predictor()
        st.success("Inference engine ready.")
    except Exception as exc:
        st.error(f"Could not load inference engine: {exc}")
        st.stop()
    uploaded_file = st.file_uploader(
        "Upload RF measurement CSV",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data")
            st.dataframe(
                df.head(10),
                use_container_width=True
            )
            if len(df.columns) < 2:
                st.error(
                    "CSV must contain frequency and magnitude columns."
                )
                st.stop()
            frequency_column = find_frequency_column(df)
            magnitude_columns = find_magnitude_columns(
                df,
                frequency_column
            )
            if not magnitude_columns:
                st.error(
                    "No numeric magnitude columns were detected."
                )
                st.stop()
            selected_magnitude = st.selectbox(
                "Select magnitude response curve",
                magnitude_columns
            )
            # Convert frequency and magnitude into numeric values
            signal_df = pd.DataFrame({
                "frequency": pd.to_numeric(
                    df[frequency_column],
                    errors="coerce"
                ),
                "magnitude": pd.to_numeric(
                    df[selected_magnitude],
                    errors="coerce"
                )
            })

            # Remove missing and invalid values
            signal_df = signal_df.dropna()

            # Sort frequency in ascending order
            signal_df = signal_df.sort_values(
                by="frequency",
                ascending=True
            )

            # Handle duplicate frequency values by averaging magnitude
            signal_df = (
                signal_df
                .groupby("frequency", as_index=False)["magnitude"]
                .mean()
            )

            # Extract NumPy arrays
            frequency = signal_df["frequency"].to_numpy(dtype=float)
            magnitude = signal_df["magnitude"].to_numpy(dtype=float)

            # Validate frequency
            if len(frequency) < 2:
                st.error("Insufficient valid frequency points.")
                st.stop()

            if not np.all(np.diff(frequency) > 0):
                st.error("Frequency values are still not strictly increasing.")
                st.stop()

            st.success(
                f"RF signal prepared successfully: {len(frequency)} frequency points."
            )
            if st.button(
                "Run Multiparameter Prediction",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Processing RF response..."):
                    result = predictor.predict(
                        frequency,
                        magnitude
                    )
                st.session_state.prediction_result = result
                st.session_state.prediction_curve = {
                    "frequency": frequency,
                    "magnitude": magnitude,
                }
                st.success(
                    "Multiparameter inference completed."
                )
            result = st.session_state.prediction_result
            if result is not None:
                st.divider()
                st.subheader("Prediction Results")
                predictions = result["predictions"]
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Temperature",
                    f'{predictions["temperature"]["prediction"]:.3f} °C'
                )
                c2.metric(
                    "Water Volume",
                    f'{predictions["water_volume"]["prediction"]:.3f} mL'
                )
                c3.metric(
                    "Relative Humidity",
                    f'{predictions["relative_humidity"]["prediction"]:.3f} %'
                )
                st.write(
                    f"Frequency points: {result['frequency_points']}"
                )
                st.write(
                    f"Extracted features: {result['feature_count']}"
                )
                st.write(
                    f"Model input shape: {result['input_shape']}"
                )
                st.subheader("Model Details")
                model_rows = []
                for target, data in predictions.items():
                    model_rows.append({
                        "Parameter": target,
                        "Prediction": data["prediction"],
                        "Unit": data["unit"],
                        "Model": data["model_name"],
                        "Experimental": data["experimental"],
                        "Label Status": data["label_status"],
                        "Training Samples": data["training_samples"],
                    })
                st.dataframe(
                    pd.DataFrame(model_rows),
                    use_container_width=True,
                    hide_index=True
                )
                st.subheader("Scientific Warnings")
                for warning in result["warnings"]:
                    st.warning(warning)
                export_df = pd.DataFrame([
                    {
                        "parameter": target,
                        "prediction": data["prediction"],
                        "unit": data["unit"],
                        "model": data["model_name"],
                        "experimental": data["experimental"],
                        "label_status": data["label_status"],
                    }
                    for target, data in predictions.items()
                ])
                st.download_button(
                    "Download Prediction CSV",
                    data=export_df.to_csv(index=False),
                    file_name="crfid_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(
                f"Could not process uploaded data: {exc}"
            )
# ============================================================
# PAGE 3: RF SIGNAL ANALYSIS
# ============================================================
elif page == "RF Signal Analysis":
    st.title("RF Signal Analysis")
    curve = st.session_state.prediction_curve
    if curve is None:
        st.info(
            "Run an inference first to display its RF response."
        )
    else:
        frequency = curve["frequency"]
        magnitude = curve["magnitude"]
        st.subheader("Frequency Response")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            frequency,
            magnitude,
            linewidth=1.5
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Chipless RFID Frequency Response")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Signal Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Frequency Points",
            len(frequency)
        )
        c2.metric(
            "Maximum Magnitude",
            f"{np.max(magnitude):.3f} dB"
        )
        c3.metric(
            "Minimum Magnitude",
            f"{np.min(magnitude):.3f} dB"
        )
        c4.metric(
            "Mean Magnitude",
            f"{np.mean(magnitude):.3f} dB"
        )
        st.write(
            pd.DataFrame({
                "Frequency (Hz)": frequency,
                "Magnitude (dB)": magnitude,
            })
        )
# ============================================================
# PAGE 4: MODEL EVALUATION
# ============================================================
elif page == "Model Evaluation":
    st.title("ML Model Evaluation")
    st.write(
        "Explore saved cross-validation and model-comparison results."
    )
    st.info(
        "MAE, RMSE, MSE and R² must be calculated from "
        "valid held-out or out-of-fold predictions. "
        "Training-fit predictions are not validation results."
    )
    evaluation_files = load_evaluation_files()
    if not evaluation_files:
        st.warning(
            "No comparison CSV files found in data/processed/. "
            "Check your training output files."
        )
    else:
        selected_file = st.selectbox(
            "Select evaluation result",
            evaluation_files,
            format_func=lambda x: x.name
        )
        show_model_evaluation_file(selected_file)
    st.divider()
    st.subheader("Relative Humidity Evaluation")
    rh_summary_path = (
        PROCESSED_DATA
        / "relative_humidity_summary.json"
    )
    if rh_summary_path.exists():
        try:
            with open(
                rh_summary_path,
                "r",
                encoding="utf-8"
            ) as file:
                rh_summary = json.load(file)
            st.json(rh_summary)
            st.warning(
                "The RH model has only three independent curves "
                "with assumed labels. No independent validation "
                "metrics are available."
            )
        except Exception as exc:
            st.error(
                f"Could not read RH summary: {exc}"
            )
    else:
        st.warning(
            "RH summary JSON was not found."
        )
# ============================================================
# PAGE 5: DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    if not RAW_DATA.exists():
        st.error(
            f"Raw data directory not found: {RAW_DATA}"
        )
    else:
        csv_files = list(RAW_DATA.glob("*.csv"))
        if not csv_files:
            st.warning("No CSV datasets found.")
        else:
            selected_dataset = st.selectbox(
                "Select dataset",
                csv_files,
                format_func=lambda x: x.name
            )
            try:
                # RH file is headerless in this project.
                if selected_dataset.name.lower() == "realhumid cf.csv":
                    df = pd.read_csv(
                        selected_dataset,
                        header=None
                    )
                    if df.shape[1] == 4:
                        df.columns = [
                            "frequency_hz",
                            "rh_curve_1",
                            "rh_curve_2",
                            "rh_curve_3",
                        ]
                else:
                    df = pd.read_csv(selected_dataset)
                st.subheader(selected_dataset.name)
                st.dataframe(
                    df.head(20),
                    use_container_width=True
                )
                c1, c2 = st.columns(2)
                c1.metric(
                    "Rows",
                    df.shape[0]
                )
                c2.metric(
                    "Columns",
                    df.shape[1]
                )
                st.subheader("Dataset Statistics")
                st.dataframe(
                    df.describe(),
                    use_container_width=True
                )
            except Exception as exc:
                st.error(
                    f"Could not load dataset: {exc}"
                )
# ============================================================
# PAGE 6: MODEL INFORMATION
# ============================================================
elif page == "Model Information":
    st.title("Trained Model Information")
    model_files = []
    if SAVED_MODELS.exists():
        model_files = list(
            SAVED_MODELS.rglob("*.joblib")
        )
    if not model_files:
        st.warning("No saved model artifacts found.")
    else:
        model_data = []
        for path in model_files:
            model_data.append({
                "Model File": path.name,
                "Target": path.parent.name,
                "Path": str(path.relative_to(ROOT)),
            })
        st.dataframe(
            pd.DataFrame(model_data),
            use_container_width=True,
            hide_index=True
        )
    st.divider()
    st.subheader("Selected Models")
    selected_models = {
        "Temperature": "temperature_extra_trees.joblib",
        "Water Volume": "water_volume_svr_rbf.joblib",
        "Relative Humidity": "relative_humidity_extra_trees.joblib",
    }
    for target, filename in selected_models.items():
        st.write(f"**{target}:** `{filename}`")
    st.divider()
    st.subheader("Evaluation Limitations")
    st.markdown(
        """
        - Temperature has only 10 independent measured curves.
        - Water-volume data has only 5 independent curves.
        - RH data has only 3 independent curves.
        - RH labels are provisional assumptions.
        - Water-volume LOOCV R² is negative.
        - Model compatibility does not establish real-world accuracy.
        """
    )
# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "CRFID Multiparameter Sensing | "
    "Machine Learning Research Dashboard | "
    "Experimental results require independent validation."
)
