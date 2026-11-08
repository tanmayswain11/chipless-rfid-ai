
# from sklearn.ensemble import (
#     ExtraTreesRegressor,
#     RandomForestRegressor,
#     GradientBoostingRegressor,
# )

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import Ridge

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler


# def _build_model_dictionary():
#     """
#     Create the common regression model collection.

#     The same model definitions are reused by normal training
#     and experimental RH training.
#     """

#     models = {

#         "Extra Trees": ExtraTreesRegressor(
#             n_estimators=300,
#             random_state=42,
#             min_samples_leaf=1,
#             max_features=1.0,
#             n_jobs=-1
#         ),

#         "Random Forest": RandomForestRegressor(
#             n_estimators=300,
#             random_state=42,
#             min_samples_leaf=1,
#             max_features=1.0,
#             n_jobs=-1
#         ),

#         "Gradient Boosting": GradientBoostingRegressor(
#             n_estimators=100,
#             learning_rate=0.03,
#             max_depth=2,
#             loss="huber",
#             random_state=42
#         ),

#         "Decision Tree": DecisionTreeRegressor(
#             max_depth=3,
#             min_samples_leaf=2,
#             random_state=42
#         ),

#         "SVR RBF": Pipeline([
#             ("scaler", StandardScaler()),
#             ("regressor", SVR(
#                 kernel="rbf",
#                 C=10.0,
#                 epsilon=0.1,
#                 gamma="scale"
#             ))
#         ]),

#         "Ridge Regression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("regressor", Ridge(alpha=10.0))
#         ]),
#     }

#     return models


# def get_models(n_samples):
#     """
#     Models for normal supervised regression.

#     Temperature and water-volume datasets must have
#     at least four independent labelled curves.
#     """

#     if n_samples < 4:
#         raise ValueError(
#             "Normal model evaluation requires at least "
#             "4 independent labelled curves."
#         )

#     return _build_model_dictionary()


# def get_rh_models(n_samples):
#     """
#     Separate model factory for experimental RH training.

#     Allows fitting with three independent labelled curves.

#     IMPORTANT:
#     Three samples are not sufficient to establish reliable
#     generalization performance. This function is intended
#     for experimental model fitting only.
#     """

#     if n_samples < 3:
#         raise ValueError(
#             "Experimental RH training requires at least "
#             "3 independent labelled curves."
#         )

#     return _build_model_dictionary()


# def get_model_names():
#     """
#     Return all available model names.
#     """

#     return list(_build_model_dictionary().keys())











# ============================================================
# CHIPLESS RFID MULTIPARAMETER SENSING DASHBOARD
# File: app.py
# ============================================================
# from pathlib import Path
# import io
# import json
# import warnings
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score,
#     median_absolute_error,
#     explained_variance_score,
# )
# from ml.inference import MultiparameterPredictor
# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================
# st.set_page_config(
#     page_title="CRFID Multiparameter Sensing",
#     page_icon="📡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# # ============================================================
# # PROJECT PATHS
# # ============================================================
# ROOT = Path(__file__).resolve().parent
# RAW_DATA = ROOT / "data" / "raw"
# PROCESSED_DATA = ROOT / "data" / "processed"
# SAVED_MODELS = ROOT / "saved_models"
# # ============================================================
# # CUSTOM CSS
# # ============================================================
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #f7f9fc;
#     }
#     .block-container {
#         padding-top: 1.5rem;
#         padding-bottom: 2rem;
#     }
#     .dashboard-title {
#         font-size: 30px;
#         font-weight: 750;
#         color: #172554;
#     }
#     .dashboard-subtitle {
#         color: #64748b;
#         font-size: 14px;
#     }
#     .metric-card {
#         background-color: white;
#         padding: 18px;
#         border-radius: 12px;
#         border: 1px solid #e2e8f0;
#     }
#     .section-title {
#         font-size: 21px;
#         font-weight: 700;
#         color: #172554;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# # ============================================================
# # SESSION STATE
# # ============================================================
# if "prediction_result" not in st.session_state:
#     st.session_state.prediction_result = None
# if "prediction_curve" not in st.session_state:
#     st.session_state.prediction_curve = None
# if "uploaded_csv_bytes" not in st.session_state:
#     st.session_state.uploaded_csv_bytes = None
# if "uploaded_csv_name" not in st.session_state:
#     st.session_state.uploaded_csv_name = None

# def save_uploaded_csv():
#     """Keep uploaded CSV bytes outside the file_uploader widget state."""
#     uploaded = st.session_state.get("rf_csv_uploader")
#     if uploaded is not None:
#         st.session_state.uploaded_csv_bytes = uploaded.getvalue()
#         st.session_state.uploaded_csv_name = uploaded.name
# # ============================================================
# # LOAD INFERENCE ENGINE
# # ============================================================
# @st.cache_resource
# def load_predictor():
#     return MultiparameterPredictor(
#         project_root=ROOT
#     )
# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================
# def find_frequency_column(df):
#     for column in df.columns:
#         name = str(column).lower()
#         if "freq" in name:
#             return column
#     return df.columns[0]
# def find_magnitude_columns(df, frequency_column):
#     return [
#         column
#         for column in df.columns
#         if column != frequency_column
#         and pd.api.types.is_numeric_dtype(df[column])
#     ]
# def calculate_metrics(actual, predicted):
#     actual = np.asarray(actual, dtype=float)
#     predicted = np.asarray(predicted, dtype=float)
#     valid = (
#         np.isfinite(actual)
#         & np.isfinite(predicted)
#     )
#     actual = actual[valid]
#     predicted = predicted[valid]
#     if len(actual) == 0:
#         raise ValueError("No valid actual-predicted pairs.")
#     mse = mean_squared_error(actual, predicted)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(actual, predicted)
#     # R2 requires at least two observations
#     if len(actual) >= 2:
#         r2 = r2_score(actual, predicted)
#         explained_variance = explained_variance_score(
#             actual, predicted
#         )
#     else:
#         r2 = np.nan
#         explained_variance = np.nan
#     return {
#         "Samples": len(actual),
#         "MAE": mae,
#         "MSE": mse,
#         "RMSE": rmse,
#         "R2": r2,
#         "Median Absolute Error": median_absolute_error(
#             actual, predicted
#         ),
#         "Explained Variance": explained_variance,
#     }
# def find_metric_columns(df):
#     aliases = {
#         "MAE": ["mae", "mean_absolute_error"],
#         "RMSE": ["rmse", "root_mean_squared_error"],
#         "MSE": ["mse", "mean_squared_error"],
#         "R2": ["r2", "r2_score", "r2_oof"],
#     }
#     result = {}
#     for metric, candidates in aliases.items():
#         for column in df.columns:
#             normalized = (
#                 str(column)
#                 .lower()
#                 .replace(" ", "_")
#                 .replace("-", "_")
#             )
#             if any(
#                 normalized == candidate
#                 or normalized.endswith("_" + candidate)
#                 for candidate in candidates
#             ):
#                 result[metric] = column
#                 break
#     return result
# def find_prediction_columns(df):
#     actual_column = None
#     predicted_column = None
#     for column in df.columns:
#         name = str(column).lower().replace(" ", "_")
#         if actual_column is None and (
#             "actual" in name or "true" in name or "ground_truth" in name
#         ):
#             actual_column = column
#         if predicted_column is None and (
#             "predicted" in name or "prediction" in name or "y_pred" in name
#         ):
#             predicted_column = column
#     return actual_column, predicted_column
# def load_evaluation_files():
#     if not PROCESSED_DATA.exists():
#         return []
#     files = []
#     for path in PROCESSED_DATA.glob("*.csv"):
#         if "comparison" in path.name.lower():
#             files.append(path)
#     return files
# def display_metric_cards(metrics):
#     columns = st.columns(4)
#     items = [
#         ("MAE", metrics["MAE"]),
#         ("RMSE", metrics["RMSE"]),
#         ("MSE", metrics["MSE"]),
#         ("R²", metrics["R2"]),
#     ]
#     for col, (name, value) in zip(columns, items):
#         with col:
#             if pd.isna(value):
#                 display_value = "N/A"
#             else:
#                 display_value = f"{value:.4f}"
#             st.metric(
#                 label=name,
#                 value=display_value
#             )
# def show_model_evaluation_file(path):
#     try:
#         df = pd.read_csv(path)
#     except Exception as exc:
#         st.error(f"Could not read {path.name}: {exc}")
#         return
#     st.markdown(f"**File:** `{path.name}`")
#     if df.empty:
#         st.warning("This evaluation file is empty.")
#         return
#     st.dataframe(
#         df,
#         use_container_width=True,
#         hide_index=True
#     )
#     metric_columns = find_metric_columns(df)
#     if metric_columns:
#         chart_data = pd.DataFrame()
#         for metric, column in metric_columns.items():
#             chart_data[metric] = pd.to_numeric(
#                 df[column],
#                 errors="coerce"
#             )
#         if "Model" in df.columns:
#             chart_data.index = df["Model"].astype(str)
#         elif "model_name" in df.columns:
#             chart_data.index = df["model_name"].astype(str)
#         else:
#             chart_data.index = [
#                 f"Model {i + 1}"
#                 for i in range(len(chart_data))
#             ]
#         st.markdown("**Model comparison**")
#         st.bar_chart(
#             chart_data[[
#                 col for col in ["MAE", "RMSE"]
#                 if col in chart_data.columns
#             ]],
#             use_container_width=True
#         )
#         if "R2" in chart_data.columns:
#             st.markdown("**R² comparison**")
#             st.bar_chart(
#                 chart_data[["R2"]],
#                 use_container_width=True
#             )
#     else:
#         st.info(
#             "No recognized MAE, RMSE, MSE or R² columns "
#             "were found in this file. Showing the saved table only."
#         )
#     actual_col, predicted_col = find_prediction_columns(df)
#     if actual_col is not None and predicted_col is not None:
#         actual = pd.to_numeric(
#             df[actual_col],
#             errors="coerce"
#         )
#         predicted = pd.to_numeric(
#             df[predicted_col],
#             errors="coerce"
#         )
#         valid = actual.notna() & predicted.notna()
#         if valid.sum() >= 2:
#             metrics = calculate_metrics(
#                 actual[valid],
#                 predicted[valid]
#             )
#             st.markdown("**Metrics from actual-predicted pairs**")
#             display_metric_cards(metrics)
#             fig, ax = plt.subplots(figsize=(7, 4))
#             ax.scatter(
#                 actual[valid],
#                 predicted[valid],
#                 alpha=0.8
#             )
#             lower = min(
#                 actual[valid].min(),
#                 predicted[valid].min()
#             )
#             upper = max(
#                 actual[valid].max(),
#                 predicted[valid].max()
#             )
#             ax.plot(
#                 [lower, upper],
#                 [lower, upper],
#                 linestyle="--"
#             )
#             ax.set_xlabel("Actual")
#             ax.set_ylabel("Predicted")
#             ax.set_title("Actual vs Predicted")
#             st.pyplot(fig)
#             plt.close(fig)
#             residuals = (
#                 actual[valid] - predicted[valid]
#             )
#             fig, ax = plt.subplots(figsize=(7, 4))
#             ax.scatter(
#                 predicted[valid],
#                 residuals,
#                 alpha=0.8
#             )
#             ax.axhline(0, linestyle="--")
#             ax.set_xlabel("Predicted")
#             ax.set_ylabel("Residual")
#             ax.set_title("Residual Analysis")
#             st.pyplot(fig)
#             plt.close(fig)
# # ============================================================
# # SIDEBAR
# # ============================================================
# st.sidebar.title("📡 CRFID Dashboard")
# st.sidebar.caption(
#     "Chipless RFID Multiparameter Sensing System"
# )
# page = st.sidebar.radio(
#     "Navigation",
#     [
#         "Overview",
#         "Live Inference",
#         "RF Signal Analysis",
#         "Model Evaluation",
#         "Dataset Explorer",
#         "Model Information",
#     ]
# )
# st.sidebar.divider()
# st.sidebar.markdown(
#     """
#     **Developed for**
#     Chipless RFID Multiparameter Sensing
#     **Parameters**
#     - Temperature
#     - Water Volume
#     - Relative Humidity
#     """
# )
# # ============================================================
# # PAGE 1: OVERVIEW
# # ============================================================
# # ============================================================
# # PAGE 1: OVERVIEW
# # ============================================================
# if page == "Overview":
#     st.markdown(
#         '<div class="dashboard-title">'
#         'CRFID Multiparameter Sensing Dashboard'
#         '</div>',
#         unsafe_allow_html=True
#     )
#     st.markdown(
#         '<div class="dashboard-subtitle">'
#         'Machine Learning Based RF Sensor Analysis and Prediction'
#         '</div>',
#         unsafe_allow_html=True
#     )
#     st.divider()
#     try:
#         predictor = load_predictor()
#         st.success("All three trained models loaded successfully.")
#     except Exception as exc:
#         predictor = None
#         st.error(f"Model loading failed: {exc}")
#     st.subheader("Model Performance Overview")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("### 🌡️ Temperature")
#         st.success("Model Loaded")
#         st.markdown("**Extra Trees Regressor**")
#         st.metric("Training Curves", "10")
#         st.metric("MAE", "9.493 °C")
#         st.metric("RMSE", "11.108 °C")
#         st.metric("R²", "0.820")
#     with col2:
#         st.markdown("### 💧 Water Volume")
#         st.success("Model Loaded")
#         st.markdown("**SVR RBF**")
#         st.metric("Training Curves", "5")
#         st.metric("MAE", "7.638 mL")
#         st.metric("RMSE", "8.784 mL")
#         st.metric("R²", "-0.543")
#         st.warning("Exploratory performance is poor.")
#     with col3:
#         st.markdown("### 💦 Relative Humidity")
#         st.success("Model Loaded")
#         st.markdown("**Extra Trees Regressor**")
#         st.metric("Training Curves", "3")
#         st.warning("Labels are assumed.")
#         st.error("No independent validation metrics.")
#     st.divider()
#     st.subheader("Latest Prediction")
#     result = st.session_state.prediction_result
#     if result is not None:
#         predictions = result["predictions"]
#         c1, c2, c3 = st.columns(3)
#         c1.metric(
#             "Temperature",
#             f'{predictions["temperature"]["prediction"]:.3f} °C'
#         )
#         c2.metric(
#             "Water Volume",
#             f'{predictions["water_volume"]["prediction"]:.3f} mL'
#         )
#         c3.metric(
#             "Relative Humidity",
#             f'{predictions["relative_humidity"]["prediction"]:.3f} %'
#         )
#     else:
#         st.info("No predictions available. Run Live Inference first.")
#     st.divider()
#     st.subheader("Evaluation Files")
#     evaluation_files = load_evaluation_files()
#     if evaluation_files:
#         for path in evaluation_files:
#             st.write(f"📄 {path.name}")
#     else:
#         st.info("No model comparison CSV files found.")
# # ============================================================
# # PAGE 2: LIVE INFERENCE
# # ============================================================
# elif page == "Live Inference":
#     st.title("Live Multiparameter Inference")
#     st.write(
#         "Upload an RF measurement CSV containing frequency "
#         "and magnitude response columns."
#     )
#     try:
#         predictor = load_predictor()
#         st.success("Inference engine ready.")
#     except Exception as exc:
#         st.error(f"Could not load inference engine: {exc}")
#         st.stop()
#     st.file_uploader(
#         "Upload RF measurement CSV (upload once; it stays available while you navigate)",
#         type=["csv"],
#         key="rf_csv_uploader",
#         on_change=save_uploaded_csv,
#     )

#     uploaded_csv_bytes = st.session_state.uploaded_csv_bytes
#     if uploaded_csv_bytes is not None:
#         try:
#             df = pd.read_csv(io.BytesIO(uploaded_csv_bytes))
#             st.success(
#                 f"Using uploaded file: {st.session_state.uploaded_csv_name}"
#             )
#             st.subheader("Uploaded Data")
#             st.dataframe(
#                 df.head(10),
#                 use_container_width=True
#             )
#             if len(df.columns) < 2:
#                 st.error(
#                     "CSV must contain frequency and magnitude columns."
#                 )
#                 st.stop()
#             frequency_column = find_frequency_column(df)
#             magnitude_columns = find_magnitude_columns(
#                 df,
#                 frequency_column
#             )
#             if not magnitude_columns:
#                 st.error(
#                     "No numeric magnitude columns were detected."
#                 )
#                 st.stop()
#             selected_magnitude = st.selectbox(
#                 "Select magnitude response curve",
#                 magnitude_columns
#             )
#             # Convert frequency and magnitude into numeric values
#             signal_df = pd.DataFrame({
#                 "frequency": pd.to_numeric(
#                     df[frequency_column],
#                     errors="coerce"
#                 ),
#                 "magnitude": pd.to_numeric(
#                     df[selected_magnitude],
#                     errors="coerce"
#                 )
#             })

#             # Remove missing and invalid values
#             signal_df = signal_df.dropna()

#             # Sort frequency in ascending order
#             signal_df = signal_df.sort_values(
#                 by="frequency",
#                 ascending=True
#             )

#             # Handle duplicate frequency values by averaging magnitude
#             signal_df = (
#                 signal_df
#                 .groupby("frequency", as_index=False)["magnitude"]
#                 .mean()
#             )

#             # Extract NumPy arrays
#             frequency = signal_df["frequency"].to_numpy(dtype=float)
#             magnitude = signal_df["magnitude"].to_numpy(dtype=float)

#             # Validate frequency
#             if len(frequency) < 2:
#                 st.error("Insufficient valid frequency points.")
#                 st.stop()

#             if not np.all(np.diff(frequency) > 0):
#                 st.error("Frequency values are still not strictly increasing.")
#                 st.stop()

#             st.success(
#                 f"RF signal prepared successfully: {len(frequency)} frequency points."
#             )
#             if st.button(
#                 "Run Multiparameter Prediction",
#                 type="primary",
#                 use_container_width=True
#             ):
#                 with st.spinner("Processing RF response..."):
#                     result = predictor.predict(
#                         frequency,
#                         magnitude
#                     )
#                 st.session_state.prediction_result = result
#                 st.session_state.prediction_curve = {
#                     "frequency": frequency,
#                     "magnitude": magnitude,
#                 }
#                 st.success(
#                     "Multiparameter inference completed."
#                 )
#             result = st.session_state.prediction_result
#             if result is not None:
#                 st.divider()
#                 st.subheader("Prediction Results")
#                 predictions = result["predictions"]
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric(
#                     "Temperature",
#                     f'{predictions["temperature"]["prediction"]:.3f} °C'
#                 )
#                 c2.metric(
#                     "Water Volume",
#                     f'{predictions["water_volume"]["prediction"]:.3f} mL'
#                 )
#                 c3.metric(
#                     "Relative Humidity",
#                     f'{predictions["relative_humidity"]["prediction"]:.3f} %'
#                 )
#                 st.write(
#                     f"Frequency points: {result['frequency_points']}"
#                 )
#                 st.write(
#                     f"Extracted features: {result['feature_count']}"
#                 )
#                 st.write(
#                     f"Model input shape: {result['input_shape']}"
#                 )
#                 st.subheader("Model Details")
#                 model_rows = []
#                 for target, data in predictions.items():
#                     model_rows.append({
#                         "Parameter": target,
#                         "Prediction": data["prediction"],
#                         "Unit": data["unit"],
#                         "Model": data["model_name"],
#                         "Experimental": data["experimental"],
#                         "Label Status": data["label_status"],
#                         "Training Samples": data["training_samples"],
#                     })
#                 st.dataframe(
#                     pd.DataFrame(model_rows),
#                     use_container_width=True,
#                     hide_index=True
#                 )
#                 st.subheader("Scientific Warnings")
#                 for warning in result["warnings"]:
#                     st.warning(warning)
#                 export_df = pd.DataFrame([
#                     {
#                         "parameter": target,
#                         "prediction": data["prediction"],
#                         "unit": data["unit"],
#                         "model": data["model_name"],
#                         "experimental": data["experimental"],
#                         "label_status": data["label_status"],
#                     }
#                     for target, data in predictions.items()
#                 ])
#                 st.download_button(
#                     "Download Prediction CSV",
#                     data=export_df.to_csv(index=False),
#                     file_name="crfid_predictions.csv",
#                     mime="text/csv",
#                 )
#         except Exception as exc:
#             st.error(
#                 f"Could not process uploaded data: {exc}"
#             )
# # ============================================================
# # PAGE 3: RF SIGNAL ANALYSIS
# # ============================================================
# elif page == "RF Signal Analysis":
#     st.title("RF Signal Analysis")
#     curve = st.session_state.prediction_curve
#     if curve is None:
#         st.info(
#             "Run an inference first to display its RF response."
#         )
#     else:
#         frequency = curve["frequency"]
#         magnitude = curve["magnitude"]
#         st.subheader("Frequency Response")
#         fig, ax = plt.subplots(figsize=(12, 5))
#         ax.plot(
#             frequency,
#             magnitude,
#             linewidth=1.5
#         )
#         ax.set_xlabel("Frequency (Hz)")
#         ax.set_ylabel("Magnitude (dB)")
#         ax.set_title("Chipless RFID Frequency Response")
#         ax.grid(True, alpha=0.3)
#         st.pyplot(fig)
#         plt.close(fig)
#         st.subheader("Signal Statistics")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric(
#             "Frequency Points",
#             len(frequency)
#         )
#         c2.metric(
#             "Maximum Magnitude",
#             f"{np.max(magnitude):.3f} dB"
#         )
#         c3.metric(
#             "Minimum Magnitude",
#             f"{np.min(magnitude):.3f} dB"
#         )
#         c4.metric(
#             "Mean Magnitude",
#             f"{np.mean(magnitude):.3f} dB"
#         )
#         st.write(
#             pd.DataFrame({
#                 "Frequency (Hz)": frequency,
#                 "Magnitude (dB)": magnitude,
#             })
#         )
# # ============================================================
# # PAGE 4: MODEL EVALUATION
# # ============================================================
# elif page == "Model Evaluation":
#     st.title("ML Model Evaluation")
#     st.write(
#         "Explore saved cross-validation and model-comparison results."
#     )
#     st.info(
#         "MAE, RMSE, MSE and R² must be calculated from "
#         "valid held-out or out-of-fold predictions. "
#         "Training-fit predictions are not validation results."
#     )
#     evaluation_files = load_evaluation_files()
#     if not evaluation_files:
#         st.warning(
#             "No comparison CSV files found in data/processed/. "
#             "Check your training output files."
#         )
#     else:
#         selected_file = st.selectbox(
#             "Select evaluation result",
#             evaluation_files,
#             format_func=lambda x: x.name
#         )
#         show_model_evaluation_file(selected_file)
#     st.divider()
#     st.subheader("Relative Humidity Evaluation")
#     rh_summary_path = (
#         PROCESSED_DATA
#         / "relative_humidity_summary.json"
#     )
#     if rh_summary_path.exists():
#         try:
#             with open(
#                 rh_summary_path,
#                 "r",
#                 encoding="utf-8"
#             ) as file:
#                 rh_summary = json.load(file)
#             st.json(rh_summary)
#             st.warning(
#                 "The RH model has only three independent curves "
#                 "with assumed labels. No independent validation "
#                 "metrics are available."
#             )
#         except Exception as exc:
#             st.error(
#                 f"Could not read RH summary: {exc}"
#             )
#     else:
#         st.warning(
#             "RH summary JSON was not found."
#         )
# # ============================================================
# # PAGE 5: DATASET EXPLORER
# # ============================================================
# elif page == "Dataset Explorer":
#     st.title("Dataset Explorer")
#     if not RAW_DATA.exists():
#         st.error(
#             f"Raw data directory not found: {RAW_DATA}"
#         )
#     else:
#         csv_files = list(RAW_DATA.glob("*.csv"))
#         if not csv_files:
#             st.warning("No CSV datasets found.")
#         else:
#             selected_dataset = st.selectbox(
#                 "Select dataset",
#                 csv_files,
#                 format_func=lambda x: x.name
#             )
#             try:
#                 # RH file is headerless in this project.
#                 if selected_dataset.name.lower() == "realhumid cf.csv":
#                     df = pd.read_csv(
#                         selected_dataset,
#                         header=None
#                     )
#                     if df.shape[1] == 4:
#                         df.columns = [
#                             "frequency_hz",
#                             "rh_curve_1",
#                             "rh_curve_2",
#                             "rh_curve_3",
#                         ]
#                 else:
#                     df = pd.read_csv(selected_dataset)
#                 st.subheader(selected_dataset.name)
#                 st.dataframe(
#                     df.head(20),
#                     use_container_width=True
#                 )
#                 c1, c2 = st.columns(2)
#                 c1.metric(
#                     "Rows",
#                     df.shape[0]
#                 )
#                 c2.metric(
#                     "Columns",
#                     df.shape[1]
#                 )
#                 st.subheader("Dataset Statistics")
#                 st.dataframe(
#                     df.describe(),
#                     use_container_width=True
#                 )
#             except Exception as exc:
#                 st.error(
#                     f"Could not load dataset: {exc}"
#                 )
# # ============================================================
# # PAGE 6: MODEL INFORMATION
# # ============================================================
# elif page == "Model Information":
#     st.title("Trained Model Information")
#     model_files = []
#     if SAVED_MODELS.exists():
#         model_files = list(
#             SAVED_MODELS.rglob("*.joblib")
#         )
#     if not model_files:
#         st.warning("No saved model artifacts found.")
#     else:
#         model_data = []
#         for path in model_files:
#             model_data.append({
#                 "Model File": path.name,
#                 "Target": path.parent.name,
#                 "Path": str(path.relative_to(ROOT)),
#             })
#         st.dataframe(
#             pd.DataFrame(model_data),
#             use_container_width=True,
#             hide_index=True
#         )
#     st.divider()
#     st.subheader("Selected Models")
#     selected_models = {
#         "Temperature": "temperature_extra_trees.joblib",
#         "Water Volume": "water_volume_svr_rbf.joblib",
#         "Relative Humidity": "relative_humidity_extra_trees.joblib",
#     }
#     for target, filename in selected_models.items():
#         st.write(f"**{target}:** `{filename}`")
#     st.divider()
#     st.subheader("Evaluation Limitations")
#     st.markdown(
#         """
#         - Temperature has only 10 independent measured curves.
#         - Water-volume data has only 5 independent curves.
#         - RH data has only 3 independent curves.
#         - RH labels are provisional assumptions.
#         - Water-volume LOOCV R² is negative.
#         - Model compatibility does not establish real-world accuracy.
#         """
#     )
# # ============================================================
# # FOOTER
# # ============================================================
# st.divider()
# st.caption(
#     "CRFID Multiparameter Sensing | "
#     "Machine Learning Research Dashboard | "
#     "Experimental results require independent validation."
# )





# ============================================================
# CHIPLESS RFID MULTIPARAMETER SENSING DASHBOARD
# File: app.py
# ============================================================

from pathlib import Path
import io
import json
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
# SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_STATE = {
    "prediction_result": None,
    "prediction_curve": None,
    "uploaded_csv_bytes": None,
    "uploaded_csv_name": None,
    "selected_magnitude": None,
    "last_prediction_source": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# CSV UPLOAD PERSISTENCE
# ============================================================

def save_uploaded_csv():
    """
    Save uploaded CSV bytes independently of the uploader widget.

    The uploader is rendered in the sidebar on every dashboard page.
    This prevents page navigation from removing the uploaded measurement.
    """

    uploaded = st.session_state.get("rf_csv_uploader_global")

    if uploaded is not None:

        new_bytes = uploaded.getvalue()

        if new_bytes != st.session_state.uploaded_csv_bytes:

            st.session_state.uploaded_csv_bytes = new_bytes
            st.session_state.uploaded_csv_name = uploaded.name

            # Reset prediction only when a new file is uploaded.
            st.session_state.selected_magnitude = None
            st.session_state.prediction_result = None
            st.session_state.prediction_curve = None
            st.session_state.last_prediction_source = None


def save_selected_magnitude():
    """Persist selected magnitude column."""

    st.session_state.selected_magnitude = (
        st.session_state.get("selected_magnitude_widget")
    )


def clear_uploaded_csv():
    """Clear the stored RF measurement and its predictions."""

    st.session_state.uploaded_csv_bytes = None
    st.session_state.uploaded_csv_name = None
    st.session_state.selected_magnitude = None
    st.session_state.prediction_result = None
    st.session_state.prediction_curve = None
    st.session_state.last_prediction_source = None

    # Reset uploader widget on next render.
    st.session_state.rf_csv_uploader_global = None


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


def prepare_rf_signal(df, frequency_column, magnitude_column):
    """
    Prepare RF signal:
    - Convert values to numeric.
    - Remove missing values.
    - Sort frequency.
    - Average duplicate frequencies.
    - Validate frequency ordering.
    """

    signal_df = pd.DataFrame({

        "frequency": pd.to_numeric(
            df[frequency_column],
            errors="coerce"
        ),

        "magnitude": pd.to_numeric(
            df[magnitude_column],
            errors="coerce"
        )

    })

    signal_df = signal_df.dropna()

    signal_df = signal_df.sort_values(
        by="frequency",
        ascending=True
    )

    signal_df = (
        signal_df
        .groupby("frequency", as_index=False)["magnitude"]
        .mean()
    )

    frequency = signal_df["frequency"].to_numpy(dtype=float)
    magnitude = signal_df["magnitude"].to_numpy(dtype=float)

    if len(frequency) < 2:
        raise ValueError("Insufficient valid frequency points.")

    if not np.all(np.diff(frequency) > 0):
        raise ValueError(
            "Frequency values must be strictly increasing."
        )

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Magnitude contains invalid values.")

    return frequency, magnitude


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
        raise ValueError(
            "No valid actual-predicted pairs."
        )

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    if len(actual) >= 2:

        r2 = r2_score(actual, predicted)

        explained_variance = explained_variance_score(
            actual,
            predicted
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
            actual,
            predicted
        ),

        "Explained Variance": explained_variance,

    }


def find_metric_columns(df):

    aliases = {

        "MAE": [
            "mae",
            "mean_absolute_error"
        ],

        "RMSE": [
            "rmse",
            "root_mean_squared_error"
        ],

        "MSE": [
            "mse",
            "mean_squared_error"
        ],

        "R2": [
            "r2",
            "r2_score",
            "r2_oof"
        ],

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
            "actual" in name
            or "true" in name
            or "ground_truth" in name
        ):

            actual_column = column

        if predicted_column is None and (
            "predicted" in name
            or "prediction" in name
            or "y_pred" in name
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

            display_value = (
                "N/A"
                if pd.isna(value)
                else f"{value:.4f}"
            )

            st.metric(
                label=name,
                value=display_value
            )


def show_model_evaluation_file(path):

    try:

        df = pd.read_csv(path)

    except Exception as exc:

        st.error(
            f"Could not read {path.name}: {exc}"
        )

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

        error_metrics = [
            col for col in ["MAE", "RMSE"]
            if col in chart_data.columns
        ]

        if error_metrics:

            st.bar_chart(
                chart_data[error_metrics],
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

            st.markdown(
                "**Metrics from actual-predicted pairs**"
            )

            display_metric_cards(metrics)

            # Actual versus predicted

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

            # Residual analysis

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

# IMPORTANT:
# The uploader must remain rendered on every page.
# Do not put it inside the Live Inference page condition.

st.sidebar.markdown("### RF Measurement CSV")

st.sidebar.file_uploader(
    "Upload once; reuse across dashboard pages",
    type=["csv"],
    key="rf_csv_uploader_global",
    on_change=save_uploaded_csv,
    help=(
        "The uploaded file is retained in this browser session "
        "while navigating between dashboard pages."
    ),
)

if st.session_state.uploaded_csv_name:

    st.sidebar.success(
        f"Stored: {st.session_state.uploaded_csv_name}"
    )

    st.sidebar.button(
        "Clear uploaded CSV",
        on_click=clear_uploaded_csv,
        use_container_width=True
    )

else:

    st.sidebar.caption(
        "No RF CSV stored in this session."
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

        st.success(
            "All three trained models loaded successfully."
        )

    except Exception as exc:

        predictor = None

        st.error(
            f"Model loading failed: {exc}"
        )

    st.subheader("Model Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🌡️ Temperature")

        st.success("Model Loaded")

        st.markdown("**Extra Trees Regressor**")

        st.metric("Training Curves", "10")

        st.metric("MAE", "9.493 °C")

        st.metric("RMSE", "11.108 °C")

        st.metric("R² (LOOCV)", "0.8201")

        st.caption(
            "R²-based explained variance: 82.01%. "
            "This is not classification accuracy."
        )

    with col2:

        st.markdown("### 💧 Water Volume")

        st.success("Model Loaded")

        st.markdown("**SVR RBF**")

        st.metric("Training Curves", "5")

        st.metric("MAE", "7.638 mL")

        st.metric("RMSE", "8.784 mL")

        st.metric("R² (LOOCV)", "-0.5431")

        st.caption(
            "Negative R²: this model performed worse than "
            "predicting the mean on these held-out folds."
        )

        st.warning(
            "Exploratory performance is poor; "
            "do not report a positive accuracy percentage."
        )

    with col3:

        st.markdown("### 💦 Relative Humidity")

        st.success("Model Loaded")

        st.markdown("**Extra Trees Regressor**")

        st.metric("Training Curves", "3")

        st.warning("Labels are assumed.")

        st.error(
            "No independent validation metrics "
            "or defensible accuracy percentage."
        )

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

        st.info(
            "No predictions available. Run Live Inference first."
        )

    st.divider()

    st.subheader("Evaluation Files")

    evaluation_files = load_evaluation_files()

    if evaluation_files:

        for path in evaluation_files:

            st.write(f"📄 {path.name}")

    else:

        st.info(
            "No model comparison CSV files found."
        )


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

        st.error(
            f"Could not load inference engine: {exc}"
        )

        st.stop()

    uploaded_csv_bytes = st.session_state.uploaded_csv_bytes

    if uploaded_csv_bytes is None:

        st.info(
            "Upload an RF CSV using the persistent uploader "
            "in the sidebar. Once uploaded, it remains available "
            "while you move between dashboard pages."
        )

    if uploaded_csv_bytes is not None:

        try:

            df = pd.read_csv(
                io.BytesIO(uploaded_csv_bytes)
            )

            st.success(
                f"Using stored CSV: "
                f"{st.session_state.uploaded_csv_name}"
            )

            st.caption(
                "This CSV is retained in session state. "
                "Navigate to other pages and return without "
                "uploading it again."
            )

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

            # Restore previous selection.

            if (
                st.session_state.selected_magnitude
                not in magnitude_columns
            ):

                st.session_state.selected_magnitude = (
                    magnitude_columns[0]
                )

            st.selectbox(

                "Select magnitude response curve",

                magnitude_columns,

                index=magnitude_columns.index(
                    st.session_state.selected_magnitude
                ),

                key="selected_magnitude_widget",

                on_change=save_selected_magnitude,

            )

            selected_magnitude = st.session_state.get(
                "selected_magnitude_widget",
                st.session_state.selected_magnitude
            )

            st.session_state.selected_magnitude = (
                selected_magnitude
            )

            # Prepare RF signal.

            frequency, magnitude = prepare_rf_signal(
                df,
                frequency_column,
                selected_magnitude
            )

            st.success(
                f"RF signal prepared successfully: "
                f"{len(frequency)} frequency points."
            )

            # Prediction button.

            if st.button(
                "Run Multiparameter Prediction",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Processing RF response..."
                ):

                    result = predictor.predict(
                        frequency,
                        magnitude
                    )

                st.session_state.prediction_result = result

                st.session_state.prediction_curve = {

                    "frequency": frequency.copy(),

                    "magnitude": magnitude.copy(),

                    "source_file": (
                        st.session_state.uploaded_csv_name
                    ),

                    "magnitude_column": selected_magnitude,

                }

                st.session_state.last_prediction_source = {

                    "file": st.session_state.uploaded_csv_name,

                    "magnitude_column": selected_magnitude,

                }

                st.success(
                    "Multiparameter inference completed."
                )

            # Display stored prediction.

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

    # Prefer the last uploaded measurement even if prediction
    # has not yet been run.

    curve = st.session_state.prediction_curve

    if curve is None and st.session_state.uploaded_csv_bytes:

        try:

            stored_df = pd.read_csv(
                io.BytesIO(
                    st.session_state.uploaded_csv_bytes
                )
            )

            frequency_column = find_frequency_column(stored_df)

            magnitude_columns = find_magnitude_columns(
                stored_df,
                frequency_column
            )

            selected = st.session_state.selected_magnitude

            if selected not in magnitude_columns and magnitude_columns:

                selected = magnitude_columns[0]

            if selected is not None:

                frequency, magnitude = prepare_rf_signal(
                    stored_df,
                    frequency_column,
                    selected
                )

                curve = {

                    "frequency": frequency,

                    "magnitude": magnitude,

                    "source_file": st.session_state.uploaded_csv_name,

                    "magnitude_column": selected,

                }

        except Exception as exc:

            st.error(
                f"Could not prepare stored RF signal: {exc}"
            )

    if curve is None:

        st.info(
            "Upload an RF CSV in the sidebar to display its "
            "frequency response."
        )

    else:

        frequency = curve["frequency"]

        magnitude = curve["magnitude"]

        st.caption(
            f"Source: {curve.get('source_file', 'Stored RF measurement')} | "
            f"Magnitude curve: "
            f"{curve.get('magnitude_column', 'selected curve')}"
        )

        st.subheader("Frequency Response")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            frequency,
            magnitude,
            linewidth=1.5
        )

        ax.set_xlabel("Frequency (Hz)")

        ax.set_ylabel("Magnitude (dB)")

        ax.set_title(
            "Chipless RFID Frequency Response"
        )

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

    st.subheader(
        "Selected Model: Out-of-Fold Performance"
    )

    st.caption(
        "These are exploratory leave-one-out cross-validation "
        "(LOOCV) results from the current small datasets. "
        "R² is not accuracy."
    )

    selected_cv = pd.DataFrame([

        {

            "Parameter": "Temperature",

            "Selected model": "Extra Trees",

            "Independent curves": 10,

            "MAE": 9.493333,

            "RMSE": 11.108150,

            "R²": 0.82013,

            "Interpretation": (
                "Exploratory; limited sample size"
            ),

        },

        {

            "Parameter": "Water volume",

            "Selected model": "SVR (RBF)",

            "Independent curves": 5,

            "MAE": 7.6379,

            "RMSE": 8.7838,

            "R²": -0.5431,

            "Interpretation": (
                "Poor predictive performance"
            ),

        },

        {

            "Parameter": "Relative humidity",

            "Selected model": "Extra Trees",

            "Independent curves": 3,

            "MAE": np.nan,

            "RMSE": np.nan,

            "R²": np.nan,

            "Interpretation": (
                "Assumed labels; no independent validation"
            ),

        },

    ])

    st.dataframe(
        selected_cv,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Do not label R² × 100 as model accuracy. "
        "Temperature R²=0.82013 means 82.013% of variance "
        "explained in these LOOCV predictions. "
        "Water-volume R² is negative. "
        "RH accuracy is not established."
    )

    # Model comparison files.

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

        csv_files = list(
            RAW_DATA.glob("*.csv")
        )

        if not csv_files:

            st.warning(
                "No CSV datasets found."
            )

        else:

            selected_dataset = st.selectbox(

                "Select dataset",

                csv_files,

                format_func=lambda x: x.name

            )

            try:

                # RH file is headerless in this project.

                if (
                    selected_dataset.name.lower()
                    == "realhumid cf.csv"
                ):

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

                    df = pd.read_csv(
                        selected_dataset
                    )

                st.subheader(
                    selected_dataset.name
                )

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

                st.subheader(
                    "Dataset Statistics"
                )

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

        st.warning(
            "No saved model artifacts found."
        )

    else:

        model_data = []

        for path in model_files:

            model_data.append({

                "Model File": path.name,

                "Target": path.parent.name,

                "Path": str(
                    path.relative_to(ROOT)
                ),

            })

        st.dataframe(
            pd.DataFrame(model_data),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("Selected Models")

    selected_models = {

        "Temperature": (
            "temperature_extra_trees.joblib"
        ),

        "Water Volume": (
            "water_volume_svr_rbf.joblib"
        ),

        "Relative Humidity": (
            "relative_humidity_extra_trees.joblib"
        ),

    }

    for target, filename in selected_models.items():

        st.write(
            f"**{target}:** `{filename}`"
        )

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
        - Independent experimental validation is required.
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
