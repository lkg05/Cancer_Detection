# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("🩺 Breast Cancer Prediction App")
st.write("Compare XGBoost and Deep Neural Network (DNN) performance on breast cancer prediction.")

# ----------------- MODEL LOADING -----------------
MODEL_PATH = "models"
SCALER_FILE = f"{MODEL_PATH}/minmax_scaler.joblib"
RFE_FILE = f"{MODEL_PATH}/rfe_support.joblib"
XGB_FILE = f"{MODEL_PATH}/xgb_model.json"
DNN_FILE = f"{MODEL_PATH}/dnn_saved_model.keras"

scaler = joblib.load(SCALER_FILE)
rfe = joblib.load(RFE_FILE)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_FILE)

try:
    dnn_model = load_model(DNN_FILE)
    dnn_available = True
    st.success("✅ DNN model loaded successfully.")
except:
    dnn_model = None
    dnn_available = False
    st.warning("⚠ DNN model not available. Only XGBoost predictions will be used.")

N_FEATURES = 30

# ----------------- SINGLE PATIENT PREDICTION -----------------
st.sidebar.header("🔹 Single Patient Prediction")
features = [st.sidebar.number_input(f"Feature {i+1}", value=0.0, format="%.4f") for i in range(N_FEATURES)]

if st.sidebar.button("Predict Single Patient"):
    input_array = np.array([features])
    input_scaled = scaler.transform(input_array)
    input_reduced = input_scaled[:, rfe]

    xgb_proba = xgb_model.predict_proba(input_reduced)[0][1]
    xgb_pred = "Malignant" if xgb_proba >= 0.5 else "Benign"

    if dnn_available:
        dnn_proba = dnn_model.predict(input_reduced, verbose=0)[0][0]
        dnn_pred = "Malignant" if dnn_proba >= 0.5 else "Benign"
    else:
        dnn_proba = None
        dnn_pred = "N/A"

    st.subheader("Prediction Result for Single Patient")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("XGBoost Prediction", xgb_pred, f"{xgb_proba:.4f}")
    with col2:
        if dnn_available:
            st.metric("DNN Prediction", dnn_pred, f"{dnn_proba:.4f}")
        else:
            st.warning("⚠ DNN model not available.")

# ----------------- BATCH PREDICTION -----------------
st.header("📂 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with feature columns", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded file with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None

    if df is not None:
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # ----------------- EDA: Missing Values -----------------
        st.write("**📊 Exploratory Data Analysis (EDA)**")
        missing = df.isnull().sum()
        st.write("**Missing Values:**")
        if missing.sum() > 0:
            st.dataframe(missing[missing>0])
        else:
            st.write("No missing values.")

        # ----------------- Prepare features -----------------
        feature_columns = [col for col in df.columns if col.lower() not in ["actual_label","diagnosis","target","label"]]
        N_USED = min(len(feature_columns), N_FEATURES)
        df_features = df[feature_columns[:N_USED]].astype(float)

        input_scaled = scaler.transform(df_features.values)
        input_reduced = input_scaled[:, rfe[:N_USED]]

        # ----------------- Predictions -----------------
        xgb_proba = xgb_model.predict_proba(input_reduced)[:,1]
        xgb_pred = ["Malignant" if p>=0.5 else "Benign" for p in xgb_proba]

        if dnn_available:
            dnn_proba = dnn_model.predict(input_reduced, verbose=0).flatten()
            dnn_pred = ["Malignant" if p>=0.5 else "Benign" for p in dnn_proba]

        results_df = df_features.copy()
        results_df["XGBoost_Prediction"] = xgb_pred
        results_df["XGBoost_Confidence"] = np.round(xgb_proba,4)
        if dnn_available:
            results_df["DNN_Prediction"] = dnn_pred
            results_df["DNN_Confidence"] = np.round(dnn_proba,4)

        # ----------------- Actual_Label if exists -----------------
        target_candidates = [c for c in df.columns if c.lower() in ["diagnosis","actual_label","target","label"]]
        target_col = target_candidates[0] if target_candidates else None
        if target_col:
            results_df["Actual_Label"] = df[target_col]

        st.subheader("🧠 Prediction Results")
        st.dataframe(results_df)

        # ----------------- Download CSV -----------------
        csv = results_df.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

        # ----------------- VISUALIZATIONS -----------------
        st.header("📊 Model Performance Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            counts_xgb = results_df["XGBoost_Prediction"].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(counts_xgb, labels=counts_xgb.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
            ax1.set_title("XGBoost Prediction Distribution")
            st.pyplot(fig1)

        if dnn_available:
            with col2:
                counts_dnn = results_df["DNN_Prediction"].value_counts()
                fig2, ax2 = plt.subplots()
                ax2.pie(counts_dnn, labels=counts_dnn.index, autopct='%1.1f%%', colors=['#99ff99','#ffcc99'])
                ax2.set_title("DNN Prediction Distribution")
                st.pyplot(fig2)

        st.subheader("📈 Average Confidence Comparison")
        avg_conf = {"XGBoost": results_df["XGBoost_Confidence"].mean()}
        if dnn_available:
            avg_conf["DNN"] = results_df["DNN_Confidence"].mean()
        avg_df = pd.DataFrame.from_dict(avg_conf, orient='index', columns=['Average_Confidence'])
        st.bar_chart(avg_df)

        # ----------------- CONFUSION MATRIX & ROC -----------------
        y_true = None
        if target_col:
            y_data = df[target_col].copy()
            if y_data.dtype=='O':
                y_data = y_data.str.strip().map({"B":0,"M":1,"Benign":0,"Malignant":1})
            y_true = y_data.dropna()

        if y_true is not None and len(y_true)>0:
            st.subheader("📉 Confusion Matrices & ROC Curves")
            y_pred_xgb = results_df.loc[y_true.index, "XGBoost_Prediction"].map({"Benign":0,"Malignant":1})
            cm_xgb = confusion_matrix(y_true, y_pred_xgb)
            fig3, ax3 = plt.subplots()
            ConfusionMatrixDisplay(cm_xgb, display_labels=["Benign","Malignant"]).plot(ax=ax3)
            ax3.set_title("XGBoost Confusion Matrix")
            st.pyplot(fig3)

            if dnn_available:
                y_pred_dnn = results_df.loc[y_true.index, "DNN_Prediction"].map({"Benign":0,"Malignant":1})
                cm_dnn = confusion_matrix(y_true, y_pred_dnn)
                fig4, ax4 = plt.subplots()
                ConfusionMatrixDisplay(cm_dnn, display_labels=["Benign","Malignant"]).plot(ax=ax4)
                ax4.set_title("DNN Confusion Matrix")
                st.pyplot(fig4)

            fig5, ax5 = plt.subplots()
            fpr_xgb, tpr_xgb, _ = roc_curve(y_true, results_df.loc[y_true.index,"XGBoost_Confidence"])
            auc_xgb = auc(fpr_xgb, tpr_xgb)
            ax5.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.2f})")
            if dnn_available:
                fpr_dnn, tpr_dnn, _ = roc_curve(y_true, results_df.loc[y_true.index,"DNN_Confidence"])
                auc_dnn = auc(fpr_dnn, tpr_dnn)
                ax5.plot(fpr_dnn, tpr_dnn, label=f"DNN (AUC={auc_dnn:.2f})")
            ax5.plot([0,1],[0,1],"k--")
            ax5.set_xlabel("False Positive Rate")
            ax5.set_ylabel("True Positive Rate")
            ax5.set_title("ROC Curve Comparison")
            ax5.legend()
            st.pyplot(fig5)
        else:
            st.warning("⚠ Target column missing or has invalid/missing labels. Confusion Matrix & ROC cannot be generated.")

        # ----------------- XGBoost Feature Importance -----------------
        st.subheader("📍 XGBoost Feature Importance")
        feature_names_reduced = np.array(feature_columns[:N_USED])[rfe[:N_USED]]
        importance = xgb_model.feature_importances_
        fig6, ax6 = plt.subplots(figsize=(10,6))
        ax6.barh(feature_names_reduced, importance)
        ax6.set_title("Top Features Selected by RFE (XGBoost Importance)")
        st.pyplot(fig6)

        # ----------------- PDF REPORT -----------------
        st.subheader("📄 Download PDF Report")
        doctor_name = st.text_input("Doctor Name")
        doctor_id = st.text_input("Doctor ID")
        if st.button("Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial","B",16)
            pdf.cell(0,10,"Breast Cancer Prediction Report", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial","",12)
            pdf.multi_cell(0,8,"This report contains predictions using XGBoost and DNN models.")
            pdf.ln(5)
            pdf.cell(0,8,f"Doctor Name: {doctor_name}", ln=True)
            pdf.cell(0,8,f"Doctor ID: {doctor_id}", ln=True)
            pdf.cell(0,8,"Signature: ____________________", ln=True)
            pdf.ln(5)

            # Add table of predictions
            pdf.set_font("Arial","B",10)
            col_width = pdf.w / (len(results_df.columns) + 1)
            pdf.set_font("Arial","",8)
            pdf.add_page()
            for i, row in results_df.iterrows():
                line = ""
                for col in results_df.columns:
                    line += f"{col}: {row[col]}  "
                pdf.multi_cell(0,6,line)

            # Add plots
            plots = {
                "XGBoost Prediction Distribution": fig1,
                "ROC Curve Comparison": fig5,
                "XGBoost Feature Importance": fig6
            }
            if dnn_available:
                plots["DNN Prediction Distribution"] = fig2
            if y_true is not None and len(y_true) > 0:
                plots["XGBoost Confusion Matrix"] = fig3
                if dnn_available:
                    plots["DNN Confusion Matrix"] = fig4

            for title, fig in plots.items():
                pdf.add_page()
                pdf.set_font("Arial","B",14)
                pdf.cell(0,10,title, ln=True, align="C")
                pdf.ln(5)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches='tight')
                    pdf.image(tmpfile.name, x=15, w=pdf.w-30)
                    tmpfile.close()
                    os.remove(tmpfile.name)
                plt.close(fig)

            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button("Download PDF Report", pdf_bytes, "final_report.pdf", "application/pdf")
