import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Maintenance XAI", layout="wide", page_icon="🔧")

st.title("🔧 Predictive Maintenance with Explainable AI")
st.markdown("**Machine Learning + Deep Learning + SHAP Explanation**")

# ====================== LOAD ASSETS ======================
@st.cache_resource
def load_assets():
    with open('output/model_metadata.json') as f:
        metadata = json.load(f)
    
    scaler = joblib.load('models/scaler.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')

    # LSTM Class
    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :]).squeeze(1)

    checkpoint = torch.load('models/lstm_model.pt', map_location='cpu')
    lstm_model = LSTMClassifier(
        n_features=checkpoint['n_features'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()

    return metadata, scaler, xgb_model, lstm_model

metadata, scaler, xgb_model, lstm_model = load_assets()
FEATURE_COLS = metadata['feature_cols']

# ====================== HELPER FUNCTIONS ======================
TYPE_MAP = {'H': 0, 'L': 1, 'M': 2}
OSF_THRESH = {'L': 11000, 'M': 12000, 'H': 13000}

def compute_features(air_temp, proc_temp, rpm, torque, tool_wear, product_type):
    power_w = torque * (rpm * 2 * np.pi / 60)
    temp_diff = proc_temp - air_temp
    overstrain = tool_wear * torque
    hdf_risk = int(temp_diff < 8.6 and rpm < 1380)
    pwf_risk = int(power_w < 3500 or power_w > 9000)
    osf_risk = int(overstrain > OSF_THRESH[product_type])
    wear_ratio = tool_wear / 240
    risk_score = hdf_risk + pwf_risk + osf_risk

    features = np.array([[
        TYPE_MAP[product_type], air_temp, proc_temp, rpm, torque, tool_wear,
        power_w, temp_diff, overstrain, hdf_risk, pwf_risk, osf_risk,
        wear_ratio, risk_score
    ]])

    return features, {
        'Power (W)': round(power_w, 1),
        'Temp Diff (K)': round(temp_diff, 2),
        'Overstrain Value': round(overstrain, 1),
        'HDF Risk': hdf_risk,
        'PWF Risk': pwf_risk,
        'OSF Risk': osf_risk,
        'Tool Wear Ratio': round(wear_ratio, 3),
        'Risk Score': risk_score,
    }

def predict(features_raw, scaler, xgb_model, lstm_model):
    features_scaled = scaler.transform(features_raw)
    
    # XGBoost
    xgb_prob = xgb_model.predict_proba(features_scaled)[0][1]
    xgb_pred = int(xgb_prob >= 0.5)
    
    # LSTM
    X_tensor = torch.tensor(features_scaled.reshape(-1, 1, features_scaled.shape[1]), dtype=torch.float32)
    with torch.no_grad():
        lstm_prob = lstm_model(X_tensor).item()
    lstm_pred = int(lstm_prob >= 0.5)
    
    ensemble_prob = (xgb_prob + lstm_prob) / 2
    ensemble_pred = int(ensemble_prob >= 0.5)
    
    return {
        'xgb_prob': round(xgb_prob, 4), 'xgb_pred': xgb_pred,
        'lstm_prob': round(lstm_prob, 4), 'lstm_pred': lstm_pred,
        'ensemble_prob': round(ensemble_prob, 4), 'ensemble_pred': ensemble_pred,
    }

# ====================== SIDEBAR ======================
st.sidebar.title("🔧 Input Sensor Mesin")
product_type = st.sidebar.selectbox("Product Type", ['L', 'M', 'H'])
air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1)
proc_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1)
rpm = st.sidebar.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
torque = st.sidebar.slider("Torque (Nm)", 3.8, 76.6, 40.0, step=0.1)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 253, 100)

if st.sidebar.button("🔍 Prediksi Sekarang", use_container_width=True):
    features_raw, derived = compute_features(air_temp, proc_temp, rpm, torque, tool_wear, product_type)
    results = predict(features_raw, scaler, xgb_model, lstm_model)

    # ====================== HASIL PREDIKSI ======================
    st.subheader("📊 Hasil Prediksi")
    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost", f"{results['xgb_prob']:.1%}")
    col2.metric("LSTM", f"{results['lstm_prob']:.1%}")
    col3.metric("Ensemble (Final)", f"{results['ensemble_prob']:.1%}")

    if results['ensemble_pred'] == 1:
        st.error("🔴 HIGH RISK — Segera lakukan maintenance!")
    else:
        st.success("🟢 LOW RISK — Mesin aman untuk dioperasikan.")

    # ====================== XAI (SHAP) ======================
    st.subheader("🧠 Explainable AI — Mengapa Model Memberi Prediksi Ini?")
    
    # Summary plot (sudah ada dari modelling.py)
    st.image("output/shap_bar.png", caption="Feature Importance (SHAP Bar)")
    st.image("output/shap_beeswarm.png", caption="SHAP Beeswarm Plot")

    # Waterfall plot untuk input saat ini
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(scaler.transform(features_raw))

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            feature_names=FEATURE_COLS,
            data=features_raw[0]
        ),
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("📌 **Merah** = meningkatkan risiko failure | **Biru** = menurunkan risiko failure")

else:
    st.info("👈 Masukkan nilai sensor di sidebar lalu klik **Prediksi Sekarang**")

st.divider()
st.caption("Project Predictive Maintenance | ML + Deep Learning + XAI + MLflow")