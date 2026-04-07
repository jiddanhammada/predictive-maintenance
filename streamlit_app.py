import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import joblib
import json
import shap
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_assets():
    with open('output/model_metadata.json') as f:
        metadata = json.load(f)
    scaler    = joblib.load('models/scaler.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out.squeeze(1)

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

# ============================================================
# HELPER FUNCTIONS
# ============================================================
TYPE_MAP   = {'H': 0, 'L': 1, 'M': 2}
OSF_THRESH = {'L': 11000, 'M': 12000, 'H': 13000}

def compute_features(air_temp, proc_temp, rpm, torque, tool_wear, product_type):
    power_w    = torque * (rpm * 2 * np.pi / 60)
    temp_diff  = proc_temp - air_temp
    overstrain = tool_wear * torque
    hdf_risk   = int(temp_diff < 8.6 and rpm < 1380)
    pwf_risk   = int(power_w < 3500 or power_w > 9000)
    osf_risk   = int(overstrain > OSF_THRESH[product_type])
    wear_ratio = tool_wear / 240
    risk_score = hdf_risk + pwf_risk + osf_risk

    features = np.array([[
        TYPE_MAP[product_type], air_temp, proc_temp, rpm, torque, tool_wear,
        power_w, temp_diff, overstrain, hdf_risk, pwf_risk, osf_risk,
        wear_ratio, risk_score
    ]])

    return features, {
        'Power (W)'       : round(power_w, 1),
        'Temp Diff (K)'   : round(temp_diff, 2),
        'Overstrain Value': round(overstrain, 1),
        'HDF Risk'        : hdf_risk,
        'PWF Risk'        : pwf_risk,
        'OSF Risk'        : osf_risk,
        'Tool Wear Ratio' : round(wear_ratio, 3),
        'Risk Score'      : risk_score,
    }

def predict(features_raw, scaler, xgb_model, lstm_model):
    features_scaled = scaler.transform(features_raw)

    xgb_prob = xgb_model.predict_proba(features_scaled)[0][1]
    xgb_pred = int(xgb_prob >= 0.5)

    X_tensor = torch.tensor(
        features_scaled.reshape(-1, 1, features_scaled.shape[1]),
        dtype=torch.float32
    )
    with torch.no_grad():
        lstm_prob = lstm_model(X_tensor).item()
    lstm_pred = int(lstm_prob >= 0.5)

    ensemble_prob = (xgb_prob + lstm_prob) / 2
    ensemble_pred = int(ensemble_prob >= 0.5)

    return {
        'xgb_prob'     : round(xgb_prob, 4),
        'xgb_pred'     : xgb_pred,
        'lstm_prob'    : round(lstm_prob, 4),
        'lstm_pred'    : lstm_pred,
        'ensemble_prob': round(ensemble_prob, 4),
        'ensemble_pred': ensemble_pred,
    }

def get_risk_level(prob):
    if prob >= 0.7:
        return '🔴 HIGH RISK', 'red'
    elif prob >= 0.4:
        return '🟡 MEDIUM RISK', 'orange'
    else:
        return '🟢 LOW RISK', 'green'

# ============================================================
# SIDEBAR INPUT
# ============================================================
st.sidebar.title("🔧 Input Sensor")
st.sidebar.markdown("Masukkan nilai sensor mesin:")

product_type = st.sidebar.selectbox(
    "Product Type", ['L', 'M', 'H'],
    help="L=Low (50%), M=Medium (30%), H=High (20%)"
)
air_temp  = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1)
proc_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1)
rpm       = st.sidebar.slider("Rotational Speed (rpm)", 1168, 2886, 1500)
torque    = st.sidebar.slider("Torque (Nm)", 3.8, 76.6, 40.0, step=0.1)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 253, 100)

predict_btn = st.sidebar.button("🔍 Prediksi Sekarang", use_container_width=True)

# ============================================================
# MAIN DASHBOARD
# ============================================================
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("Sistem prediksi kerusakan mesin berbasis **Machine Learning + Deep Learning + Explainable AI**")
st.divider()

if predict_btn:
    features_raw, derived = compute_features(
        air_temp, proc_temp, rpm, torque, tool_wear, product_type
    )
    results = predict(features_raw, scaler, xgb_model, lstm_model)

    st.subheader("📊 Hasil Prediksi")
    col1, col2, col3 = st.columns(3)
    risk_label, risk_color = get_risk_level(results['ensemble_prob'])

    col1.metric("XGBoost Probability",  f"{results['xgb_prob']:.1%}",
                "Failure" if results['xgb_pred'] == 1 else "Normal")
    col2.metric("LSTM Probability",     f"{results['lstm_prob']:.1%}",
                "Failure" if results['lstm_pred'] == 1 else "Normal")
    col3.metric("Ensemble Probability", f"{results['ensemble_prob']:.1%}",
                "Failure" if results['ensemble_pred'] == 1 else "Normal")

    st.markdown(
        f"<h2 style='text-align:center; color:{risk_color}'>{risk_label}</h2>",
        unsafe_allow_html=True
    )

    if results['ensemble_pred'] == 1:
        st.error("⚠️ **Rekomendasi:** Segera lakukan inspeksi dan maintenance pada mesin ini.")
    else:
        st.success("✅ **Rekomendasi:** Mesin dalam kondisi normal. Lanjutkan operasi.")

    st.divider()

    st.subheader("⚙️ Derived Features")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Nilai yang Dihitung Otomatis:**")
        st.dataframe(
            pd.DataFrame(derived.items(), columns=['Feature', 'Value']),
            use_container_width=True, hide_index=True
        )

    with col_b:
        st.markdown("**Risk Flags:**")
        for flag, key in [
            ('Heat Dissipation Failure (HDF)', 'HDF Risk'),
            ('Power Failure (PWF)',            'PWF Risk'),
            ('Overstrain Failure (OSF)',       'OSF Risk'),
        ]:
            if derived[key] == 1:
                st.error(f"🔴 {flag}: ACTIVE")
            else:
                st.success(f"🟢 {flag}: OK")
        st.metric("Total Risk Score", f"{derived['Risk Score']} / 3")

    st.divider()

    st.subheader("🧠 Explainable AI — Kenapa Model Prediksi Ini?")
    features_scaled = scaler.transform(features_raw)
    explainer       = shap.TreeExplainer(xgb_model)
    shap_values     = explainer.shap_values(features_scaled)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            feature_names=FEATURE_COLS,
            data=features_scaled[0]
        ),
        show=False,
        max_display=14
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("📌 Bar merah = fitur yang **meningkatkan** risiko failure. Bar biru = fitur yang **menurunkan** risiko failure.")

else:
    st.info("👈 Masukkan nilai sensor di sidebar kiri, lalu klik **Prediksi Sekarang**.")
    st.divider()

    tab1, tab2 = st.tabs(["📈 Model Performance", "📂 Dataset Overview"])

    with tab1:
        st.subheader("Perbandingan Model")
        st.dataframe(pd.DataFrame({
            'Model'   : ['XGBoost', 'LSTM (PyTorch)', 'Ensemble'],
            'F1 Score': [
                metadata['xgb_metrics']['f1'],
                metadata['lstm_metrics']['f1'],
                round((metadata['xgb_metrics']['f1'] +
                       metadata['lstm_metrics']['f1']) / 2, 4)
            ],
            'ROC-AUC' : [
                metadata['xgb_metrics']['roc_auc'],
                metadata['lstm_metrics']['roc_auc'],
                round((metadata['xgb_metrics']['roc_auc'] +
                       metadata['lstm_metrics']['roc_auc']) / 2, 4)
            ]
        }), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**SHAP Feature Importance:**")
            st.image('output/shap_bar.png', use_container_width=True)
        with col2:
            st.markdown("**SHAP Beeswarm Plot:**")
            st.image('output/shap_beeswarm.png', use_container_width=True)

    with tab2:
        st.subheader("Dataset Overview")
        df_display = pd.read_csv('data/dataset_ml_ready.csv')
        st.markdown(f"**Shape:** {df_display.shape[0]:,} baris × {df_display.shape[1]} kolom")
        st.markdown(f"**Failure Rate:** {df_display['machine_failure'].mean():.2%}")
        st.dataframe(df_display.head(10), use_container_width=True)