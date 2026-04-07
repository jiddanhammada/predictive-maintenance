# ============================================================
# SECTION 1: IMPORT & KONFIGURASI
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, roc_auc_score,
                              classification_report,
                              confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
import mlflow.xgboost

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Cek device (CPU/GPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Path konfigurasi
DATA_PATH  = 'data/dataset_ml_ready.csv'
MODEL_DIR  = 'models'
OUTPUT_DIR = 'output'
MLFLOW_URI = 'mlruns'

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment('predictive_maintenance')

print("✅ Section 1 selesai — Setup OK")


# ============================================================
# SECTION 2: LOAD & SPLIT DATA
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape : {df.shape}")
print(f"Failure rate  : {df['machine_failure'].mean():.2%}")

FEATURE_COLS = [
    'type_encoded',
    'air_temperature_k',
    'process_temperature_k',
    'rotational_speed_rpm',
    'torque_nm',
    'tool_wear_min',
    'power_w',
    'temp_diff_k',
    'overstrain_value',
    'hdf_risk',
    'pwf_risk',
    'osf_risk',
    'tool_wear_ratio',
    'risk_score'
]

TARGET_COL = 'machine_failure'

# Validasi kolom
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    print(f"⚠️  Kolom tidak ditemukan: {missing_cols}")
    print(f"Kolom tersedia: {df.columns.tolist()}")
else:
    print("✅ Semua kolom tersedia")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Split 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain : {X_train.shape} | Failure: {y_train.mean():.2%}")
print(f"Test  : {X_test.shape}  | Failure: {y_test.mean():.2%}")

# Scaling — fit HANYA pada train
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SMOTE — hanya pada train
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print(f"\nSetelah SMOTE — Train: {X_train_res.shape}")
print(f"Class distribution  : {pd.Series(y_train_res).value_counts().to_dict()}")

# Simpan scaler
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
print(f"\n✅ Section 2 selesai — Scaler disimpan → {MODEL_DIR}/scaler.pkl")


# ============================================================
# SECTION 3: HELPER FUNCTION EVALUASI
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluasi model dan return dict metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'f1'     : round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  F1 Score : {metrics['f1']}")
    print(f"  ROC-AUC  : {metrics['roc_auc']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    return metrics

print("✅ Section 3 selesai — Helper function siap")


# ============================================================
# SECTION 4: XGBOOST + MLFLOW
# ============================================================
xgb_params = {
    'n_estimators'    : 300,
    'max_depth'       : 6,
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'random_state'    : 42,
    'eval_metric'     : 'logloss',
    'verbosity'       : 0
}

with mlflow.start_run(run_name='XGBoost'):

    # Train
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluasi
    xgb_metrics = evaluate_model(
        xgb_model, X_test_scaled, y_test, 'XGBoost'
    )

    # Log ke MLflow
    mlflow.log_params(xgb_params)
    mlflow.log_metrics(xgb_metrics)
    mlflow.log_artifact(f"{OUTPUT_DIR}/cm_xgboost.png")
    mlflow.xgboost.log_model(xgb_model, 'xgb_model')

    # Simpan lokal
    joblib.dump(xgb_model, f'{MODEL_DIR}/xgb_model.pkl')

print(f"\n✅ Section 4 selesai — XGBoost disimpan → {MODEL_DIR}/xgb_model.pkl")


# ============================================================
# SECTION 5: SHAP / EXPLAINABLE AI
# ============================================================
print("\nMenghitung SHAP values...")

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# 1. Summary plot bar → untuk PowerPoint
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=FEATURE_COLS,
    plot_type='bar',
    show=False
)
plt.title('Feature Importance (SHAP) — XGBoost')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Summary plot beeswarm → untuk PowerPoint
shap.summary_plot(
    shap_values, X_test_scaled,
    feature_names=FEATURE_COLS,
    show=False
)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Export SHAP values ke CSV → untuk Tableau
shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS)
shap_df['actual']       = y_test.values
shap_df['predicted']    = xgb_model.predict(X_test_scaled)
shap_df['failure_prob'] = xgb_model.predict_proba(X_test_scaled)[:, 1]
shap_df.to_csv(f'{OUTPUT_DIR}/shap_values.csv', index=False)

print(f"✅ Section 5 selesai")
print(f"   → {OUTPUT_DIR}/shap_bar.png")
print(f"   → {OUTPUT_DIR}/shap_beeswarm.png")
print(f"   → {OUTPUT_DIR}/shap_values.csv")


# ============================================================
# SECTION 6: LSTM (DEEP LEARNING - PYTORCH)
# ============================================================

# --- 6A. Definisi arsitektur LSTM ---
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


# --- 6B. Siapkan data untuk PyTorch ---
n_features = X_train_res.shape[1]

X_train_t = torch.tensor(
    X_train_res.reshape(-1, 1, n_features), dtype=torch.float32
).to(DEVICE)

X_test_t = torch.tensor(
    X_test_scaled.reshape(-1, 1, n_features), dtype=torch.float32
).to(DEVICE)

y_train_t = torch.tensor(
    y_train_res.values, dtype=torch.float32
).to(DEVICE)

y_test_t = torch.tensor(
    y_test.values, dtype=torch.float32
).to(DEVICE)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)


# --- 6C. Training loop ---
def train_lstm(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_lstm(model, X_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        y_prob = model(X_tensor).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = y_tensor.cpu().numpy().astype(int)
    return y_prob, y_pred, y_true


# --- 6D. Train dengan MLflow ---
with mlflow.start_run(run_name='LSTM_PyTorch'):

    lstm_model = LSTMClassifier(
        n_features=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    EPOCHS       = 50
    best_val_f1  = 0
    patience     = 7
    patience_cnt = 0
    train_losses = []
    val_f1s      = []

    print(f"\nTraining LSTM ({EPOCHS} epochs max, early stop patience={patience})...")

    for epoch in range(EPOCHS):
        train_loss = train_lstm(
            lstm_model, train_loader, optimizer, criterion, DEVICE
        )

        y_prob_val, y_pred_val, y_true_val = evaluate_lstm(
            lstm_model, X_test_t, y_test_t
        )
        val_f1 = f1_score(y_true_val, y_pred_val)

        train_losses.append(train_loss)
        val_f1s.append(val_f1)
        scheduler.step(train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(lstm_model.state_dict(), f'{MODEL_DIR}/lstm_best.pt')
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping di epoch {epoch+1}")
                break

    # Load best model
    lstm_model.load_state_dict(
        torch.load(f'{MODEL_DIR}/lstm_best.pt', map_location=DEVICE)
    )

    # Evaluasi final
    y_prob_lstm, y_pred_lstm, y_true_lstm = evaluate_lstm(
        lstm_model, X_test_t, y_test_t
    )

    lstm_metrics = {
        'f1'     : round(float(f1_score(y_true_lstm, y_pred_lstm)), 4),
        'roc_auc': round(float(roc_auc_score(y_true_lstm, y_prob_lstm)), 4),
    }

    print(f"\n✅ LSTM Final — F1: {lstm_metrics['f1']} | ROC-AUC: {lstm_metrics['roc_auc']}")
    print(classification_report(
        y_true_lstm, y_pred_lstm,
        target_names=['Normal', 'Failure']
    ))

    # Plot training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, color='steelblue', linewidth=1.5)
    axes[0].set_title('Training Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].plot(val_f1s, color='darkorange', linewidth=1.5)
    axes[1].set_title('Validation F1 per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/lstm_history.png', dpi=150)
    plt.close()

    # Log ke MLflow
    mlflow.log_params({
        'hidden_size': 64,
        'num_layers' : 2,
        'dropout'    : 0.3,
        'lr'         : 0.001,
        'batch_size' : 64,
        'framework'  : 'PyTorch'
    })
    mlflow.log_metrics(lstm_metrics)
    mlflow.log_artifact(f'{OUTPUT_DIR}/lstm_history.png')

    # Simpan model
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'n_features'      : n_features,
        'hidden_size'     : 64,
        'num_layers'      : 2,
        'dropout'         : 0.3
    }, f'{MODEL_DIR}/lstm_model.pt')

print(f"\n✅ Section 6 selesai — LSTM disimpan → {MODEL_DIR}/lstm_model.pt")


# ============================================================
# SECTION 7: PERBANDINGAN MODEL & SIMPAN METADATA
# ============================================================
results = pd.DataFrame({
    'Model'  : ['XGBoost', 'LSTM'],
    'F1'     : [xgb_metrics['f1'],      lstm_metrics['f1']],
    'ROC_AUC': [xgb_metrics['roc_auc'], lstm_metrics['roc_auc']]
})

print("\n" + "="*40)
print("  PERBANDINGAN MODEL")
print("="*40)
print(results.to_string(index=False))

best_model_name = results.loc[results['F1'].idxmax(), 'Model']
print(f"\n🏆 Best model: {best_model_name}")

# Simpan metadata
metadata = {
    'best_model'  : best_model_name,
    'feature_cols': FEATURE_COLS,
    'xgb_metrics' : xgb_metrics,
    'lstm_metrics': lstm_metrics,
    'scaler_path' : f'{MODEL_DIR}/scaler.pkl',
    'xgb_path'    : f'{MODEL_DIR}/xgb_model.pkl',
    'lstm_path'   : f'{MODEL_DIR}/lstm_model.pt',
}

with open(f'{OUTPUT_DIR}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Section 7 selesai — Metadata disimpan → {OUTPUT_DIR}/model_metadata.json")
print("\n" + "="*40)
print("  SEMUA OUTPUT TERSIMPAN")
print("="*40)
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  → output/{f}")
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"  → models/{f}")