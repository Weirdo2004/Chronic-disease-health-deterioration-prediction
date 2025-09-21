import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import requests

# =====================================================
# 1. Transformer Encoder (must match training definition)
# =====================================================
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x  # embedding only

# =====================================================
# 2. Load Models + Artifacts
# =====================================================
ts_features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk'
]
static_features = ['Sex', 'Age', 'Education', 'Income']

transformer = TransformerEncoder(input_dim=len(ts_features))
state_dict = torch.load("transformer_encoder.pt", map_location="cpu")
transformer.load_state_dict(state_dict, strict=False)  # ignore fc_out mismatch if needed
transformer.eval()

xgb_model = joblib.load("xgb_hybrid_model.joblib")

static_mean = np.load("scaler_static_mean.npy")
static_std = np.load("scaler_static_std.npy")

# =====================================================
# 3. Helper Functions
# =====================================================
def get_embeddings(model, X_seq):
    model.eval()
    with torch.no_grad():
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32)
        x = model.input_proj(X_seq_torch)
        x = model.transformer(x)
        x = x.transpose(1, 2)
        x = model.pool(x).squeeze(-1)
        return x.numpy()

def preprocess_and_predict(patient_df: pd.DataFrame, patient_id: str):
    """Takes patient history dataframe and returns prediction + probability."""

    group = patient_df[patient_df["patient_id"] == patient_id].sort_values("day").copy()

    # Encode Sex as numeric (male=1, female=0)
    if "Sex" in group.columns:
        group["Sex"] = group["Sex"].astype(str).str.lower().map({"male": 1, "female": 0})

    # Build sequence (pad/trim to 180 days)
    seq = group[ts_features].values
    if len(seq) < 180:
        seq = np.vstack([seq, np.zeros((180 - len(seq), len(ts_features)))])
    elif len(seq) > 180:
        seq = seq[:180]

    # Static features
    static = group[static_features].iloc[0].values.astype(float)
    static_scaled = (static - static_mean) / (static_std + 1e-6)

    # Get transformer embedding
    emb = get_embeddings(transformer, seq[np.newaxis, :, :])

    # Final input for XGBoost
    X_final = np.hstack([emb, static_scaled.reshape(1, -1)])

    # Predict probability
    proba = xgb_model.predict_proba(X_final)[:, 1][0]
    pred = int(proba > 0.5)

    return pred, proba


def call_gemini_api(temp_data: dict, prediction: int, disease: str):
    GEMINI_API_KEY = "AIzaSyBSxSAksIjBg5RzoU0RK3HBO0aTIiP0SUw"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
You are a medical assistant.
A patient has been analyzed for **{disease}**.

Prediction result: {"High risk (1)" if prediction==1 else "Low risk (0)"}.

Temporary patient lifestyle/environment data:
- Medications: {temp_data.get("medicatios(comma saperated all medicines)", "N/A")}
- Exercise routine: {temp_data.get("exercise routine", "N/A")}
- Diet habits: {temp_data.get("diet habits", "N/A")}
- Stress levels: {temp_data.get("stress levels", "N/A")}
- Calories burned: {temp_data.get("calories burned", "N/A")}
- Seasonal allergies: {temp_data.get("Seasonal allergies", "N/A")}
- Air quality: {temp_data.get("air quality", "N/A")}
- Sleep pattern: {temp_data.get("sleep pattern", "N/A")}

Task:
1. Explain possible causes for this prediction.
2. Suggest preventive lifestyle improvements.
3. Do NOT prescribe medication.
4. Keep the explanation simple for patients.

Return a structured explanation for frontend display.
"""

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# =====================================================
# 4. Main Workflow
# =====================================================
TEMP_COLS = [
    "medicatios(comma saperated all medicines)", "exercise routine", "diet habits",
    "stress levels", "calories burned", "Seasonal allergies", "air quality", "sleep pattern"
]

def process_user_input(user_csv_path: str, patient_data_path: str, disease="Heart Disease"):
    user_row = pd.read_csv(user_csv_path)
    patient_id = user_row["patient_id"].iloc[0]

    # Append to t1.csv (permanent data), keep only permanent cols
    df = pd.read_csv(patient_data_path)
    permanent_cols = [c for c in user_row.columns if c not in TEMP_COLS]
    user_row_perm = user_row[permanent_cols]
    df = pd.concat([df, user_row_perm], ignore_index=True)
    df.to_csv(patient_data_path, index=False)

    # Predict
    pred, proba = preprocess_and_predict(df, patient_id)

    # Extract temporary lifestyle/environment data
    temp_data = user_row.iloc[0][TEMP_COLS].to_dict()

    # Get Gemini explanation
    explanation = call_gemini_api(temp_data, pred, disease)

    return {
        "patient_id": patient_id,
        "prediction": pred,
        "probability": proba,
        "gemini_explanation": explanation
    }

# =====================================================
# 5. Example Run
# =====================================================
if __name__ == "__main__":
    result = process_user_input(
        user_csv_path="dd.csv",       # new incoming row with TEMP cols
        patient_data_path="test_zero.csv",   # ongoing patient history
        disease="Heart Health Deterioration"
    )
    print("Final Result for Frontend:")
    print(result)
