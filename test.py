import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.utils.data import Dataset, DataLoader

# =====================================================
# 1. Define the Transformer Model (same as training)
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
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.fc_out(x)
        return x.squeeze(-1)

# =====================================================
# 2. Load Models and Scalers
# =====================================================
def load_models():
    """Load all saved models and scalers"""
    # Load XGBoost model
    xgb_model = joblib.load('xgb_hybrid_model.joblib')
    
    # Load transformer
    transformer = TransformerEncoder(input_dim=17)
    transformer.load_state_dict(torch.load('transformer_encoder.pt'))
    transformer.eval()  # Set to evaluation mode
    
    # Load scalers
    static_mean = np.load('scaler_static_mean.npy')
    static_std = np.load('scaler_static_std.npy')
    
    return xgb_model, transformer, static_mean, static_std

# =====================================================
# 3. Preprocessing Function (FIXED)
# =====================================================
def preprocess_new_data(df):
    """Preprocess new data in the same way as training"""
    # Time-series features (per day)
    ts_features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk'
    ]

    # Static features (per patient)
    static_features = ['Sex', 'Age', 'Education', 'Income']
    
    # Encode Sex (if needed)
    if 'Sex' in df.columns:
        df['Sex'] = (df['Sex'].astype(str).str.lower() == 'male').astype(int)
    
    # Group by patient_id
    predictions = {}
    
    for patient_id, group in df.groupby('patient_id'):
        group = group.sort_values('day')
        
        # Extract time-series data (180 days, 17 features)
        X_seq = group[ts_features].values
        
        # Extract static features (take first row) - FIX: ensure it's 1D array
        X_static = group[static_features].iloc[0].values
        
        predictions[patient_id] = (X_seq, X_static)
    
    return predictions

# =====================================================
# 4. Prediction Function (FIXED)
# =====================================================
def get_embeddings(model, X_seq):
    """Get embeddings from transformer"""
    model.eval()
    with torch.no_grad():
        X_seq_torch = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        x = model.input_proj(X_seq_torch)
        x = model.transformer(x)
        x = x.transpose(1, 2)
        x = model.pool(x).squeeze(-1)
        return x.numpy()

def predict_patient(xgb_model, transformer, static_mean, static_std, X_seq, X_static):
    """Make prediction for a single patient"""
    # Get embeddings from transformer
    emb = get_embeddings(transformer, X_seq)
    
    # Scale static features (using same scaler as training)
    X_static_scaled = (X_static - static_mean) / static_std
    
    # Combine embeddings and static features - FIX: ensure proper dimensions
    # emb shape: (1, d_model) - we need to squeeze to (d_model,)
    # X_static_scaled shape: (4,) - already 1D
    combined_features = np.hstack([emb.squeeze(0), X_static_scaled])
    
    # Make prediction
    probability = xgb_model.predict_proba(combined_features.reshape(1, -1))[0, 1]
    prediction = 1 if probability >= 0.5 else 0
    
    return probability, prediction

# =====================================================
# 5. Main Test Function
# =====================================================
def test_new_data(csv_file_path):
    """Test the model on new CSV data"""
    print("Loading models...")
    xgb_model, transformer, static_mean, static_std = load_models()
    
    print("Loading and preprocessing new data...")
    new_df = pd.read_csv(csv_file_path)
    patient_data = preprocess_new_data(new_df)
    
    print("\nMaking predictions...")
    results = []
    
    for patient_id, (X_seq, X_static) in patient_data.items():
        print(f"Processing patient {patient_id}...")
        print(f"X_seq shape: {X_seq.shape}")  # Should be (180, 17)
        print(f"X_static shape: {X_static.shape}")  # Should be (4,)
        
        proba, pred = predict_patient(xgb_model, transformer, static_mean, static_std, X_seq, X_static)
        results.append({
            'patient_id': patient_id,
            'prediction': pred,
            'probability': proba,
            'risk_level': 'High' if proba >= 0.5 else 'Low'
        })
        print(f"Patient {patient_id}: Prediction={pred}, Probability={proba:.4f}, Risk={results[-1]['risk_level']}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = 'predictions_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    return results_df

# =====================================================
# 6. Run the test
# =====================================================
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "test_zero.csv"
    
    try:
        results = test_new_data(csv_file)
        print("\n✅ Prediction completed successfully!")
        
        # Summary statistics
        high_risk_count = (results['probability'] >= 0.5).sum()
        total_patients = len(results)
        print(f"\nSummary: {high_risk_count}/{total_patients} patients at high risk")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure:")
        print("1. Your CSV has the same columns as training data")
        print("2. All model files are in the same directory")
        print("3. You have 180 days of data per patient")