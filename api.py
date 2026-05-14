"""
FastAPI application for M1 Purchase Prediction and Customer Segmentation
Endpoint: GET /predict?trust_apple=...&user_pcmac=...&familiarity_m1=...&...
"""

import pickle
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI(
    title="M1 Purchase Prediction API",
    description="Real-time M1 purchase prediction and customer segmentation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models for M1 prediction
kmeans_model = None
m1_classifier = None
scaler = None
encoders = None

try:
    with open('./models/kmeans_m1_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    print("✓ K-Means M1 model loaded (for customer segmentation)")
except Exception as e:
    print(f"✗ Error loading K-Means M1 model: {e}")

try:
    with open('./models/m1_classifier.pkl', 'rb') as f:
        m1_classifier = pickle.load(f)
    print("✓ M1 Purchase Classifier loaded")
except Exception as e:
    print(f"✗ Error loading M1 Classifier: {e}")

try:
    with open('./models/scaler_m1.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded for M1 preprocessing")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")

try:
    with open('./models/encoders_m1.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("✓ Label encoders loaded for M1 preprocessing")
except Exception as e:
    print(f"✗ Error loading encoders: {e}")


def _ensure_label_encoders():
    """Fit gender/status/domain encoders from training raw CSV if pickle omitted them."""
    global encoders
    if encoders is None:
        return
    if encoders.get('label_encoders'):
        return
    raw_path = Path(__file__).resolve().parent / 'data' / 'raw' / 'M1_data.csv'
    if not raw_path.exists():
        print('✗ Cannot hydrate label encoders: data/raw/M1_data.csv missing')
        return
    raw = pd.read_csv(raw_path)
    les = {}
    for col in ['gender', 'status', 'domain']:
        le = LabelEncoder()
        le.fit(raw[col].astype(str))
        les[col] = le
    encoders['label_encoders'] = les
    print('✓ Hydrated label encoders for gender/status/domain from M1_data.csv')


_ensure_label_encoders()


def _scaler_feature_order():
    """Column order must match StandardScaler.fit in preprocessing."""
    if scaler is not None and getattr(scaler, 'feature_names_in_', None) is not None:
        return list(scaler.feature_names_in_)
    return [
        'trust_apple', 'interest_computers', 'age_computer', 'user_pcmac',
        'appleproducts_count', 'familiarity_m1', 'f_batterylife', 'f_price', 'f_size',
        'f_multitasking', 'f_noise', 'f_performance', 'f_neural', 'f_synergy',
        'f_performanceloss', 'm1_consideration', 'gender', 'age_group', 'income_group',
        'status', 'domain',
    ]


def _build_raw_row_from_query(trust_apple, user_pcmac, familiarity_m1, interest_computers,
                               age_computer, appleproducts_count, f_batterylife, f_price,
                               f_size, f_multitasking, f_noise, f_performance, f_neural,
                               f_synergy, f_performanceloss, m1_consideration, age_group,
                               income_group, gender, status, domain):
    return {
        'trust_apple': trust_apple,
        'user_pcmac': user_pcmac,
        'familiarity_m1': familiarity_m1,
        'interest_computers': interest_computers,
        'age_computer': age_computer,
        'appleproducts_count': appleproducts_count,
        'f_batterylife': f_batterylife,
        'f_price': f_price,
        'f_size': f_size,
        'f_multitasking': f_multitasking,
        'f_noise': f_noise,
        'f_performance': f_performance,
        'f_neural': f_neural,
        'f_synergy': f_synergy,
        'f_performanceloss': f_performanceloss,
        'm1_consideration': m1_consideration,
        'age_group': age_group,
        'income_group': income_group,
        'gender': gender,
        'status': status,
        'domain': domain,
    }


def preprocess_exact(raw_row):
    """Use saved scaler and encoders to transform a single-row raw_row dict into model input for M1."""
    if scaler is None or encoders is None:
        raise RuntimeError('Scaler and encoders not available')

    feature_cols = _scaler_feature_order()

    row = {}
    for col in feature_cols:
        val = raw_row.get(col)
        if val is None:
            raise ValueError(f'Missing value for {col}')
        row[col] = val

    raw_df = pd.DataFrame([row])

    label_encoders = encoders.get('label_encoders', {})

    # 1a. Yes/No fields — strip + capitalize (Yes/No)
    for col in ['trust_apple', 'familiarity_m1']:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.strip().str.capitalize()

    # 1b. user_pcmac — strip + capitalize (Apple/Pc/Yes/No)
    if 'user_pcmac' in raw_df.columns:
        raw_df['user_pcmac'] = raw_df['user_pcmac'].astype(str).str.strip().str.capitalize()

    # 1c. gender, status, domain — strip only (must match training strings, e.g. IT & Technology)
    for col in ['gender', 'status', 'domain']:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.strip()

    # 2. Binary Yes/No for trust_apple and familiarity_m1
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['trust_apple', 'familiarity_m1']:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].map(binary_map)
            if raw_df[col].isnull().any():
                raise ValueError(f"Unknown {col} value. Must be Yes or No")

    # user_pcmac: device type (Apple/PC/...) or form values Yes/No (Mac/PC)
    pcmac_map = (encoders or {}).get(
        'user_pcmac_map',
        {'Apple': 1, 'Pc': 0, 'Hp': 0, 'Other': 0, 'Yes': 1, 'No': 0},
    )
    if 'user_pcmac' in raw_df.columns:
        raw_df['user_pcmac'] = raw_df['user_pcmac'].map(pcmac_map)
        if raw_df['user_pcmac'].isnull().any():
            raise ValueError("Unknown user_pcmac value. Use Apple/PC (or Yes/No for Mac/PC).")
    
    # 3. Encode categorical columns using saved label encoders
    for col in ['gender', 'status', 'domain']:
        if col in raw_df.columns and col in label_encoders:
            encoder = label_encoders[col]
            raw_df[col] = encoder.transform(raw_df[col].astype(str))
        elif col in raw_df.columns:
            raise ValueError(f"No encoder found for {col}")
    
    # 4. Convert all to float and apply scaler
    X = raw_df[feature_cols].astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled

@app.get("/")
def home():
    """Serve the index.html file"""
    html_file = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    return {
        "status": "API Running",
        "message": "index.html not found. Please ensure index.html is in the root directory"
    }

@app.get("/predict")
def predict(
    trust_apple: str = Query(..., description="Trust Apple (Yes/No)"),
    user_pcmac: str = Query(..., description="Mac/PC: Yes=Mac, No=PC (or Apple/Pc)"),
    familiarity_m1: str = Query(..., description="Familiarity M1 (Yes/No)"),
    interest_computers: str = Query(..., description="Interest in Computers (1-10)"),
    age_computer: str = Query(..., description="Years Using Computers (0-60)"),
    appleproducts_count: str = Query(..., description="Apple Products Count (0-10)"),
    f_batterylife: str = Query("3", description="Battery life importance (1-5); default 3 if omitted"),
    f_price: str = Query(..., description="Features: Price importance (1-5)"),
    f_size: str = Query("3", description="Size importance (1-5); default 3"),
    f_multitasking: str = Query("3", description="Multitasking importance (1-5); default 3"),
    f_noise: str = Query("3", description="Noise importance (1-5); default 3"),
    f_performance: str = Query(..., description="Features: Performance importance (1-5)"),
    f_neural: str = Query("3", description="Neural Engine importance (1-5); default 3"),
    f_synergy: str = Query("3", description="Ecosystem synergy importance (1-5); default 3"),
    f_performanceloss: str = Query("3", description="Performance vs Mac importance (1-5); default 3"),
    m1_consideration: str = Query(..., description="M1 Consideration level (1-10)"),
    age_group: str = Query("2", description="Age group code as in survey (default 2)"),
    income_group: str = Query("2", description="Income group code (default 2)"),
    gender: str = Query(..., description="Gender (Male/Female)"),
    status: str = Query("Student", description="Employment status (default Student)"),
    domain: str = Query("IT & Technology", description="Field/domain (default IT & Technology)"),
):
    """
    Predict M1 customer cluster AND M1 purchase likelihood.
    
    Returns:
    - predicted_cluster: Customer segment (from K-Means)
    - predicted_m1_purchase: Whether customer will purchase M1 (from M1 Classifier)
    - m1_purchase_probability: Confidence score for purchase prediction
    - input_features: Echo of input parameters
    """
    
    if kmeans_model is None or m1_classifier is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Models not loaded. Please check model files are present."}
        )

    try:
        # Build raw row from query parameters
        raw_row = _build_raw_row_from_query(
            trust_apple, user_pcmac, familiarity_m1, interest_computers,
            age_computer, appleproducts_count, f_batterylife, f_price,
            f_size, f_multitasking, f_noise, f_performance, f_neural,
            f_synergy, f_performanceloss, m1_consideration, age_group,
            income_group, gender, status, domain,
        )
        
        # Preprocess using saved scaler and encoders
        input_data = preprocess_exact(raw_row)

        # Predict cluster (K-Means) — unsupervised segmentation
        cluster_prediction = kmeans_model.predict(input_data)[0]
        
        # Predict M1 purchase (Binary Classification: 0=No, 1=Yes)
        m1_pred_numeric = m1_classifier.predict(input_data)[0]
        m1_probabilities = m1_classifier.predict_proba(input_data)[0]
        
        # Map 0/1 prediction to "No"/"Yes"
        m1_pred_str = "Yes" if m1_pred_numeric == 1 else "No"
        
        # Get probability for "Yes" (class 1)
        # Classes are [0, 1], so index 1 gives probability of "Yes"
        yes_prob = float(m1_probabilities[1]) if 1 in m1_classifier.classes_ else 0.0
        
        # Create response
        return {
            "status": "success",
            "predicted_cluster": int(cluster_prediction),
            "predicted_m1_purchase": m1_pred_str,
            "m1_purchase_probability": yes_prob,
            "input_features": {
                "trust_apple": trust_apple,
                "user_pcmac": user_pcmac,
                "familiarity_m1": familiarity_m1,
                "interest_computers": interest_computers,
                "age_computer": age_computer,
                "appleproducts_count": appleproducts_count,
                "f_batterylife": f_batterylife,
                "f_price": f_price,
                "f_size": f_size,
                "f_multitasking": f_multitasking,
                "f_noise": f_noise,
                "f_performance": f_performance,
                "f_neural": f_neural,
                "f_synergy": f_synergy,
                "f_performanceloss": f_performanceloss,
                "m1_consideration": m1_consideration,
                "age_group": age_group,
                "income_group": income_group,
                "gender": gender,
                "status": status,
                "domain": domain
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Prediction error: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
