import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import os

# Set up paths
MODEL_DIR = Path("models")
os.makedirs(MODEL_DIR, exist_ok=True)

def find_model_file(base_name):
    """Find model file with flexible naming"""
    possible_names = [
        f"{base_name}.pkl",
        f"{base_name.lower()}.pkl",
        f"{base_name.replace(' ', '_')}.pkl",
        f"{base_name.lower().replace(' ', '_')}.pkl"
    ]
    
    for name in possible_names:
        path = MODEL_DIR / name
        if path.exists():
            return path
    
    # If no exact match, try versioned files
    versioned_files = list(MODEL_DIR.glob(f"{base_name.lower().replace(' ', '_')}*.pkl"))
    if versioned_files:
        return max(versioned_files, key=lambda f: f.stat().st_mtime)
    
    return None

def load_model_safely(name):
    """Load model with multiple fallback options"""
    path = find_model_file(name)
    if path is None:
        st.error(f"Model file not found for: {name}")
        st.write("Available model files:")
        files = list(MODEL_DIR.glob("*.pkl"))
        if files:
            for f in files:
                st.write(f"- {f.name}")
        else:
            st.write("No model files found in models directory")
        st.stop()
    
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model {path.name}: {str(e)}")
        st.stop()

# Load models with flexible naming
try:
    models = {
        "Random Forest": load_model_safely("random_forest"),
        "SVM": load_model_safely("svm")
    }
    scaler = load_model_safely("scaler")
    le = load_model_safely("label_encoder")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Input sliders
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    try:
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        input_scaled = scaler.transform(input_data)
        model = models[model_name]
        pred = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)[0]
        
        st.success(f"Prediction: {le.inverse_transform(pred)[0]}")
        proba_df = pd.DataFrame({
            "Species": le.classes_,
            "Probability": proba
        }).set_index("Species")
        st.bar_chart(proba_df)
        st.dataframe(proba_df.style.format("{:.2%}"))
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")