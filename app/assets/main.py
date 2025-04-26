import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime

# --- Path Handling ---
def get_models_dir():
    """Return the correct models directory path"""
    try:
        base_path = Path(__file__).parent.parent
        models_dir = base_path / "models"
        if not models_dir.exists():
            models_dir = Path("models")  # Try local models directory
        os.makedirs(models_dir, exist_ok=True)
        return models_dir
    except Exception as e:
        st.error(f"Path resolution error: {str(e)}")
        st.stop()

# --- Model Loading ---
def safe_load_model(path):
    """Safely load a model with error handling"""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading {path.name}: {str(e)}")
        st.stop()

def find_latest_model(base_name):
    """Find the latest versioned model file"""
    models_dir = get_models_dir()
    versions = []
    for f in models_dir.glob(f"{base_name}_v*.pkl"):
        try:
            version_str = f.stem.split('_v')[-1]
            version = datetime.strptime(version_str, "%Y%m%d")
            versions.append((version, f))
        except:
            continue
    
    if versions:
        latest = sorted(versions, reverse=True)[0][1]
        return latest
    return None

# Initialize
models_dir = get_models_dir()

# --- Load Artifacts ---
try:
    # Try loading non-versioned files first
    models = {
        "Random Forest": safe_load_model(models_dir / "random_forest.pkl"),
        "SVM": safe_load_model(models_dir / "svm.pkl"),
        "Stacked Ensemble": safe_load_model(models_dir / "stacked_model.pkl")
    }
except:
    # Fall back to versioned files if non-versioned not found
    try:
        rf_path = find_latest_model("random_forest") or (models_dir / "random_forest_v20250426.pkl")
        svm_path = find_latest_model("svm") or (models_dir / "svm_v20250426.pkl")
        
        models = {
            "Random Forest": safe_load_model(rf_path),
            "SVM": safe_load_model(svm_path),
            "Stacked Ensemble": safe_load_model(models_dir / "stacked_model.pkl")
        }
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.write("Available model files:")
        for f in models_dir.glob("*.pkl"):
            st.write(f"- {f.name}")
        st.stop()

try:
    scaler = safe_load_model(models_dir / "scaler.pkl")
    le = safe_load_model(models_dir / "label_encoder.pkl")
except Exception as e:
    st.error(f"Artifact loading failed: {str(e)}")
    st.stop()

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("🌸 Iris Flower Classifier")

# Sidebar
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Main interface
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
        st.write("Confidence Scores:")
        st.dataframe(proba_df.style.format("{:.2%}"))
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Show EDA visualization
try:
    assets_dir = Path(__file__).parent / "assets"
    st.image(str(assets_dir / "pairplot.png"))
except Exception as e:
    st.warning(f"Couldn't load EDA visualization: {str(e)}")