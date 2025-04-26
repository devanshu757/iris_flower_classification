import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# --- Path Handling ---
def get_models_dir():
    """Return the correct models directory path"""
    try:
        # Try relative path first
        base_path = Path(__file__).parent.parent
        models_dir = base_path / "models"
        
        # Verify directory exists
        if not models_dir.exists():
            # Fallback to absolute path
            models_dir = Path("F:/Projects/iris_flower_classification/models")
            if not models_dir.exists():
                raise FileNotFoundError("Models directory not found")
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

# Initialize
models_dir = get_models_dir()

# --- Load Artifacts with Version Tolerance ---
def load_latest_version(base_name):
    """Find and load the latest version of a model"""
    model_files = list(models_dir.glob(f"{base_name}_v*.pkl"))
    if not model_files:
        model_files = list(models_dir.glob(f"{base_name}.pkl"))
        if not model_files:
            st.error(f"No {base_name} model found in {models_dir}")
            st.stop()
    latest_model = max(model_files, key=os.path.getmtime)
    return safe_load_model(latest_model)

# Load all required artifacts
try:
    models = {
        "Random Forest": load_latest_version("random_forest"),
        "SVM": load_latest_version("svm"),
        "Stacked Ensemble": safe_load_model(models_dir / "stacked_model.pkl")
    }
    scaler = safe_load_model(models_dir / "scaler.pkl")
    le = safe_load_model(models_dir / "label_encoder.pkl")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.write("Available model files:")
    for f in models_dir.glob("*.pkl"):
        st.write(f"- {f.name}")
    st.stop()

# --- App Layout ---
st.set_page_config(layout="wide")
st.title("🌸 Advanced Iris Classifier")

# Sidebar controls
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
show_shap = st.sidebar.checkbox("Show SHAP Explanation")

# Main interface
col1, col2 = st.columns(2)

with col1:
    # Input sliders
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
        
        # Display results
        st.success(f"Prediction: {le.inverse_transform(pred)[0]}")
        
        # Probability plot
        proba_df = pd.DataFrame({
            "Species": le.classes_,
            "Probability": proba
        }).set_index("Species")
        
        st.bar_chart(proba_df)
        st.write("Confidence Scores:")
        st.dataframe(proba_df.style.format("{:.2%}").background_gradient(cmap='Blues'))
        
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Show EDA visualization
try:
    assets_dir = Path(__file__).parent / "assets"
    st.image(str(assets_dir / "pairplot.png"))
except Exception as e:
    st.warning(f"Couldn't load EDA visualization: {str(e)}")

# Debug section (visible only in development)
if "DEBUG" in os.environ:
    st.sidebar.markdown("### Debug Info")
    st.sidebar.write(f"Models dir: {models_dir}")
    st.sidebar.write("Loaded models:", list(models.keys()))