import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.exceptions import NotFittedError

# --- Path Configuration ---
def get_base_path():
    """Get the correct base path for the project"""
    try:
        return Path(__file__).parent.parent
    except:
        return Path("F:/Projects/iris_flower_classification")

# --- Model Loading ---
def load_latest_model(model_type):
    """Load the most recent version of a model"""
    models_dir = get_base_path() / "models"
    try:
        model_files = list(models_dir.glob(f"{model_type}_v*.pkl"))
        if not model_files:
            # Try without version if no versioned files exist
            model_files = list(models_dir.glob(f"{model_type}.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No {model_type} models found")
        
        latest_model = max(model_files, key=os.path.getmtime)
        return joblib.load(latest_model)
    except Exception as e:
        st.error(f"Error loading {model_type} model: {str(e)}")
        st.stop()

def load_scaler_and_encoder():
    """Load scaler and label encoder"""
    base_path = get_base_path()
    try:
        scaler = joblib.load(base_path / "models" / "scaler.pkl")
        le = joblib.load(base_path / "models" / "label_encoder.pkl")
        return scaler, le
    except Exception as e:
        st.error(f"Error loading preprocessing artifacts: {str(e)}")
        st.stop()

# --- Initialize App ---
st.set_page_config(layout="wide")
st.title("🌸 Iris Flower Classifier")

# --- Load Artifacts ---
scaler, le = load_scaler_and_encoder()
models = {
    "Random Forest": load_latest_model("random_forest"),
    "SVM": load_latest_model("svm"),
    "Stacked Ensemble": load_latest_model("stacked_model")
}

# --- Sidebar Controls ---
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
show_probabilities = st.sidebar.checkbox("Show Probabilities", True)

# --- Main Interface ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# --- Prediction ---
if st.button("Predict"):
    try:
        # Create input DataFrame with feature names
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Get selected model
        model = models[model_name]
        
        # Make prediction
        pred = model.predict(input_scaled)
        species = le.inverse_transform(pred)[0]
        
        # Display results
        st.success(f"Predicted Species: **{species}**")
        
        if show_probabilities:
            try:
                proba = model.predict_proba(input_scaled)[0]
                proba_df = pd.DataFrame({
                    "Species": le.classes_,
                    "Probability": proba
                }).set_index("Species")
                
                st.bar_chart(proba_df)
                st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))
            except (AttributeError, NotFittedError):
                st.warning("Selected model doesn't support probability predictions")
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# --- EDA Visualization ---
try:
    assets_dir = get_base_path() / "app" / "assets"
    img_path = assets_dir / "pairplot.png"
    if img_path.exists():
        st.image(str(img_path), caption="Feature Relationships")
    else:
        st.warning("EDA visualization not found")
except Exception as e:
    st.warning(f"Couldn't load EDA visualization: {str(e)}")

# --- Debug Info (visible only in development) ---
if st.sidebar.checkbox("Show Debug Info", False):
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Base path: {get_base_path()}")
    st.sidebar.write("Available models:")
    for name, model in models.items():
        st.sidebar.write(f"- {name}: {type(model).__name__}")