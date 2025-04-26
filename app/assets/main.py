import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- Path Handling ---
def get_models_dir():
    """Return the correct models directory path"""
    try:
        base_path = Path(__file__).parent.parent
        models_dir = base_path / "models"
        if not models_dir.exists():
            models_dir = Path("F:/Projects/iris_flower_classification/models")
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

# --- Load Artifacts ---
try:
    models = {
        "Random Forest": safe_load_model(models_dir / "random_forest_v20250426.pkl"),
        "SVM": safe_load_model(models_dir / "svm_v20250426.pkl"),
        "Stacked Ensemble": safe_load_model(models_dir / "stacked_model.pkl")
    }
    scaler = safe_load_model(models_dir / "scaler_v.pkl")
    le = safe_load_model(models_dir / "label_encoder_v.pkl")
except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.write("Available model files:")
    for f in models_dir.glob("*.pkl"):
        st.write(f"- {f.name}")
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

if st.button("Predict"):
    try:
        # Get feature names (either from scaler or use defaults)
        feature_names = getattr(scaler, 'feature_names_in_', 
                              ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        # Create properly labeled DataFrame
        input_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=feature_names
        )
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        pred = models[model_name].predict(input_scaled)
        proba = models[model_name].predict_proba(input_scaled)[0]
        
        # Display results
        st.success(f"Prediction: {le.inverse_transform(pred)[0]}")
        
        # Show probabilities
        proba_df = pd.DataFrame({
            'Species': le.classes_,
            'Probability': proba
        }).set_index('Species')
        
        st.bar_chart(proba_df)
        st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))
        
    except ValueError as e:
        st.error(f"Input error: Please check your values. {str(e)}")
    except AttributeError as e:
        st.error(f"Model error: {str(e)}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Technical details:")
        st.code(str(e), language='python')

        
# Show EDA visualization
try:
    # Get the correct assets directory path
    assets_dir = Path(__file__).parent / "assets"
    
    # Verify the image exists
    image_path = assets_dir / "pairplot.png"
    if not image_path.exists():
        # Alternative path if using notebook structure
        image_path = Path("F:/Projects/iris_flower_classification/app/assets/pairplot.png")
    
    if image_path.exists():
        st.image(str(image_path), caption="Feature Relationships")
    else:
        st.warning("EDA visualization not found at:")
        st.write(f"- {assets_dir / 'pairplot.png'}")
        st.write(f"- {image_path}")
except Exception as e:
    st.warning(f"Couldn't load EDA visualization: {str(e)}")