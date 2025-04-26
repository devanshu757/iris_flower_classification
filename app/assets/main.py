import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Use absolute path
MODEL_DIR = Path(r"F:\Projects\iris_flower_classification\models")

def load_model(name):
    """Load model with thorough verification"""
    path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.pkl"
    if not path.exists():
        st.error(f"Model file not found: {path.name}")
        st.write("Available model files:")
        for f in sorted(MODEL_DIR.glob("*.pkl")):
            st.write(f"- {f.name}")
        st.stop()
    return joblib.load(path)

# Load all required models and artifacts
try:
    models = {
        "Random Forest": load_model("random_forest"),
        "SVM": load_model("svm")
    }
    scaler = load_model("scaler")
    le = load_model("label_encoder")
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