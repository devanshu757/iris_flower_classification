import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.explain import explain_prediction

# Load artifacts
models = {
    "Random Forest": joblib.load("../models/random_forest_v20231025.pkl"),
    "SVM": joblib.load("../models/svm_v20231025.pkl"),
    "Stacked Ensemble": joblib.load("../models/stacked_model.pkl")
}
scaler = joblib.load("../models/scaler.pkl")
le = joblib.load("../models/label_encoder.pkl")

# App layout
st.set_page_config(layout="wide")
st.title("🌸 Advanced Iris Classifier")

# Sidebar controls
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Main interface
col1, col2 = st.columns(2)

with col1:
    # Input sliders
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
    
with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
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
    })
    st.bar_chart(proba_df.set_index("Species"))
    
    
# Show EDA visualizations
st.image("../app/assets/pairplot.png")