import joblib
from pathlib import Path
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_data, preprocess_data  # Added import

# Use absolute path
MODEL_DIR = Path(r"F:\Projects\iris_flower_classification\models")

def load_individual_models():
    """Load the pre-trained models"""
    models = {}
    for name in ["random_forest", "svm"]:
        model_path = MODEL_DIR / f"{name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[name] = joblib.load(model_path)
    return models

def create_and_save_stacked_model():
    # Load individual models
    individual_models = load_individual_models()
    
    # Create stacked model
    stacked_model = StackingClassifier(
        estimators=[
            ("rf", individual_models["random_forest"]),
            ("svm", individual_models["svm"]),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Load fresh training data (don't use test data for training!)
    df, le = load_data()
    X_train, _, y_train, _, _ = preprocess_data(df)  # Only need training data
    
    # Train stacked model
    stacked_model.fit(X_train, y_train)
    
    # Save stacked model
    stacked_path = MODEL_DIR / "stacked_model.pkl"
    joblib.dump(stacked_model, stacked_path)
    print(f"Successfully saved stacked model to: {stacked_path}")

if __name__ == "__main__":
    create_and_save_stacked_model()