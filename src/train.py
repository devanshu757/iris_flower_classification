import os
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from preprocess import load_data, preprocess_data

# Global dictionary to store trained models
best_models = {}

# Set absolute path to models directory
MODEL_DIR = Path(r"F:\Projects\iris_flower_classification\models")
MODEL_DIR.mkdir(exist_ok=True)

def train_and_save_models():
    global best_models
    
    # Load and preprocess data
    print("Loading data...")
    df, le = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train models
    print("\nTraining models...")
    models_to_train = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(probability=True, random_state=42)
    }

    for name, model in models_to_train.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        best_models[name] = model
        
        # Save each model
        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved {name} to: {model_path}")

    # Save artifacts
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    joblib.dump((X_test, y_test), MODEL_DIR / "test_data.pkl")

    print("\nAll files successfully saved to:", MODEL_DIR)
    print("Created files:")
    for f in MODEL_DIR.glob("*"):
        print(f"- {f.name}")

if __name__ == "__main__":
    train_and_save_models()