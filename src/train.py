from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
from datetime import datetime
from preprocess import load_data, preprocess_data
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pandas as pd

def get_model_path(name, version=""):
    """Get reliable model save path with consistent naming"""
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Standardize naming: lowercase with underscores and proper version formatting
    model_name = name.lower().replace(' ', '_')
    if version:
        return model_dir / f"{model_name}_v{version}.pkl"  # Ensure underscore before version
    else:
        return model_dir / f"{model_name}.pkl"

# Load and preprocess data
df, le = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Define models
models = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "svm": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# Train models
best_models = {}
for name, config in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring="accuracy"
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    
    # Save with versioning
    version = datetime.now().strftime("%Y%m%d")
    model_path = get_model_path(name, version)
    joblib.dump(grid.best_estimator_, model_path)
    
    # Save without version for easier reference
    joblib.dump(grid.best_estimator_, get_model_path(name))
    print(f"Saved {name} to {model_path}")

# Save artifacts with consistent naming
joblib.dump(scaler, get_model_path("scaler"))
joblib.dump(le, get_model_path("label_encoder"))

# Save test data
joblib.dump((X_test, y_test), get_model_path("test_data"))
print("Test data saved to test_data.pkl")
print("Training completed successfully!")