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

def get_model_path(name, version):
    """Get reliable model save path"""
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir / f"{name.lower().replace(' ', '_')}_v{version}.pkl"

# Load and preprocess data
df, le = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# After preprocessing, before saving:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # X_train should be a DataFrame

# Define models
models = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20]
        }
    },
    "SVM": {
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
    print(f"Saved {name} to {model_path}")


# Save artifacts
joblib.dump(scaler, get_model_path("scaler", ""))
joblib.dump(le, get_model_path("label_encoder", ""))
# Save test data separately
joblib.dump((X_test, y_test), "F:/Projects/iris_flower_classification/models/test_data.pkl")
print("Test data saved to test_data.pkl")
print("Training completed successfully!")