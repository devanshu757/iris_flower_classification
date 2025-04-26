from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
from datetime import datetime
from preprocess import load_data, preprocess_data
import os
from pathlib import Path

def get_model_path(name, version=None):
    """Get reliable model save path with guaranteed naming"""
    current_dir = Path(__file__).parent
    model_dir = current_dir.parent / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Standardize naming
    clean_name = name.lower().replace(' ', '_')
    if version:
        return model_dir / f"{clean_name}_v{version}.pkl"
    return model_dir / f"{clean_name}.pkl"

# Load and preprocess data
df, le = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Define and train models
models = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}
    },
    "svm": {
        "model": SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    }
}

for name, config in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(config["model"], config["params"], cv=5)
    grid.fit(X_train, y_train)
    
    # Save with and without version
    version = datetime.now().strftime("%Y%m%d")
    joblib.dump(grid.best_estimator_, get_model_path(name, version))
    joblib.dump(grid.best_estimator_, get_model_path(name))

# Save all artifacts
joblib.dump(scaler, get_model_path("scaler"))
joblib.dump(le, get_model_path("label_encoder"))
joblib.dump((X_test, y_test), get_model_path("test_data"))

print("All models and artifacts saved successfully!")