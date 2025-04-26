from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from pathlib import Path
from train import best_models
from preprocess import load_data, preprocess_data

# 1. Load data
df, le = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# 2. Ensure models directory exists
models_dir = Path(__file__).parent.parent / "models"
os.makedirs(models_dir, exist_ok=True)  # Creates directory if doesn't exist

# 3. Create stacked model - use lowercase keys to match train.py
stacked_model = StackingClassifier(
    estimators=[
        ("rf", best_models["random_forest"]),  # Changed from "Random Forest" to "random_forest"
        ("svm", best_models["svm"]),          # Changed from "SVM" to "svm"
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# 4. Train and save
stacked_model.fit(X_train, y_train)
model_path = models_dir / "stacked_model.pkl"
joblib.dump(stacked_model, model_path)

print(f"Stacked model successfully saved to: {model_path}")