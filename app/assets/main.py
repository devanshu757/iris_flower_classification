import joblib
from pathlib import Path
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from train import best_models

MODEL_DIR = Path(r"F:\Projects\iris_flower_classification\models")

def create_stacked_model():
    # Create stacked model
    stacked_model = StackingClassifier(
        estimators=[
            ("rf", best_models["random_forest"]),
            ("svm", best_models["svm"]),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Save stacked model
    stacked_path = MODEL_DIR / "stacked_model.pkl"
    joblib.dump(stacked_model, stacked_path)
    print(f"Saved stacked model to: {stacked_path}")

if __name__ == "__main__":
    create_stacked_model()