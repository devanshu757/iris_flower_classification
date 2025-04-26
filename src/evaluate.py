from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import joblib
from pathlib import Path
import numpy as np

def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro"),
        "macro_recall": recall_score(y_test, y_pred, average="macro"),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred, target_names=class_names)
    }

if __name__ == "__main__":
    try:
        models_dir = Path("F:/Projects/iris_flower_classification/models")
        
        # Load artifacts
        scaler = joblib.load(models_dir / "scaler.pkl")
        le = joblib.load(models_dir / "label_encoder.pkl")
        stacked_model = joblib.load(models_dir / "stacked_model.pkl")

        # Load test data (you'll need to save this separately during training)
        # Add this to your train.py after preprocessing:
        # joblib.dump((X_test, y_test), models_dir / "test_data.pkl")
        X_test, y_test = joblib.load(models_dir / "test_data.pkl")
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)

        metrics = evaluate_model(stacked_model, X_test_scaled, y_test, le.classes_)
        
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['macro_precision']:.4f}")
        print(f"Recall: {metrics['macro_recall']:.4f}")
        print(f"F1 Score: {metrics['macro_f1']:.4f}")
        print("\nClassification Report:")
        print(metrics['report'])

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Current models directory contents:")
        for f in models_dir.glob("*"):
            print(f" - {f.name}")
        print("\nSolution: Please run train.py first to generate test_data.pkl")