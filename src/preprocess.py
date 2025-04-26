import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def load_data():
    """Load data with guaranteed path resolution"""
    try:
        # Method 1: Try relative path first
        df = pd.read_csv("data/IRIS.csv")
    except FileNotFoundError:
        try:
            # Method 2: Try alternative path structure
            current_dir = Path(__file__).parent
            data_path = current_dir.parent / "data" / "IRIS.csv"
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure:")
            print("1. IRIS.csv exists in the data/ folder")
            print("2. Your folder structure matches:")
            print("   iris-classification/")
            print("   ├── data/")
            print("   │   └── IRIS.csv")
            print("   └── src/")
            print("       └── preprocess.py")
            raise

    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    return df, le

def preprocess_data(df):
    """Standard preprocessing"""
    X = df.drop('species', axis=1)
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler