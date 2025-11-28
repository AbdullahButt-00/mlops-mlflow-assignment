"""
Data Preprocessing Step
Cleans, scales, and splits the dataset into train/test sets.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import mlflow
import joblib

def preprocess_data(input_path='data/raw/iris.csv', test_size=0.2, random_state=42):
    """
    Preprocess the dataset: scale features and split into train/test.
    
    Args:
        input_path: Path to raw CSV file
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: Paths to saved train/test data and scaler
    """
    
    with mlflow.start_run(nested=True, run_name="preprocess_data"):
        # Load raw data
        df = pd.read_csv(input_path)
        
        # Separate features and target
        X = df.drop('target', axis=1).values
        y = df['target'].values
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Ensure balanced splits
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        train_data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        train_data['target'] = y_train
        train_path = 'data/processed/train.csv'
        train_data.to_csv(train_path, index=False)
        
        test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        test_data['target'] = y_test
        test_path = 'data/processed/test.csv'
        test_data.to_csv(test_path, index=False)
        
        # Save scaler for later use
        scaler_path = 'data/processed/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        # Log parameters and metrics
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])
        
        # Log artifacts
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
        mlflow.log_artifact(scaler_path)
        
        print(f"✓ Preprocessing completed successfully")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Scaler saved to {scaler_path}")
        
        return train_path, test_path, scaler_path

if __name__ == "__main__":
    preprocess_data()
