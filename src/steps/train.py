"""
Model Training Step
Trains a Random Forest Classifier on the preprocessed data.
"""
import os
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(train_path='data/processed/train.csv', n_estimators=100, random_state=42, max_depth=10):
    """
    Train a Random Forest Classifier.
    
    Args:
        train_path: Path to training data CSV
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        max_depth: Maximum depth of trees
        
    Returns:
        str: Path to saved model
    """
    
    with mlflow.start_run(nested=True, run_name="train_model"):
        # Load training data
        train_df = pd.read_csv(train_path)
        X_train = train_df.drop('target', axis=1).values
        y_train = train_df['target'].values
        
        # Log hyperparameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_depth", max_depth)
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        # Log training metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("training_samples", len(X_train))
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_model.pkl'
        joblib.dump(model, model_path)
        
        # Log artifact
        mlflow.log_artifact(model_path)
        
        print(f"✓ Model training completed")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Model saved to {model_path}")
        
        return model_path

if __name__ == "__main__":
    train_model()
