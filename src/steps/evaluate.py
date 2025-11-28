"""
Model Evaluation Step
Evaluates the trained model on the test set and generates metrics and plots.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

def evaluate_model(model_path='models/random_forest_model.pkl', test_path='data/processed/test.csv'):
    """
    Evaluate the trained model on test data and generate visualizations.
    
    Args:
        model_path: Path to trained model
        test_path: Path to test data CSV
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    
    with mlflow.start_run(nested=True, run_name="evaluate_model"):
        # Load model and test data
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop('target', axis=1).values
        y_test = test_df['target'].values
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_samples", len(X_test))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create visualizations
        os.makedirs('outputs', exist_ok=True)
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = 'outputs/confusion_matrix.png'
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Feature Importance
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance (Random Forest)')
        plt.yticks(range(len(feature_importance)), [f'Feature {i}' for i in range(len(feature_importance))])
        fi_path = 'outputs/feature_importance.png'
        plt.savefig(fi_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Log artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        
        # Print detailed report
        print(f"\n✓ Model evaluation completed")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1-Score: {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist()
        }
        
        return metrics

if __name__ == "__main__":
    evaluate_model()
