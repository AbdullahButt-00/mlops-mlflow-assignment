"""
MLflow Pipeline Orchestration
Main pipeline definition that coordinates all steps:
extract_data -> preprocess_data -> train_model -> evaluate_model
"""
import mlflow
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.steps.extract_data import extract_data
from src.steps.preprocess import preprocess_data
from src.steps.train import train_model
from src.steps.evaluate import evaluate_model

def main():
    """
    Main pipeline function that orchestrates all steps.
    
    Pipeline Flow:
    1. Extract: Load Iris dataset and save as CSV
    2. Preprocess: Scale features and split into train/test
    3. Train: Train Random Forest Classifier
    4. Evaluate: Evaluate model and generate metrics/plots
    """
    
    # Set MLflow experiment
    mlflow.set_experiment("iris-classification-pipeline")
    
    print("=" * 60)
    print("Starting MLOps Pipeline: Iris Classification")
    print("=" * 60)
    
    with mlflow.start_run(run_name="iris_pipeline_full"):
        try:
            # Step 1: Extract Data
            print("\n[Step 1/4] Extracting data...")
            raw_data_path = extract_data()
            
            # Step 2: Preprocess Data
            print("\n[Step 2/4] Preprocessing data...")
            train_path, test_path, scaler_path = preprocess_data()
            
            # Step 3: Train Model
            print("\n[Step 3/4] Training model...")
            model_path = train_model()
            
            # Step 4: Evaluate Model
            print("\n[Step 4/4] Evaluating model...")
            metrics = evaluate_model(model_path, test_path)
            
            # Log final pipeline metrics
            mlflow.log_param("pipeline_status", "success")
            mlflow.log_param("dataset", "Iris")
            mlflow.log_param("model", "RandomForestClassifier")
            
            print("\n" + "=" * 60)
            print("✓ Pipeline completed successfully!")
            print("=" * 60)
            print(f"\nFinal Model Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print("\nCheck MLflow UI for detailed experiment tracking:")
            print("  Run: mlflow ui")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {str(e)}")
            mlflow.log_param("pipeline_status", "failed")
            raise

if __name__ == "__main__":
    main()
