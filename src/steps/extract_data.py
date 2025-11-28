"""
Data Extraction Step
Fetches the Iris dataset and saves it as CSV.
"""
import os
import pandas as pd
from sklearn.datasets import load_iris
import mlflow

def extract_data():
    """Extract Iris dataset and save as CSV."""
    
    with mlflow.start_run(nested=True, run_name="extract_data"):
        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(
            data=iris.data,
            columns=iris.feature_names
        )
        df['target'] = iris.target
        
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Save the dataset
        output_path = 'data/raw/iris.csv'
        df.to_csv(output_path, index=False)
        
        # Log parameters and artifacts
        mlflow.log_param("dataset_name", "Iris")
        mlflow.log_param("num_rows", len(df))
        mlflow.log_param("num_features", len(iris.feature_names))
        mlflow.log_artifact(output_path)
        
        print(f"✓ Dataset extracted successfully to {output_path}")
        print(f"  Shape: {df.shape}")
        
        return output_path

if __name__ == "__main__":
    extract_data()
