# MLOps Pipeline with MLflow and DVC - Iris Classification

A complete Machine Learning Operations (MLOps) pipeline for the Iris classification problem using industry-standard tools: MLflow, DVC, GitHub Actions, and Kubernetes (optional).

##  Project Overview

This project demonstrates a production-ready MLOps pipeline that includes:

- **Data Versioning**: Using DVC to track and version the Iris dataset
- **Experiment Tracking**: MLflow for tracking parameters, metrics, and artifacts
- **Pipeline Orchestration**: Automated multi-step ML pipeline (extract → preprocess → train → evaluate)
- **Continuous Integration**: GitHub Actions for automated testing and pipeline execution
- **Containerization**: Docker support for reproducible environments
- **Kubernetes Ready**: Optional deployment using Minikube

### ML Problem
Binary classification on the Iris dataset using a Random Forest Classifier to predict iris species.

**Dataset**: Iris (150 samples, 4 features, 3 classes)
**Model**: Random Forest Classifier
**Metrics**: Accuracy, F1-Score, Confusion Matrix, Feature Importance

---

##  Quick Start

### Prerequisites
- Python 3.10+
- Git
- Docker (optional)
- Minikube (optional)
- Linux/Ubuntu environment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mlops-mlflow-assignment.git
   cd mlops-mlflow-assignment
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Ubuntu/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC**
   ```bash
   dvc init --no-scm
   dvc remote add -d local dvc_remote
   mkdir -p dvc_remote
   ```

---

##  Pipeline Architecture

```
Extract Data (Iris Dataset)
    ↓
Preprocess Data (Scale & Split)
    ↓
Train Model (Random Forest)
    ↓
Evaluate Model (Metrics & Plots)
```

### Pipeline Steps

#### Step 1: Extract Data (`src/steps/extract_data.py`)
- Loads the Iris dataset from scikit-learn
- Saves as CSV to `data/raw/iris.csv`
- **Inputs**: None (hardcoded dataset)
- **Outputs**: Raw data CSV file
- **Logged Metrics**: Dataset name, number of rows, number of features

#### Step 2: Preprocess Data (`src/steps/preprocess.py`)
- Splits data into 80% train, 20% test
- Standardizes features using StandardScaler
- **Inputs**: Raw data CSV
- **Outputs**: `data/processed/train.csv`, `data/processed/test.csv`, `data/processed/scaler.pkl`
- **Logged Parameters**: Test size, random state, train/test sample counts

#### Step 3: Train Model (`src/steps/train.py`)
- Trains Random Forest Classifier with 100 estimators
- **Inputs**: Training data CSV
- **Outputs**: Trained model (`models/random_forest_model.pkl`)
- **Logged Metrics**: Training accuracy, training samples
- **Hyperparameters**: n_estimators=100, max_depth=10, random_state=42

#### Step 4: Evaluate Model (`src/steps/evaluate.py`)
- Evaluates on test set
- Generates confusion matrix and feature importance plots
- **Inputs**: Trained model, test data
- **Outputs**: Evaluation plots in `outputs/`
- **Logged Metrics**: Test accuracy, F1-score, confusion matrix visualization

---

##  Running the Pipeline

### Local Execution

#### Run entire pipeline:
```bash
python pipeline.py
```

Output:
```
============================================================
Starting MLOps Pipeline: Iris Classification
============================================================

[Step 1/4] Extracting data...
✓ Dataset extracted successfully to data/raw/iris.csv
  Shape: (150, 5)

[Step 2/4] Preprocessing data...
✓ Preprocessing completed successfully
  Training set: (120, 4)
  Test set: (30, 4)
  Scaler saved to data/processed/scaler.pkl

[Step 3/4] Training model...
✓ Model training completed
  Training Accuracy: 0.9917
  Model saved to models/random_forest_model.pkl

[Step 4/4] Evaluating model...
✓ Model evaluation completed
  Test Accuracy: 1.0000
  Test F1-Score: 1.0000

============================================================
✓ Pipeline completed successfully!
============================================================
```

#### Run individual steps (for testing):
```bash
# Extract
python -c "from src.steps.extract_data import extract_data; extract_data()"

# Preprocess
python -c "from src.steps.preprocess import preprocess_data; preprocess_data()"

# Train
python -c "from src.steps.train import train_model; train_model()"

# Evaluate
python -c "from src.steps.evaluate import evaluate_model; evaluate_model()"
```

### Using DVC

Track and reproduce the pipeline with DVC:

```bash
# Run the full DVC pipeline
dvc repro

# Check pipeline status
dvc status

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

---

## 📈 MLflow Tracking

### Start MLflow UI

```bash
mlflow ui
```

Visit `http://localhost:5000` to view:
- All experiment runs
- Parameters (hyperparameters, dataset info)
- Metrics (accuracy, F1-score)
- Artifacts (trained model, plots, data files)
- Run comparison across experiments

### MLflow Structure
```
Experiment: iris-classification-pipeline
├── Run: iris_pipeline_full
│   ├── Run: extract_data (nested)
│   ├── Run: preprocess_data (nested)
│   ├── Run: train_model (nested)
│   └── Run: evaluate_model (nested)
```

Each nested run logs:
- Parameters specific to that step
- Metrics (accuracy, F1-score, sample counts)
- Artifacts (data files, plots, models)

---

## 🔄 CI/CD with GitHub Actions

### Workflow File: `.github/workflows/mlops-pipeline.yml`

The GitHub Actions workflow runs on:
- **Push** to `main` or `develop` branches
- **Pull Requests** to `main` or `develop`
- **Manual trigger** (`workflow_dispatch`)

### Pipeline Stages

1. **Environment Setup**
   - Checkout code (`actions/checkout@v4`)
   - Install Python 3.10 (`actions/setup-python@v5`)
   - Install dependencies from `requirements.txt`
   - Cache pip packages for faster builds

2. **Code Quality (Linting)**
   - Run `pylint` on `src/` directory
   - Non-blocking (continues even if lint fails)

3. **Testing**
   - Run `pytest` on `tests/` directory
   - Non-blocking (continues even if tests are missing or fail)

4. **DVC Setup**
   - Create DVC directories (`data/raw`, `data/processed`, `models`, `outputs`, `dvc_remote`)
   - Initialize DVC with local remote (`dvc init --no-scm`)
   - Skip if already initialized

5. **Pipeline Execution**
   - Run the full MLOps pipeline: `python pipeline.py`
   - Executes all 4 steps:
     1. **Extract Data** → CSV saved to `data/raw/`
     2. **Preprocess** → train/test splits & scaler saved to `data/processed/` and `preprocessor/`
     3. **Train Model** → RandomForest saved to `models/`
     4. **Evaluate** → Metrics computed and plots saved in `outputs/`
   - Logs parameters, metrics, and artifacts to MLflow

6. **Artifact Upload**
   - Upload pipeline outputs (`models/`, `outputs/`, `data/processed/`) and MLflow tracking directory (`mlruns/`) as GitHub Actions artifacts
   - Download via Actions tab → Artifacts

### Trigger a Workflow Run

#### Automatic (push to main)
```bash
git add .
git commit -m "Update pipeline"
git push origin main
```

#### Manual (via GitHub UI)
1. Go to repository → Actions tab
2. Select "MLOps Pipeline CI/CD"
3. Click "Run workflow" → "Run workflow"

#### Manual (via GitHub CLI)
```bash
gh workflow run mlops-pipeline.yml -r main
```

### View Workflow Results

1. Go to **Actions** tab in GitHub
2. Click the workflow run
3. View logs for each step
4. Download artifacts (outputs, models, plots)

---

## 🐳 Docker Setup

### Build Docker Image
```bash
docker build -t iris-mlops:latest .
```

### Run Pipeline in Container
```bash
docker run \
  -v $(pwd)/dvc_remote:/app/dvc_remote \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/models:/app/models \
  iris-mlops:latest
```

### Run MLflow UI in Container
```bash
docker run -p 5000:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  iris-mlops:latest \
  mlflow ui --host 0.0.0.0
```

---

## ☸️ Kubernetes Deployment (Minikube)

### Prerequisites
```bash
minikube version  # Verify installation
# minikube version: v1.37.0
```

### Create Kubernetes Manifests

**k8s/iris-job.yaml**

### Deploy to Minikube
```bash
# Start Minikube
minikube start

# Build image in Minikube context
eval $(minikube docker-env)
docker build -t iris-mlops:latest .

# Create namespace
kubectl create namespace mlops

# Deploy job
kubectl apply -f k8s/iris-job.yaml -n mlops

# Monitor job
kubectl get jobs -n mlops
kubectl logs job/iris-pipeline-job -n mlops -f

# Get outputs
kubectl cp mlops/iris-pipeline-job:/app/outputs ./k8s_outputs
```

---

## 📁 Project Structure

```
mlops-mlflow-assignment/
├── data/
│   ├── raw/
│   │   └── iris.csv              # Raw dataset (tracked by DVC)
│   └── processed/
│       ├── train.csv             # Training data
│       ├── test.csv              # Test data
│       └── scaler.pkl            # StandardScaler artifact
├── src/
│   └── steps/
│       ├── extract_data.py       # Extract step
│       ├── preprocess.py         # Preprocess step
│       ├── train.py              # Training step
│       └── evaluate.py           # Evaluation step
├── models/
│   └── random_forest_model.pkl   # Trained model artifact
├── outputs/
│   ├── confusion_matrix.png      # Evaluation plot
│   └── feature_importance.png    # Feature importance plot
├── k8s/
│   └── iris-job.yaml             # Kubernetes job manifest
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    # GitHub Actions workflow
├── .gitignore                    # Git ignore rules
├── .dvc/
│   ├── config                    # DVC configuration
│   └── .gitignore
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                       # DVC lock file (reproducibility)
├── pipeline.py                   # Main pipeline orchestration
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔐 DVC Remote Configuration

### Local Remote (default)
```bash
dvc remote add -d local dvc_remote
dvc push   # Push data to dvc_remote/
dvc pull   # Pull data from dvc_remote/
```

### AWS S3 Remote
```bash
dvc remote add -d s3remote s3://my-bucket/dvc-storage
dvc remote modify s3remote credentialpath ~/.aws/credentials
dvc push -r s3remote
```

### Google Drive Remote
```bash
dvc remote add -d gdrive 'gdrive://FOLDER_ID'
dvc push -r gdrive
```

---

##  Metrics and Artifacts

### Model Performance
- **Test Accuracy**: ~1.0 (100%)
- **F1-Score (weighted)**: ~1.0 (100%)

### Artifacts Generated
- `data/processed/train.csv` - Training dataset (120 samples)
- `data/processed/test.csv` - Test dataset (30 samples)
- `data/processed/scaler.pkl` - StandardScaler for feature scaling
- `models/random_forest_model.pkl` - Trained Random Forest model
- `outputs/confusion_matrix.png` - Confusion matrix heatmap
- `outputs/feature_importance.png` - Feature importance bar plot

---

## 🧪 Testing

Run tests (if implemented):
```bash
pytest tests/ -v
```

Lint code:
```bash
pylint src/ --disable=C0111,C0103,W0212
```

---

## 🐛 Troubleshooting

### DVC Issues
```bash
# Clear DVC cache and retry
dvc cache remove
dvc repro

# Check DVC status
dvc status
dvc dag  # View pipeline DAG
```

### MLflow Issues
```bash
# Reset MLflow runs
rm -rf mlruns/

# Restart MLflow UI
mlflow ui
```

### Pipeline Failures
```bash
# Check individual step logs
python -c "from src.steps.extract_data import extract_data; extract_data()"

# Enable verbose logging
export MLFLOW_TRACKING_URI=http://localhost:5000
python pipeline.py
```

---

##  References

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---
---

## 📄 License

This project is part of an educational MLOps assignment.

---

---

