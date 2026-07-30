import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.set_experiment("Heart Disease - CI")

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "SMSML_mamat", "Membangun_model", "heart_preprocessing")
fallback_dir = os.path.join(os.path.dirname(__file__), "heart_preprocessing")
if os.path.exists(fallback_dir):
    DATA_DIR = fallback_dir

df = pd.read_csv(os.path.join(DATA_DIR, 'heartdiseasedataset_preprocessing.csv'))
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train[:5]
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")
