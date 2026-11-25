"""Train the student marks, pass/fail, KNN similarity and clustering models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, f1_score, r2_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import generate_dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FEATURES = [
    "attendance", "study_hours", "previous_marks",
    "assignment_marks", "internal_marks"
]
TARGET = "final_exam_marks"


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "student_performance.csv"
    if not path.exists():
        print("No dataset found. Creating a sample dataset...")
        generate_dataset().to_csv(path, index=False)
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.drop_duplicates().copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())
    df = df.dropna(subset=required)
    df = df[(df[TARGET] >= 0) & (df[TARGET] <= 100)]
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
    X = df[FEATURES]
    y_marks = df[TARGET]
    y_pass = (y_marks >= 40).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_marks, test_size=0.2, random_state=42
    )
    # Regression with cross-validation to estimate generalization.
    regression = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(
        regression, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    ).mean()
    regression.fit(X_train, y_train)
    marks_pred = np.clip(regression.predict(X_test), 0, 100)
    regression_metrics = {
        "mae": float(mean_absolute_error(y_test, marks_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, marks_pred))),
        "r2": float(r2_score(y_test, marks_pred)),
        "cv_mae": float(cv_mae),
    }

    Xtr, Xte, ytr, yte = train_test_split(
        X, y_pass, test_size=0.2, random_state=42, stratify=y_pass
    )
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(logistic, Xtr, ytr, cv=strat_cv, scoring="accuracy").mean()
    logistic.fit(Xtr, ytr)
    pass_pred = logistic.predict(Xte)
    pass_prob = logistic.predict_proba(Xte)[:, 1]
    logistic_metrics = {
        "accuracy": float(accuracy_score(yte, pass_pred)),
        "precision": float(precision_score(yte, pass_pred, zero_division=0)),
        "recall": float(recall_score(yte, pass_pred, zero_division=0)),
        "f1": float(f1_score(yte, pass_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, pass_prob)),
        "cv_accuracy": float(cv_accuracy),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # KNN stores scaled student records to retrieve similar learning profiles.
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_scaled)

    # K-means creates three groups. Labels are assigned after fitting by
    # ranking each cluster's mean final mark, not by assuming cluster IDs.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    cluster_means = pd.DataFrame({"cluster": cluster_ids, "marks": y_marks}).groupby("cluster")["marks"].mean()
    ordered = cluster_means.sort_values().index.tolist()
    cluster_labels = {
        int(ordered[0]): "At-Risk",
        int(ordered[1]): "Average",
        int(ordered[2]): "High Performing",
    }

    bundle = {
        "features": FEATURES,
        "regression": regression,
        "logistic": logistic,
        "scaler": scaler,
        "knn": knn,
        "kmeans": kmeans,
        "cluster_labels": cluster_labels,
        "training_data": X.reset_index(drop=True),
        "training_marks": y_marks.reset_index(drop=True),
        "training_student_ids": df["student_id"].reset_index(drop=True) if "student_id" in df else pd.Series(
            [f"STUDENT-{i+1:04d}" for i in range(len(df))]
        ),
        "metrics": {"regression": regression_metrics, "classification": logistic_metrics},
        "pass_mark": 40,
    }
    joblib.dump(bundle, MODEL_DIR / "student_models.joblib")
    metadata = {
        "dataset_rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "pass_mark": 40,
        "regression_metrics": regression_metrics,
        "classification_metrics": logistic_metrics,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "note": "Demonstration dataset unless replaced with institution-approved student data."
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nTraining completed successfully.")
    print(f"Rows used: {len(df)}")
    print(f"Regression metrics: {regression_metrics}")
    print(f"Pass/fail metrics: {logistic_metrics}")
    print(f"Saved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
