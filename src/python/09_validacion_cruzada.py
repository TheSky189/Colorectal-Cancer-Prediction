from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"

TARGET = "Cancer"

NUMERIC_FEATURES = ["Age"]

CATEGORICAL_FEATURES = [
    "Gender",
    "Family_History",
    "Smoking_History",
    "Alcohol_Consumption",
    "Obesity_BMI",
    "Diet_Risk",
    "Physical_Activity",
    "Diabetes",
    "Inflammatory_Bowel_Disease",
    "Genetic_Mutation",
    "Urban_or_Rural",
    "Economic_Classification",
    "Healthcare_Access",
    "Insurance_Status",
]

def construir_pipeline():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline

def validar():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    pipeline = construir_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    print("ROC AUC por fold:")
    print(np.round(auc_scores, 4))
    print(f"\nMedia ROC AUC: {auc_scores.mean():.4f}")
    print(f"Desviación estándar: {auc_scores.std():.4f}")

if __name__ == "__main__":
    validar()