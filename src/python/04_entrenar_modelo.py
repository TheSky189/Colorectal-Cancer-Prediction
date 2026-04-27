from pathlib import Path

"""
Entrenamiento del Modelo 1 
"""

import json
import joblib
import sys
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Importar el esquema centralizado
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paciente_schema import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
    TARGET,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "models" / "clinical_model_bundle.joblib"
METRICS_OUTPUT_PATH = BASE_DIR / "models" / "metricas_modelo.json"
TEST_SET_PATH = BASE_DIR / "data" / "processed" / "test_dataset.csv"


def construir_preprocesador() -> ColumnTransformer:
    """Construye el ColumnTransformer usando el esquema centralizado."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURE_NAMES),
        ]
    )


def evaluar(nombre: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def entrenar() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {DATA_PATH}. Ejecuta antes los scripts 01, 02 y 03."
        )

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("El dataset de entrenamiento está vacío.")

    # Verificación de columnas: el dataset debe contener TODAS las que
    # nuestro esquema espera. Esto evita errores silenciosos.
    columnas_esperadas = set(NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES + [TARGET])
    columnas_faltantes = columnas_esperadas - set(df.columns)
    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas en el dataset de entrenamiento: {columnas_faltantes}"
        )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    # Guardar test set para evaluaciones posteriores
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    TEST_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_SET_PATH, index=False)

    preprocessor = construir_preprocesador()

    pipeline_lr = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    pipeline_rf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Entrenando Logistic Regression...")
    pipeline_lr.fit(X_train, y_train)
    res_lr = evaluar("LogisticRegression", pipeline_lr, X_test, y_test)

    print("Entrenando Random Forest...")
    pipeline_rf.fit(X_train, y_train)
    res_rf = evaluar("RandomForest", pipeline_rf, X_test, y_test)

    print("\nResultados Logistic Regression:", res_lr)
    print("\nResultados Random Forest:", res_rf)

    # Selección del mejor modelo por ROC AUC
    if res_rf["roc_auc"] >= res_lr["roc_auc"]:
        mejor_modelo, mejor_resultado = pipeline_rf, res_rf
    else:
        mejor_modelo, mejor_resultado = pipeline_lr, res_lr

    bundle = {
        "model": mejor_modelo,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURE_NAMES,
        "target": TARGET,
        "best_metrics": mejor_resultado,
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "logistic_regression": res_lr,
                "random_forest": res_rf,
                "best_model": mejor_resultado,
            },
            f, indent=4, ensure_ascii=False,
        )

    print(f"\nModelo guardado en: {MODEL_OUTPUT_PATH}")
    print(f"Métricas guardadas en: {METRICS_OUTPUT_PATH}")
    print(f"Test set guardado en: {TEST_SET_PATH}")
    print(f"\nMejor modelo seleccionado: {mejor_resultado}")


if __name__ == "__main__":
    entrenar()
