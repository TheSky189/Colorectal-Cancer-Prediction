from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score

# Importar el esquema centralizado
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paciente_schema import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
    TARGET,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
OUTPUT_PATH = BASE_DIR / "models" / "calibrated_model.joblib"


def construir_pipeline_no_entrenado() -> Pipeline:
    """
    Construye un pipeline NUEVO sin entrenar.

    Importante: CalibratedClassifierCV requiere un estimador NO ajustado.
    Por eso este pipeline se construye desde cero en lugar de cargar el
    bundle ya entrenado.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURE_NAMES),
    ])
    return Pipeline(steps=[
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


def calibrar() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES]
    y = df[TARGET]

    # Split externo: nunca se usa para entrenar ni calibrar.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=42,
    )

    # Pipeline NUEVO, sin entrenar — CalibratedClassifierCV lo entrenará
    # internamente con cv=3 (3 folds disjuntos para entrenar y calibrar).
    base_pipeline = construir_pipeline_no_entrenado()

    calibrated = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="isotonic",
        cv=3,
    )

    print("Ajustando calibración isotónica con CV=3...")
    calibrated.fit(X_train, y_train)

    # Evaluación honesta sobre el test set externo
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"AUC sobre test externo: {auc:.4f}")
    print(f"Brier score sobre test externo: {brier:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, OUTPUT_PATH)
    print(f"Modelo calibrado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    calibrar()

"""
Calibración del Modelo 1 mediante CalibratedClassifierCV.
- Construye un pipeline NUEVO (no entrenado).
- CalibratedClassifierCV se encarga de hacer cv=3 internamente, entrenando
  el estimador y calibrando en folds disjuntos.
- Se evalúa sobre un test set externo nunca usado para entrenar ni calibrar.
"""