from pathlib import Path
import sys
import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paciente_schema import (
    validar_paciente,
    paciente_ejemplo_alto_riesgo,
    paciente_ejemplo_bajo_riesgo,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
)
from utils.risk_levels import (
    nivel_riesgo,
    nivel_riesgo_literatura,
    recomendacion_clinica,
)
from utils.calculadora_riesgo_literatura import calcular_riesgo_literatura

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "calibrated_model.joblib"


def predecir_un_paciente(modelo, paciente: dict, etiqueta: str) -> None:
    print(f"\n========= {etiqueta} =========")

    errores = validar_paciente(paciente)
    if errores:
        print("ERRORES DE VALIDACIÓN:")
        for e in errores:
            print(f"  - {e}")
        return

    columnas = NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES
    paciente_df = pd.DataFrame([{c: paciente[c] for c in columnas}])

    # Predicción del modelo ML
    prob_ml = float(modelo.predict_proba(paciente_df)[0][1])
    nivel_ml = nivel_riesgo(prob_ml)

    # Predicción de la calculadora basada en literatura
    # IMPORTANTE: usamos nivel_riesgo_literatura (umbrales 1.5%/5%)
    # porque la calculadora produce un riesgo absoluto a 5 años, no
    # una probabilidad en la escala del modelo ML.
    resultado_lit = calcular_riesgo_literatura(paciente)
    prob_lit = resultado_lit["riesgo_5_anos"]
    nivel_lit = nivel_riesgo_literatura(prob_lit)

    print(f"[Modelo ML]         Probabilidad: {prob_ml:.4f}  → Nivel: {nivel_ml}")
    print(f"[Calc. literatura]  Riesgo 5 años: {prob_lit:.4f}  → Nivel: {nivel_lit}")
    print(f"\nRecomendación (basada en modelo ML): {recomendacion_clinica(nivel_ml)}")


def test() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo calibrado: {MODEL_PATH}. "
            "Ejecuta antes 04_entrenar_modelo.py y 10_calibracion.py."
        )

    modelo = joblib.load(MODEL_PATH)

    predecir_un_paciente(
        modelo, paciente_ejemplo_alto_riesgo(),
        "Paciente de ejemplo: ALTO RIESGO",
    )

    predecir_un_paciente(
        modelo, paciente_ejemplo_bajo_riesgo(),
        "Paciente de ejemplo: BAJO RIESGO",
    )


if __name__ == "__main__":
    test()


"""
Estratificación de riesgo: prueba con varios pacientes de ejemplo.
"""