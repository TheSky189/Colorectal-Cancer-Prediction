from pathlib import Path

"""
Predicción individual del Modelo 1 (clínico).
"""
import sys
import joblib
import pandas as pd

# Importar módulos centralizados
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paciente_schema import (
    validar_paciente,
    paciente_ejemplo_alto_riesgo,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
)
from utils.risk_levels import nivel_riesgo, recomendacion_clinica

BASE_DIR = Path(__file__).resolve().parents[2]

# Por defecto cargamos el modelo CALIBRADO (mejores probabilidades).
# Si no existe, fallback al bundle no calibrado.
CALIBRATED_PATH = BASE_DIR / "models" / "calibrated_model.joblib"
BUNDLE_PATH = BASE_DIR / "models" / "clinical_model_bundle.joblib"


def cargar_mejor_modelo():
    """Carga el calibrado si existe; si no, el bundle no calibrado."""
    if CALIBRATED_PATH.exists():
        print(f"Usando modelo calibrado: {CALIBRATED_PATH}")
        return joblib.load(CALIBRATED_PATH)
    if BUNDLE_PATH.exists():
        print(f"Usando bundle no calibrado: {BUNDLE_PATH}")
        bundle = joblib.load(BUNDLE_PATH)
        return bundle["model"]
    raise FileNotFoundError(
        "No se encontró ningún modelo. Ejecuta antes 04_entrenar_modelo.py "
        "y opcionalmente 10_calibracion.py."
    )


def predecir_paciente() -> None:
    modelo = cargar_mejor_modelo()

    paciente = paciente_ejemplo_alto_riesgo()

    # Validación previa (clave: detecta categorías mal escritas).
    errores = validar_paciente(paciente)
    if errores:
        print("ERRORES DE VALIDACIÓN del paciente de ejemplo:")
        for e in errores:
            print(f"  - {e}")
        return


    columnas = NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES
    paciente_df = pd.DataFrame([{c: paciente[c] for c in columnas}])

    prob = float(modelo.predict_proba(paciente_df)[0][1])
    nivel = nivel_riesgo(prob)

    print("\nResultado de predicción individual")
    print("----------------------------------")
    print(f"Probabilidad estimada (modelo ML): {prob:.4f}")
    print(f"Nivel de riesgo: {nivel}")
    print(recomendacion_clinica(nivel))


if __name__ == "__main__":
    predecir_paciente()
