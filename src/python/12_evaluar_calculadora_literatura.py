from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paciente_schema import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
    TARGET,
)
from utils.risk_levels import nivel_riesgo, nivel_riesgo_literatura
from utils.calculadora_riesgo_literatura import calcular_riesgo_literatura

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_PATH = BASE_DIR / "data" / "processed" / "test_dataset.csv"
CALIBRATED_PATH = BASE_DIR / "models" / "calibrated_model.joblib"
OUTPUT_DIR = BASE_DIR / "models" / "comparacion_motores"


def predecir_literatura_batch(df: pd.DataFrame) -> np.ndarray:
    """Aplica la calculadora basada en literatura fila a fila."""
    probas = []
    for _, fila in df.iterrows():
        paciente = {c: fila[c] for c in NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES}
        resultado = calcular_riesgo_literatura(paciente)
        probas.append(resultado["riesgo_5_anos"])
    return np.array(probas)


def comparar() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"No existe el test set: {TEST_PATH}")
    if not CALIBRATED_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo calibrado: {CALIBRATED_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEST_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES]
    y = df[TARGET].values

    modelo_ml = joblib.load(CALIBRATED_PATH)

    print(f"Aplicando modelo ML a {len(df)} pacientes...")
    prob_ml = modelo_ml.predict_proba(X)[:, 1]

    print(f"Aplicando calculadora literatura a {len(df)} pacientes...")
    prob_lit = predecir_literatura_batch(X)

    
    # 1. Estadísticas descriptivas de cada motor
    print("\n=== Distribución de probabilidades ===")
    print(f"ML        — media: {prob_ml.mean():.4f}  std: {prob_ml.std():.4f}  "
          f"min: {prob_ml.min():.4f}  max: {prob_ml.max():.4f}")
    print(f"Literatura — media: {prob_lit.mean():.4f}  std: {prob_lit.std():.4f}  "
          f"min: {prob_lit.min():.4f}  max: {prob_lit.max():.4f}")


    # 2. AUC y Brier (el AUC del literatura sobre este test es informativo,
    #    no clínicamente significativo, por la composición sintética del
    #    grupo control)
    print("\n=== Métricas sobre test set (con cautela: ver docstring) ===")
    print(f"AUC ML:         {roc_auc_score(y, prob_ml):.4f}")
    print(f"AUC Literatura: {roc_auc_score(y, prob_lit):.4f}")
    print(f"Brier ML:         {brier_score_loss(y, prob_ml):.4f}")
    print(f"Brier Literatura: {brier_score_loss(y, prob_lit):.4f}")


    # 3. Correlación entre ambos motores
    rho, pval = spearmanr(prob_ml, prob_lit)
    print(f"\n=== Concordancia entre motores ===")
    print(f"Spearman ρ: {rho:.4f}  (p = {pval:.2e})")


    # 4. Matriz de concordancia de niveles
    nivel_ml_arr = np.array([nivel_riesgo(p) for p in prob_ml])
    nivel_lit_arr = np.array([nivel_riesgo_literatura(p) for p in prob_lit])

    niveles = ["Bajo", "Medio", "Alto"]
    matriz = pd.DataFrame(
        0,
        index=[f"ML={n}" for n in niveles],
        columns=[f"Lit={n}" for n in niveles],
    )
    for n_ml, n_lit in zip(nivel_ml_arr, nivel_lit_arr):
        matriz.loc[f"ML={n_ml}", f"Lit={n_lit}"] += 1

    print("\n=== Matriz de concordancia de niveles ===")
    print(matriz)

    concordancia = (nivel_ml_arr == nivel_lit_arr).mean()
    print(f"\nConcordancia exacta de niveles: {concordancia*100:.2f}%")


    # 5. Guardar resultados
    df_resultados = X.copy()
    df_resultados["Cancer"] = y
    df_resultados["prob_ml"] = prob_ml
    df_resultados["prob_literatura"] = prob_lit
    df_resultados["nivel_ml"] = nivel_ml_arr
    df_resultados["nivel_literatura"] = nivel_lit_arr

    salida = OUTPUT_DIR / "comparacion_test.csv"
    df_resultados.to_csv(salida, index=False)
    matriz.to_csv(OUTPUT_DIR / "matriz_concordancia.csv")

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    comparar()


"""
Comparación de la calculadora basada en literatura con el modelo ML.
"""