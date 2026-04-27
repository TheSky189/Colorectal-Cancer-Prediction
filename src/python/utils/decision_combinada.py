from typing import Dict, Optional
from dataclasses import dataclass, field


# Niveles y umbrales

# Los umbrales para ML son los clasicos (0.3 / 0.7) que usa el RF.
# Los umbrales para literatura son los de riesgo absoluto a 5 años
# segun guidelines clinicas (>5% a 5 años = alto).
UMBRAL_ML_BAJO = 0.30
UMBRAL_ML_ALTO = 0.70

UMBRAL_LIT_BAJO = 0.015   # <1.5% a 5 años = bajo
UMBRAL_LIT_ALTO = 0.05    # >5% a 5 años = alto


def nivel_ml(prob: float) -> str:
    if prob >= UMBRAL_ML_ALTO:
        return "Alto"
    if prob >= UMBRAL_ML_BAJO:
        return "Medio"
    return "Bajo"


def nivel_literatura(riesgo_5a: float) -> str:
    if riesgo_5a >= UMBRAL_LIT_ALTO:
        return "Alto"
    if riesgo_5a >= UMBRAL_LIT_BAJO:
        return "Medio"
    return "Bajo"



# Mapeo de niveles a valores numericos para ponderacion

NIVEL_A_NUM = {"Bajo": 0, "Medio": 1, "Alto": 2}
NUM_A_NIVEL = {0: "Bajo", 1: "Medio", 2: "Alto"}


@dataclass
class DecisionCombinada:
    """Estructura de salida de la decision combinada."""
    # Resultado final
    nivel_final: str                    # "Bajo" / "Medio" / "Alto"
    score_combinado: float              # 0 - 1 (para el gauge visual)

    # Componentes
    nivel_ml: str
    nivel_literatura: str
    prob_ml: float
    riesgo_literatura_5a: float

    # Logica aplicada
    override_aplicado: bool             # True si la regla clinica dura modifico el resultado
    razon_override: Optional[str]       # motivo del override (si aplicado)
    discrepancia: bool                  # True si ML y literatura difieren 2+ niveles
    concordancia_descripcion: str       # texto descriptivo

    # Mensaje de recomendacion clinica
    recomendacion: str

    # Factores detectados (para mostrar en UI)
    factores_riesgo: list = field(default_factory=list)
    factores_protectores: list = field(default_factory=list)


def _calcular_score_combinado(prob_ml: float, riesgo_lit_5a: float) -> float:
    """
    Combina ambas probabilidades en un score unico de 0 a 1.

    Estrategia:
    - Normalizar literatura: los riesgos absolutos a 5 anos rara vez
      pasan del 20-25% incluso en alto riesgo. Para que el score sea
      comparable con la prob del ML (0-1), escalamos asumiendo que
      un riesgo a 5 anos del 15% equivale a una probabilidad de 1.0
      en la escala "relativa".
    - Ponderar: 70% literatura + 30% ML.
    """
    # Normalizar literatura a escala 0-1 (saturando en 15%)
    lit_normalizada = min(riesgo_lit_5a / 0.15, 1.0)

    # Ponderacion
    score = 0.70 * lit_normalizada + 0.30 * prob_ml
    return min(max(score, 0.0), 1.0)


def _aplicar_reglas_clinicas_duras(paciente: Dict, nivel_actual: str) -> tuple:
    """
    Returns:
        (nivel_resultante, override_aplicado, razon_override)
    """
    factores_presentes = []

    # Regla 1: mutacion genetica conocida -> ALTO obligatorio
    # Sindromes hereditarios (Lynch, FAP, MUTYH) elevan mucho el riesgo.
    if paciente.get("Genetic_Mutation") == "Yes":
        factores_presentes.append("mutacion genetica")

    # Regla 2: enfermedad inflamatoria intestinal -> ALTO obligatorio
    # EII con >8 años de evolucion tiene RR ~3 para CCR.
    if paciente.get("Inflammatory_Bowel_Disease") == "Yes":
        factores_presentes.append("enfermedad inflamatoria intestinal")

    # Si hay alguno de estos dos factores, forzar ALTO
    if factores_presentes:
        razon = (
            "Por presencia de " + " + ".join(factores_presentes) +
            ", la evidencia clinica recomienda estratificacion como RIESGO ALTO "
            "independientemente del score estadistico."
        )
        return "Alto", nivel_actual != "Alto", razon

    # Regla 3: combinacion multiple de factores de riesgo modificables
    # Si hay >=4 factores de riesgo simultaneos, elevar a MEDIO al menos
    factores_riesgo_mod = 0
    if paciente.get("Family_History") == "Yes": factores_riesgo_mod += 1
    if paciente.get("Smoking_History") == "Yes": factores_riesgo_mod += 1
    if paciente.get("Alcohol_Consumption") == "Yes": factores_riesgo_mod += 1
    if paciente.get("Obesity_BMI") == "Obese": factores_riesgo_mod += 1
    if paciente.get("Diet_Risk") == "High": factores_riesgo_mod += 1
    if paciente.get("Physical_Activity") == "Low": factores_riesgo_mod += 1
    if paciente.get("Diabetes") == "Yes": factores_riesgo_mod += 1

    if factores_riesgo_mod >= 5 and nivel_actual == "Bajo":
        razon = (
            f"Concurrencia de {factores_riesgo_mod} factores de riesgo modificables. "
            "Se eleva el nivel minimo a MEDIO para recomendar intervencion "
            "preventiva sobre estilo de vida."
        )
        return "Medio", True, razon

    return nivel_actual, False, None


def _detectar_discrepancia(nivel_ml: str, nivel_lit: str) -> tuple:
    """Detecta discrepancia de 2 niveles entre ambos motores."""
    n_ml = NIVEL_A_NUM[nivel_ml]
    n_lit = NIVEL_A_NUM[nivel_lit]
    diff = abs(n_ml - n_lit)

    if diff >= 2:
        return True, f"Discrepancia fuerte: ML={nivel_ml} vs Literatura={nivel_lit}"
    if diff == 1:
        return False, f"Discrepancia leve: ML={nivel_ml} vs Literatura={nivel_lit}"
    return False, f"Ambos motores coinciden en nivel {nivel_ml}"


def _generar_recomendacion(nivel: str, override: bool, discrepancia: bool) -> str:
    if override:
        return (
            "Se recomienda evaluacion por gastroenterologo y realizar "
            "colonoscopia segun guias NCCN/ESMO. El paciente tiene "
            "indicacion clinica directa."
        )

    if discrepancia:
        return (
            "Se detecta discrepancia entre motores. Se recomienda revision "
            "clinica individualizada ya que los resultados pueden ser no "
            "concluyentes. Considerar factores no incluidos en el modelo."
        )

    if nivel == "Alto":
        return (
            "RIESGO ALTO: se recomienda evaluacion gastroenterologica y "
            "realizar colonoscopia segun guias NCCN/ESMO. Modificar factores "
            "de riesgo modificables (dieta, tabaco, actividad fisica)."
        )
    if nivel == "Medio":
        return (
            "RIESGO INTERMEDIO: se recomienda cribado segun guias (colonoscopia "
            "o test de sangre oculta) y revision de factores modificables. "
            "Revisar cada 1-2 anos."
        )
    return (
        "RIESGO BAJO: mantener cribado estandar segun edad (colonoscopia a "
        "partir de los 45-50 anos). Mantener habitos saludables."
    )


def calcular_decision_combinada(
    paciente: Dict,
    prob_ml: float,
    resultado_literatura: Dict,
) -> DecisionCombinada:
    """
    Funcion principal de decision combinada.

    Args:
        paciente: dict con los campos del paciente (debe incluir al menos
                  Genetic_Mutation, Inflammatory_Bowel_Disease, etc.)
        prob_ml: probabilidad [0, 1] del modelo ML calibrado.
        resultado_literatura: dict devuelto por calcular_riesgo_literatura().
                              Debe tener 'riesgo_5_anos' y
                              'factores_aplicados'.

    Returns:
        DecisionCombinada con toda la informacion para la UI.
    """
    # 1. Clasificacion por cada motor
    n_ml = nivel_ml(prob_ml)
    riesgo_lit_5a = resultado_literatura["riesgo_5_anos"]
    n_lit = nivel_literatura(riesgo_lit_5a)

    # 2. Score combinado (ponderado)
    score = _calcular_score_combinado(prob_ml, riesgo_lit_5a)

    # 3. Nivel inicial segun score ponderado
    if score >= 0.55:
        nivel_inicial = "Alto"
    elif score >= 0.25:
        nivel_inicial = "Medio"
    else:
        nivel_inicial = "Bajo"

    # 4. Aplicar reglas clinicas duras
    nivel_final, override, razon = _aplicar_reglas_clinicas_duras(
        paciente, nivel_inicial
    )

    # 5. Detectar discrepancias entre motores
    discrepancia, descripcion_concordancia = _detectar_discrepancia(n_ml, n_lit)

    # 6. Recomendacion clinica
    recomendacion = _generar_recomendacion(nivel_final, override, discrepancia)

    # 7. Factores detectados (para chips en UI)
    factores_riesgo, factores_protectores = _listar_factores(
        paciente, resultado_literatura.get("factores_aplicados", {})
    )

    return DecisionCombinada(
        nivel_final=nivel_final,
        score_combinado=score,
        nivel_ml=n_ml,
        nivel_literatura=n_lit,
        prob_ml=prob_ml,
        riesgo_literatura_5a=riesgo_lit_5a,
        override_aplicado=override,
        razon_override=razon,
        discrepancia=discrepancia,
        concordancia_descripcion=descripcion_concordancia,
        recomendacion=recomendacion,
        factores_riesgo=factores_riesgo,
        factores_protectores=factores_protectores,
    )


# Listado de factores para UI

FACTOR_NOMBRE_ES = {
    "Family_History": "Antecedentes familiares",
    "Inflammatory_Bowel_Disease": "EII",
    "Genetic_Mutation": "Mutacion genetica",
    "Smoking_History": "Tabaquismo",
    "Alcohol_Consumption": "Alcohol",
    "Obesity_BMI": "Obesidad",
    "Diet_Risk": "Dieta alto riesgo",
    "Physical_Activity": "Actividad fisica",
    "Diabetes": "Diabetes tipo 2",
}


def _listar_factores(paciente: Dict, factores_rr: Dict) -> tuple:
    """Devuelve (factores_riesgo, factores_protectores) ordenados por RR."""
    factores_riesgo = []
    factores_protectores = []

    for clave, rr in factores_rr.items():
        nombre = FACTOR_NOMBRE_ES.get(clave, clave)
        item = {"nombre": nombre, "rr": rr, "clave": clave}
        if rr > 1.01:
            factores_riesgo.append(item)
        elif rr < 0.99:
            factores_protectores.append(item)

    factores_riesgo.sort(key=lambda x: x["rr"], reverse=True)
    factores_protectores.sort(key=lambda x: x["rr"])

    return factores_riesgo, factores_protectores


# Test rapido
if __name__ == "__main__":
    from calculadora_riesgo_literatura import calcular_riesgo_literatura

    print("=" * 60)
    print("  TEST: PACIENTE ALTO RIESGO EXTREMO")
    print("=" * 60)
    paciente_alto = {
        "Age": 68, "Gender": "M",
        "Family_History": "Yes", "Smoking_History": "Yes",
        "Alcohol_Consumption": "Yes", "Obesity_BMI": "Obese",
        "Diet_Risk": "High", "Physical_Activity": "Low",
        "Diabetes": "Yes", "Inflammatory_Bowel_Disease": "Yes",
        "Genetic_Mutation": "Yes", "Urban_or_Rural": "Urban",
        "Healthcare_Access": "Moderate",
    }

    lit_alto = calcular_riesgo_literatura(paciente_alto)
    decision = calcular_decision_combinada(
        paciente_alto, prob_ml=0.66, resultado_literatura=lit_alto
    )
    print(f"Nivel ML:         {decision.nivel_ml} ({decision.prob_ml:.2%})")
    print(f"Nivel Literatura: {decision.nivel_literatura} ({decision.riesgo_literatura_5a:.2%} a 5 anos)")
    print(f"Score combinado:  {decision.score_combinado:.3f}")
    print(f"Nivel FINAL:      {decision.nivel_final}")
    print(f"Override:         {decision.override_aplicado}")
    if decision.razon_override:
        print(f"  Razon: {decision.razon_override}")
    print(f"Recomendacion:    {decision.recomendacion}")

    print("\n" + "=" * 60)
    print("  TEST: PACIENTE BAJO RIESGO CLARO")
    print("=" * 60)
    paciente_bajo = {
        "Age": 35, "Gender": "F",
        "Family_History": "No", "Smoking_History": "No",
        "Alcohol_Consumption": "No", "Obesity_BMI": "Normal",
        "Diet_Risk": "Low", "Physical_Activity": "High",
        "Diabetes": "No", "Inflammatory_Bowel_Disease": "No",
        "Genetic_Mutation": "No", "Urban_or_Rural": "Urban",
        "Healthcare_Access": "High",
    }

    lit_bajo = calcular_riesgo_literatura(paciente_bajo)
    decision = calcular_decision_combinada(
        paciente_bajo, prob_ml=0.15, resultado_literatura=lit_bajo
    )
    print(f"Nivel ML:         {decision.nivel_ml} ({decision.prob_ml:.2%})")
    print(f"Nivel Literatura: {decision.nivel_literatura} ({decision.riesgo_literatura_5a:.2%} a 5 anos)")
    print(f"Score combinado:  {decision.score_combinado:.3f}")
    print(f"Nivel FINAL:      {decision.nivel_final}")
    print(f"Recomendacion:    {decision.recomendacion}")