from typing import Tuple

# Umbrales para el MODELO ML
UMBRAL_BAJO_MEDIO: float = 0.30
UMBRAL_MEDIO_ALTO: float = 0.70

# Umbrales para la CALCULADORA basada en LITERATURA
# (escala = riesgo absoluto a 5 años)
UMBRAL_LIT_BAJO_MEDIO: float = 0.015   # 1.5%
UMBRAL_LIT_MEDIO_ALTO: float = 0.05    # 5%

# Etiquetas
NIVEL_BAJO: str = "Bajo"
NIVEL_MEDIO: str = "Medio"
NIVEL_ALTO: str = "Alto"


def nivel_riesgo(probabilidad: float) -> str:
    """
    Devuelve el nivel de riesgo categórico para una probabilidad
    procedente del MODELO ML.

    Parameters
    ----------
    probabilidad : float
        Valor entre 0 y 1.

    Returns
    -------
    str : "Bajo", "Medio" o "Alto"
    """
    if probabilidad < UMBRAL_BAJO_MEDIO:
        return NIVEL_BAJO
    if probabilidad < UMBRAL_MEDIO_ALTO:
        return NIVEL_MEDIO
    return NIVEL_ALTO


def nivel_riesgo_literatura(riesgo_5_anos: float) -> str:
    """
    Devuelve el nivel de riesgo categórico para un riesgo absoluto
    a 5 años procedente de la CALCULADORA basada en literatura.

    Usa umbrales realistas:
        <1.5%  = Bajo
        1.5-5% = Medio
        >5%    = Alto
    """
    if riesgo_5_anos < UMBRAL_LIT_BAJO_MEDIO:
        return NIVEL_BAJO
    if riesgo_5_anos < UMBRAL_LIT_MEDIO_ALTO:
        return NIVEL_MEDIO
    return NIVEL_ALTO


def recomendacion_clinica(nivel: str) -> str:
    """Texto explicativo recomendado según el nivel de riesgo."""
    if nivel == NIVEL_ALTO:
        return (
            "Riesgo alto. Se recomienda evaluación diagnóstica adicional "
            "(p. ej. colonoscopia y/o análisis de imagen CT)."
        )
    if nivel == NIVEL_MEDIO:
        return (
            "Riesgo intermedio. Se recomienda seguimiento clínico, "
            "revisión del cribado y modificación de factores de riesgo "
            "modificables (dieta, tabaquismo, actividad física)."
        )
    return (
        "Riesgo bajo. Mantener el cribado preventivo habitual según "
        "la edad y guías locales."
    )


def rango_nivel(nivel: str, fuente: str = "ml") -> Tuple[float, float]:

    if fuente == "literatura":
        bajo, alto = UMBRAL_LIT_BAJO_MEDIO, UMBRAL_LIT_MEDIO_ALTO
    else:
        bajo, alto = UMBRAL_BAJO_MEDIO, UMBRAL_MEDIO_ALTO

    if nivel == NIVEL_BAJO:
        return (0.0, bajo)
    if nivel == NIVEL_MEDIO:
        return (bajo, alto)
    return (alto, 1.0)

"""
Niveles de riesgo unificados para todo el sistema.

1. Modelo ML (entrenado con dataset Kaggle + controles sintéticos):
   produce probabilidades en una escala 0-1 que NO se corresponden
   con riesgo absoluto de cáncer en una población real. Los umbrales
   0.30 / 0.70 son operativos para clasificar pacientes en este dataset.

2. Calculadora basada en literatura (NCI-style):
   produce un riesgo ABSOLUTO de desarrollar cáncer en 5 años. Los
   umbrales clínicamente significativos son mucho más bajos:
   - <1.5%   = riesgo bajo (en línea con la población general 50+)
   - 1.5-5%  = riesgo medio (similar a tener un familiar de primer grado)
   - >5%     = riesgo alto (similar a EII o múltiples factores combinados)

Justificación de los umbrales del modelo ML:
- Bajo (< 0.30):  riesgo basal poblacional. Cribado estándar.
- Medio (0.30-0.70): riesgo elevado. Evaluación clínica.
- Alto (>= 0.70): riesgo significativo. Evaluación diagnóstica adicional.
"""