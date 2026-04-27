from typing import Dict, List
"""
Esquema centralizado para los datos de entrada de un paciente del Modelo 1.
"""


# Variables numéricas

NUMERIC_FEATURES: List[str] = ["Age"]


# Variables categóricas y sus valores permitidos

CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "Gender": ["M", "F"],
    "Family_History": ["Yes", "No"],
    "Smoking_History": ["Yes", "No"],
    "Alcohol_Consumption": ["Yes", "No"],
    "Obesity_BMI": ["Normal", "Overweight", "Obese"],
    "Diet_Risk": ["Low", "Moderate", "High"],
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diabetes": ["Yes", "No"],
    "Inflammatory_Bowel_Disease": ["Yes", "No"],
    "Genetic_Mutation": ["Yes", "No"],
    "Urban_or_Rural": ["Urban", "Rural"],
    "Healthcare_Access": ["Low", "Moderate", "High"],
}

CATEGORICAL_FEATURE_NAMES: List[str] = list(CATEGORICAL_FEATURES.keys())


# Descripciones legibles para cada variable y opción.

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Age": (
        "Edad del paciente en años. La edad es el factor de riesgo no "
        "modificable más importante: la incidencia se multiplica ~3× "
        "entre los 50 y los 70 años."
    ),
    "Gender": (
        "Sexo biológico. Los hombres tienen una incidencia aproximadamente "
        "32% mayor que las mujeres (SEER)."
    ),
    "Family_History": (
        "¿Tiene algún familiar de primer grado (padre, madre, hermano, hijo) "
        "diagnosticado de cáncer colorrectal? Un familiar afectado "
        "aproximadamente DUPLICA el riesgo."
    ),
    "Smoking_History": (
        "¿Ha fumado regularmente durante al menos 10 años? El tabaquismo "
        "prolongado aumenta el riesgo ~18%."
    ),
    "Alcohol_Consumption": (
        "¿Consume alcohol de forma regular (≥1 bebida al día de media)? "
        "El consumo moderado-alto aumenta el riesgo ~21%."
    ),
    "Obesity_BMI": (
        "Categoría de índice de masa corporal (IMC = peso / altura²):\n"
        "- Normal: IMC 18.5 – 24.9\n"
        "- Overweight (sobrepeso): IMC 25 – 29.9\n"
        "- Obese (obesidad): IMC ≥ 30\n"
        "La obesidad aumenta el riesgo ~33%."
    ),
    "Diet_Risk": (
        "Nivel de riesgo dietético asociado a cáncer colorrectal:\n"
        "- LOW: dieta mediterránea o rica en fibra, frutas, verduras, "
        "legumbres, pescado. Poco consumo de carne roja/procesada.\n"
        "- MODERATE: dieta mixta occidental estándar, sin excesos "
        "claros.\n"
        "- HIGH: alta en carnes procesadas (embutidos, bacon, salchichas) "
        "y/o carne roja (>500g/semana), baja en fibra, muchos "
        "ultraprocesados.\n"
        "La IARC clasifica la carne procesada como carcinógeno del "
        "Grupo 1 para cáncer colorrectal."
    ),
    "Physical_Activity": (
        "Nivel habitual de actividad física:\n"
        "- LOW: sedentario, <30 min/semana de ejercicio moderado.\n"
        "- MODERATE: 30-150 min/semana de ejercicio moderado.\n"
        "- HIGH: >150 min/semana o ejercicio vigoroso regular.\n"
        "La actividad alta REDUCE el riesgo ~24%."
    ),
    "Diabetes": (
        "¿Tiene diagnóstico de diabetes tipo 2? La diabetes aumenta el "
        "riesgo ~27% (posible mediación por hiperinsulinismo)."
    ),
    "Inflammatory_Bowel_Disease": (
        "¿Tiene enfermedad inflamatoria intestinal (enfermedad de Crohn o "
        "colitis ulcerosa)? Triplica aproximadamente el riesgo (RR ~2.93). "
        "Es uno de los factores no genéticos de mayor peso."
    ),
    "Genetic_Mutation": (
        "¿Tiene alguna mutación genética conocida asociada a cáncer "
        "colorrectal hereditario? Ejemplos: síndrome de Lynch (MLH1, MSH2), "
        "poliposis adenomatosa familiar (APC), MUTYH. Pueden elevar el "
        "riesgo hasta 4-5 veces o más según el síndrome."
    ),
    "Urban_or_Rural": (
        "Entorno habitual de residencia. Las poblaciones urbanas tienden "
        "a presentar dietas más occidentalizadas y mayor sedentarismo, "
        "aunque la asociación causal directa con CCR es modesta."
    ),
    "Healthcare_Access": (
        "Nivel de acceso a servicios sanitarios y cribado:\n"
        "- LOW: acceso muy limitado, sin cribado regular.\n"
        "- MODERATE: acceso básico, cribado ocasional.\n"
        "- HIGH: acceso completo, cribado regular según guías.\n"
        "No afecta al riesgo biológico, pero sí a la detección precoz "
        "de lesiones premalignas (pólipos)."
    ),
}

# Etiquetas legibles para el usuario (español) — para UI más limpia
FEATURE_LABELS: Dict[str, str] = {
    "Age": "Edad",
    "Gender": "Género",
    "Family_History": "Antecedentes familiares de CCR",
    "Smoking_History": "Historial de tabaquismo",
    "Alcohol_Consumption": "Consumo regular de alcohol",
    "Obesity_BMI": "Categoría de IMC",
    "Diet_Risk": "Riesgo dietético",
    "Physical_Activity": "Actividad física",
    "Diabetes": "Diabetes tipo 2",
    "Inflammatory_Bowel_Disease": "Enfermedad inflamatoria intestinal",
    "Genetic_Mutation": "Mutación genética conocida",
    "Urban_or_Rural": "Entorno de residencia",
    "Healthcare_Access": "Acceso a servicios sanitarios",
}

# Etiquetas legibles de las opciones categóricas
OPTION_LABELS: Dict[str, Dict[str, str]] = {
    "Gender": {"M": "Hombre", "F": "Mujer"},
    "Family_History": {"Yes": "Sí", "No": "No"},
    "Smoking_History": {"Yes": "Sí", "No": "No"},
    "Alcohol_Consumption": {"Yes": "Sí", "No": "No"},
    "Obesity_BMI": {
        "Normal": "Normal (IMC 18.5–24.9)",
        "Overweight": "Sobrepeso (IMC 25–29.9)",
        "Obese": "Obesidad (IMC ≥30)",
    },
    "Diet_Risk": {
        "Low": "Bajo (mediterránea, alta en fibra)",
        "Moderate": "Moderado (mixta estándar)",
        "High": "Alto (carne roja/procesada, baja fibra)",
    },
    "Physical_Activity": {
        "Low": "Baja (sedentario)",
        "Moderate": "Moderada (30–150 min/sem)",
        "High": "Alta (>150 min/sem)",
    },
    "Diabetes": {"Yes": "Sí", "No": "No"},
    "Inflammatory_Bowel_Disease": {"Yes": "Sí", "No": "No"},
    "Genetic_Mutation": {"Yes": "Sí", "No": "No"},
    "Urban_or_Rural": {"Urban": "Urbano", "Rural": "Rural"},
    "Healthcare_Access": {
        "Low": "Bajo",
        "Moderate": "Moderado",
        "High": "Alto",
    },
}

TARGET: str = "Cancer"

AGE_MIN: int = 18
AGE_MAX: int = 100


def validar_paciente(paciente: Dict) -> List[str]:
    errores: List[str] = []

    if "Age" not in paciente:
        errores.append("Falta el campo 'Age'.")
    else:
        try:
            edad = int(paciente["Age"])
            if not (AGE_MIN <= edad <= AGE_MAX):
                errores.append(
                    f"Edad fuera de rango ({AGE_MIN}-{AGE_MAX}): {edad}"
                )
        except (TypeError, ValueError):
            errores.append(f"Edad no es un entero válido: {paciente['Age']}")

    for feature, valores_validos in CATEGORICAL_FEATURES.items():
        if feature not in paciente:
            errores.append(f"Falta el campo '{feature}'.")
            continue
        if paciente[feature] not in valores_validos:
            errores.append(
                f"Valor no válido para '{feature}': {paciente[feature]!r}. "
                f"Esperado uno de: {valores_validos}"
            )

    return errores


def paciente_ejemplo_alto_riesgo() -> Dict:
    return {
        "Age": 68,
        "Gender": "M",
        "Family_History": "Yes",
        "Smoking_History": "Yes",
        "Alcohol_Consumption": "Yes",
        "Obesity_BMI": "Obese",
        "Diet_Risk": "High",
        "Physical_Activity": "Low",
        "Diabetes": "Yes",
        "Inflammatory_Bowel_Disease": "Yes",
        "Genetic_Mutation": "Yes",
        "Urban_or_Rural": "Urban",
        "Healthcare_Access": "Moderate",
    }


def paciente_ejemplo_bajo_riesgo() -> Dict:
    return {
        "Age": 35,
        "Gender": "F",
        "Family_History": "No",
        "Smoking_History": "No",
        "Alcohol_Consumption": "No",
        "Obesity_BMI": "Normal",
        "Diet_Risk": "Low",
        "Physical_Activity": "High",
        "Diabetes": "No",
        "Inflammatory_Bowel_Disease": "No",
        "Genetic_Mutation": "No",
        "Urban_or_Rural": "Urban",
        "Healthcare_Access": "High",
    }
