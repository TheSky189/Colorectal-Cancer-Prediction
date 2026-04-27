from typing import Dict, List

# TASAS BASALES DE INCIDENCIA A 5 AÑOS POR EDAD Y SEXO

RIESGO_BASAL_5_ANOS = {
    "M": {
        (18, 29): 0.00015, (30, 39): 0.00075, (40, 44): 0.00200,
        (45, 49): 0.00425, (50, 54): 0.00750, (55, 59): 0.01100,
        (60, 64): 0.01650, (65, 69): 0.02250, (70, 74): 0.02800,
        (75, 79): 0.03100, (80, 84): 0.03250, (85, 100): 0.03200,
    },
    "F": {
        (18, 29): 0.00012, (30, 39): 0.00065, (40, 44): 0.00170,
        (45, 49): 0.00350, (50, 54): 0.00580, (55, 59): 0.00850,
        (60, 64): 0.01250, (65, 69): 0.01700, (70, 74): 0.02100,
        (75, 79): 0.02350, (80, 84): 0.02500, (85, 100): 0.02500,
    },
}


# RIESGOS RELATIVOS (RR) POR FACTOR

RR_FAMILY_HISTORY = {"Yes": 1.79, "No": 1.00}
RR_INFLAMMATORY_BOWEL_DISEASE = {"Yes": 2.93, "No": 1.00}
RR_GENETIC_MUTATION = {"Yes": 4.00, "No": 1.00}
RR_SMOKING = {"Yes": 1.18, "No": 1.00}
RR_ALCOHOL = {"Yes": 1.21, "No": 1.00}
RR_OBESITY_BMI = {"Normal": 1.00, "Overweight": 1.10, "Obese": 1.33}
RR_DIET_RISK = {"Low": 1.00, "Moderate": 1.10, "High": 1.25}
RR_PHYSICAL_ACTIVITY = {"High": 0.76, "Moderate": 0.88, "Low": 1.00}
RR_DIABETES = {"Yes": 1.27, "No": 1.00}



# Nombres legibles para mostrar al usuario

FACTOR_LABELS_ES = {
    "Family_History": "Antecedentes familiares",
    "Inflammatory_Bowel_Disease": "Enfermedad inflamatoria intestinal",
    "Genetic_Mutation": "Mutacion genetica",
    "Smoking_History": "Tabaquismo",
    "Alcohol_Consumption": "Consumo de alcohol",
    "Obesity_BMI": "Obesidad",
    "Diet_Risk": "Dieta de alto riesgo",
    "Physical_Activity": "Actividad fisica",
    "Diabetes": "Diabetes tipo 2",
}


def _riesgo_basal(edad: int, sexo: str) -> float:
    if sexo not in RIESGO_BASAL_5_ANOS:
        raise ValueError(f"Sexo no valido: {sexo!r}. Esperado 'M' o 'F'.")
    tabla = RIESGO_BASAL_5_ANOS[sexo]
    for (edad_min, edad_max), riesgo in tabla.items():
        if edad_min <= edad <= edad_max:
            return riesgo
    raise ValueError(f"Edad fuera del rango soportado: {edad}")


def calcular_riesgo_literatura(paciente: Dict) -> Dict:
    edad = int(paciente["Age"])
    sexo = paciente["Gender"]

    riesgo_basal = _riesgo_basal(edad, sexo)

    factores: Dict[str, float] = {}
    factores["Family_History"] = RR_FAMILY_HISTORY[paciente["Family_History"]]
    factores["Inflammatory_Bowel_Disease"] = RR_INFLAMMATORY_BOWEL_DISEASE[
        paciente["Inflammatory_Bowel_Disease"]
    ]
    factores["Genetic_Mutation"] = RR_GENETIC_MUTATION[paciente["Genetic_Mutation"]]
    factores["Smoking_History"] = RR_SMOKING[paciente["Smoking_History"]]
    factores["Alcohol_Consumption"] = RR_ALCOHOL[paciente["Alcohol_Consumption"]]
    factores["Obesity_BMI"] = RR_OBESITY_BMI[paciente["Obesity_BMI"]]
    factores["Diet_Risk"] = RR_DIET_RISK[paciente["Diet_Risk"]]
    factores["Physical_Activity"] = RR_PHYSICAL_ACTIVITY[paciente["Physical_Activity"]]
    factores["Diabetes"] = RR_DIABETES[paciente["Diabetes"]]

    rr_total = 1.0
    for valor in factores.values():
        rr_total *= valor

    riesgo_5_anos = min(riesgo_basal * rr_total, 0.50)

    explicacion = _construir_explicacion(
        edad, sexo, riesgo_basal, factores, rr_total, riesgo_5_anos
    )

    return {
        "riesgo_basal": riesgo_basal,
        "rr_total": rr_total,
        "riesgo_5_anos": riesgo_5_anos,
        "factores_aplicados": factores,
        "explicacion": explicacion,
    }


def calcular_riesgo_lifetime_aprox(paciente: Dict) -> float:
    """Aproximacion del riesgo a 30 anos (cuasi-lifetime)."""
    riesgo_5 = calcular_riesgo_literatura(paciente)["riesgo_5_anos"]
    riesgo_30 = 1 - (1 - riesgo_5) ** 6
    return min(riesgo_30, 0.85)


def listar_factores_detectados(paciente: Dict) -> Dict[str, List[Dict]]:
    """
    Devuelve los factores detectados en el paciente clasificados en:
    - 'riesgo':     factores con RR > 1.0 (aumentan el riesgo)
    - 'protector':  factores con RR < 1.0 (reducen el riesgo)
    - 'neutral':    factores con RR = 1.0

    Cada elemento tiene: {'nombre': str, 'rr': float, 'clave': str}
    Ordenado por magnitud del efecto (RR mas distinto de 1 primero).
    """
    resultado = calcular_riesgo_literatura(paciente)
    factores = resultado["factores_aplicados"]

    riesgo = []
    protector = []
    neutral = []

    for clave, rr in factores.items():
        nombre = FACTOR_LABELS_ES.get(clave, clave)
        item = {"clave": clave, "nombre": nombre, "rr": rr}
        if rr > 1.0:
            riesgo.append(item)
        elif rr < 1.0:
            protector.append(item)
        else:
            neutral.append(item)

    # Ordenar por magnitud de efecto
    riesgo.sort(key=lambda x: x["rr"], reverse=True)
    protector.sort(key=lambda x: x["rr"])

    return {
        "riesgo": riesgo,
        "protector": protector,
        "neutral": neutral,
    }


def _construir_explicacion(edad, sexo, riesgo_basal, factores, rr_total, riesgo_final):
    sexo_legible = "hombre" if sexo == "M" else "mujer"
    mensajes = [
        f"Riesgo basal poblacional a 5 anos para un {sexo_legible} "
        f"de {edad} anos: {riesgo_basal*100:.2f}% (fuente: SEER).",
    ]

    factores_relevantes = [(n, rr) for n, rr in factores.items() if rr != 1.00]
    factores_relevantes.sort(key=lambda x: abs(x[1] - 1.0), reverse=True)

    if not factores_relevantes:
        mensajes.append("No se han aplicado factores de riesgo adicionales.")
    else:
        mensajes.append("Multiplicadores de riesgo aplicados:")
        for nombre, rr in factores_relevantes:
            efecto = "aumenta" if rr > 1.0 else "reduce"
            mensajes.append(
                f"  - {nombre}: RR = {rr:.2f} ({efecto} el riesgo)."
            )
        mensajes.append(f"Multiplicador combinado: x{rr_total:.2f}.")

    mensajes.append(f"Riesgo estimado a 5 anos: {riesgo_final*100:.2f}%.")
    return mensajes


def riesgo_a_5_anos_a_probabilidad_evento(riesgo_5_anos: float) -> float:
    return riesgo_5_anos


if __name__ == "__main__":
    paciente_alto = {
        "Age": 68, "Gender": "M", "Family_History": "Yes",
        "Smoking_History": "Yes", "Alcohol_Consumption": "Yes",
        "Obesity_BMI": "Obese", "Diet_Risk": "High",
        "Physical_Activity": "Low", "Diabetes": "Yes",
        "Inflammatory_Bowel_Disease": "Yes", "Genetic_Mutation": "No",
    }

    resultado = calcular_riesgo_literatura(paciente_alto)
    for linea in resultado["explicacion"]:
        print(linea)

    print("\n--- Factores detectados ---")
    factores = listar_factores_detectados(paciente_alto)
    print(f"De riesgo ({len(factores['riesgo'])}):")
    for f in factores["riesgo"]:
        print(f"  {f['nombre']} (RR={f['rr']:.2f})")
    print(f"Protectores ({len(factores['protector'])}):")
    for f in factores["protector"]:
        print(f"  {f['nombre']} (RR={f['rr']:.2f})")