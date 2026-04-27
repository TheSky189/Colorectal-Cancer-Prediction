"""
Tests unitarios para los módulos del Modelo 1.

Ejecutar con:
    pytest src/python/tests/test_model1.py -v
"""

import sys
from pathlib import Path

# Permitir imports desde src/python/utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils.paciente_schema import (
    validar_paciente,
    paciente_ejemplo_alto_riesgo,
    paciente_ejemplo_bajo_riesgo,
    CATEGORICAL_FEATURES,
)
from utils.risk_levels import (
    nivel_riesgo,
    NIVEL_BAJO,
    NIVEL_MEDIO,
    NIVEL_ALTO,
    UMBRAL_BAJO_MEDIO,
    UMBRAL_MEDIO_ALTO,
)
from utils.calculadora_riesgo_literatura import (
    calcular_riesgo_literatura,
    calcular_riesgo_lifetime_aprox,
    _riesgo_basal,
)


# Tests de paciente_schema

class TestPacienteSchema:
    def test_paciente_ejemplo_alto_riesgo_es_valido(self):
        errores = validar_paciente(paciente_ejemplo_alto_riesgo())
        assert errores == [], f"No debería tener errores, pero tiene: {errores}"

    def test_paciente_ejemplo_bajo_riesgo_es_valido(self):
        errores = validar_paciente(paciente_ejemplo_bajo_riesgo())
        assert errores == [], f"No debería tener errores, pero tiene: {errores}"

    def test_falta_edad(self):
        paciente = paciente_ejemplo_alto_riesgo()
        del paciente["Age"]
        errores = validar_paciente(paciente)
        assert any("Age" in e for e in errores)

    def test_edad_fuera_de_rango(self):
        paciente = paciente_ejemplo_alto_riesgo()
        paciente["Age"] = 150
        errores = validar_paciente(paciente)
        assert any("rango" in e for e in errores)

    def test_categoria_invalida(self):
        # Bug clásico: usar "Male" en lugar de "M"
        paciente = paciente_ejemplo_alto_riesgo()
        paciente["Gender"] = "Male"
        errores = validar_paciente(paciente)
        assert any("Gender" in e for e in errores)

    def test_economic_classification_medium_invalido(self):
        # Bug clásico de 11_risk_stratification: "Medium" no existe.
        paciente = paciente_ejemplo_alto_riesgo()
        paciente["Economic_Classification"] = "Medium"
        errores = validar_paciente(paciente)
        assert any("Economic_Classification" in e for e in errores)


# Tests de risk_levels

class TestRiskLevels:
    def test_nivel_bajo(self):
        assert nivel_riesgo(0.10) == NIVEL_BAJO
        assert nivel_riesgo(0.0) == NIVEL_BAJO
        assert nivel_riesgo(UMBRAL_BAJO_MEDIO - 0.01) == NIVEL_BAJO

    def test_nivel_medio(self):
        assert nivel_riesgo(UMBRAL_BAJO_MEDIO) == NIVEL_MEDIO
        assert nivel_riesgo(0.50) == NIVEL_MEDIO
        assert nivel_riesgo(UMBRAL_MEDIO_ALTO - 0.01) == NIVEL_MEDIO

    def test_nivel_alto(self):
        assert nivel_riesgo(UMBRAL_MEDIO_ALTO) == NIVEL_ALTO
        assert nivel_riesgo(0.85) == NIVEL_ALTO
        assert nivel_riesgo(1.0) == NIVEL_ALTO

    def test_umbrales_son_consistentes(self):
        assert UMBRAL_BAJO_MEDIO < UMBRAL_MEDIO_ALTO
        assert 0 < UMBRAL_BAJO_MEDIO < 1
        assert 0 < UMBRAL_MEDIO_ALTO < 1


# Tests de calculadora basada en literatura

class TestCalculadoraLiteratura:
    def test_riesgo_basal_aumenta_con_edad(self):
        # El riesgo basal debe aumentar monótonamente con la edad
        # (al menos hasta los 80 años).
        edades = [25, 35, 45, 55, 65, 75]
        riesgos_h = [_riesgo_basal(e, "M") for e in edades]
        # Verificar monotonía estricta hasta los 75
        for i in range(len(riesgos_h) - 1):
            assert riesgos_h[i] < riesgos_h[i + 1], (
                f"Riesgo a los {edades[i]} ({riesgos_h[i]}) "
                f"debería ser menor que a los {edades[i+1]} ({riesgos_h[i+1]})"
            )

    def test_hombres_mayor_riesgo_basal_que_mujeres(self):
        # En la mayoría de tramos los hombres tienen mayor incidencia.
        for edad in [40, 50, 60, 70]:
            assert _riesgo_basal(edad, "M") > _riesgo_basal(edad, "F"), (
                f"A los {edad} años el riesgo basal de hombre "
                f"debería ser mayor que el de mujer."
            )

    def test_paciente_alto_riesgo_supera_paciente_bajo(self):
        alto = calcular_riesgo_literatura(paciente_ejemplo_alto_riesgo())
        bajo = calcular_riesgo_literatura(paciente_ejemplo_bajo_riesgo())
        assert alto["riesgo_5_anos"] > bajo["riesgo_5_anos"], (
            "El paciente de alto riesgo debería tener mayor riesgo estimado "
            "que el de bajo riesgo."
        )

    def test_paciente_bajo_riesgo_dentro_de_basal(self):
        # Una mujer joven sin factores de riesgo debe tener riesgo muy bajo.
        bajo = calcular_riesgo_literatura(paciente_ejemplo_bajo_riesgo())
        assert bajo["riesgo_5_anos"] < 0.01, (
            f"El riesgo a 5 años de una persona joven y sana debería "
            f"ser muy bajo, pero es {bajo['riesgo_5_anos']:.4f}"
        )

    def test_familia_history_aumenta_riesgo(self):
        base = paciente_ejemplo_bajo_riesgo()
        sin_fh = calcular_riesgo_literatura(base)["riesgo_5_anos"]
        base["Family_History"] = "Yes"
        con_fh = calcular_riesgo_literatura(base)["riesgo_5_anos"]
        assert con_fh > sin_fh

    def test_actividad_alta_reduce_riesgo(self):
        # Usamos un paciente intermedio (no de alto riesgo extremo) para
        # evitar que el cap del 50% enmascare el efecto del factor.
        base = paciente_ejemplo_bajo_riesgo()
        base["Age"] = 55  # subir un poco para tener riesgo medible
        base["Physical_Activity"] = "Low"
        sin_actividad = calcular_riesgo_literatura(base)["riesgo_5_anos"]
        base["Physical_Activity"] = "High"
        con_actividad = calcular_riesgo_literatura(base)["riesgo_5_anos"]
        assert con_actividad < sin_actividad

    def test_riesgo_capado_al_50_pct(self):
        # Aunque haya muchos factores adversos, el riesgo a 5 años no
        # debe superar el 50% (cap definido por seguridad).
        alto = calcular_riesgo_literatura(paciente_ejemplo_alto_riesgo())
        assert alto["riesgo_5_anos"] <= 0.50

    def test_lifetime_mayor_que_5_anos(self):
        for paciente in [paciente_ejemplo_alto_riesgo(),
                         paciente_ejemplo_bajo_riesgo()]:
            r5 = calcular_riesgo_literatura(paciente)["riesgo_5_anos"]
            r_lifetime = calcular_riesgo_lifetime_aprox(paciente)
            assert r_lifetime >= r5

    def test_explicacion_no_vacia(self):
        resultado = calcular_riesgo_literatura(paciente_ejemplo_alto_riesgo())
        assert isinstance(resultado["explicacion"], list)
        assert len(resultado["explicacion"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
