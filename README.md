# Simulador de Diagnóstico Médico de Cáncer de Colon

> Sistema híbrido de apoyo al diagnóstico que combina Machine Learning, evidencia científica publicada y análisis de imagen CT mediante red neuronal U-Net.

Autor: **Jiajiao Xu y Jordi Vidal**

---

## Descripción

Este proyecto implementa un simulador de apoyo al diagnóstico de cáncer colorrectal que integra **tres motores complementarios**:

1. **Modelo 1A — Machine Learning** (Random Forest calibrado sobre 13 variables clínicas)
2. **Modelo 1B — Calculadora de literatura** (estilo NCI CCRAT, basada en SEER + meta-análisis)
3. **Modelo 2 — Segmentación CT** (U-Net con ResNet-34 sobre el dataset MSD Task10 Colon)

La decisión final pondera ambos motores clínicos (70% literatura + 30% ML) y aplica criterios clínicos específicos basados en guías NCCN/ESMO.


---

## Características principales

- **Diseño dual-motor con prioridad a la literatura**: justificado mediante análisis estadístico en R que reveló inversiones de coeficientes en el modelo ML
- **Aplicación web profesional** desarrollada con Streamlit (paleta médica, sin emojis, Material Icons)
- **Análisis estadístico complementario en R** con regresión logística, Odds Ratios e intervalos de confianza
- **Detección automática de discrepancias** entre motores con alerta al usuario
- **Override clínico**: mutación genética o EII fuerzan el nivel "Alto" automáticamente

---

## Métricas de los modelos

| Modelo | Métrica | Valor |
|---|---|---|
| 1A — Random Forest calibrado | AUC | 0.716 |
| 1A — Random Forest calibrado | Brier score | 0.215 |
| 2 — U-Net + ResNet-34 (v4) | AUC | 0.796 |
| 2 — U-Net + ResNet-34 (v4) | Dice (positivos) | 0.470 |
| 2 — U-Net + ResNet-34 (v4) | Sensibilidad / Especificidad | 0.73 / 0.74 |

---

## Estructura del proyecto

```
colon_cancer_ai/
├── app/
│   ├── app_streamlit.py          # Aplicación principal
│   └── assets/styles.css         # Estilos CSS
├── src/
│   ├── python/
│   │   ├── utils/                # Módulos auxiliares
│   │   │   ├── paciente_schema.py
│   │   │   ├── risk_levels.py
│   │   │   ├── calculadora_riesgo_literatura.py
│   │   │   └── decision_combinada.py
│   │   ├── 01-08_*.py            # Pipeline ML
│   │   ├── 10_calibracion.py
│   │   ├── 11_risk_stratification.py
│   │   ├── 12_evaluar_calculadora_literatura.py
│   │   ├── 13_evaluar_modelo_ct.py
│   │   ├── entrenar_msd_colon_unet_v4.py
│   │   └── evaluar_msd_colon_unet_cached.py
│   └── r/
│       ├── 01_analisis_exploratorio.R
│       ├── 02_regresion_logistica.R
│       └── 03_visualizaciones_memoria.R
├── tests/
│   └── test_model1.py
├── reports/                       # Salidas R (PNG)
├── requirements.txt
└── README.md
```

> Las carpetas `data/`, `models/` y `_archivados_*/` están en `.gitignore` por su tamaño. Para regenerar:
> 1. Descarga el dataset Kaggle (ver sección Datos)
> 2. Ejecuta el pipeline (sección Entrenamiento)

---

## Instalación rápida

### Requisitos
- Python 3.11+
- R 4.5+ (opcional, para el análisis estadístico)
- (Opcional) GPU NVIDIA con CUDA 11.8+ para acelerar el modelo CT

### Pasos

```bash
# 1. Clonar
git clone https://github.com/TheSky189/colon_cancer_ai.git
cd colon_cancer_ai

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate      # macOS/Linux

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Opcional) Instalar PyTorch con CUDA si tienes GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Lanzar la aplicación
streamlit run app/app_streamlit.py
```

La app se abrirá en `http://localhost:8501`.

---

## Datos

Este proyecto utiliza dos datasets públicos:

1. **Dataset clínico**: [Colorectal Cancer Risk Factors](https://www.kaggle.com/) en Kaggle (167 497 casos)
   - Generación adicional de 167 497 controles sintéticos
2. **Dataset CT**: [Medical Segmentation Decathlon — Task10 Colon](http://medicaldecathlon.com/) (126 volúmenes CT)

Tras descargarlos, ubicarlos en `data/raw/` y ejecutar el pipeline de preprocesado:

```bash
python src/python/01_limpiar_dataset.py
python src/python/02_generar_controles.py
python src/python/03_construir_dataset.py
python src/python/preparar_msd_colon.py
python src/python/preprocesar_msd_colon_slices_256.py
```

---

## Entrenamiento

### Modelo 1A (clínico)
```bash
python src/python/04_entrenar_modelo.py
python src/python/10_calibracion.py
```

### Modelo 2 (CT)
```bash
python src/python/entrenar_msd_colon_unet_v4.py
```

### Análisis estadístico complementario (R)
```bash
Rscript src/r/01_analisis_exploratorio.R
Rscript src/r/02_regresion_logistica.R
Rscript src/r/03_visualizaciones_memoria.R
```

---

## Referencias científicas

- Freedman AN et al. (2009). *Colorectal cancer risk prediction tool*. **J Clin Oncol** 27(5):686-693.
- Butterworth AS et al. (2006). *Familial CCR meta-analysis*. **Eur J Cancer** 42(2):216-227.
- Ma Y et al. (2013). *Obesity and CCR*. **PLoS One** 8(1):e53916.
- Johnson CM et al. (2013). *Meta-analysis of CCR risk factors*. **Cancer Causes Control** 24(6):1207-1222.
- Ronneberger O, Fischer P, Brox T. (2015). *U-Net*. **MICCAI**.
- Antonelli M et al. (2022). *Medical Segmentation Decathlon*. **Nature Communications** 13:4128.
- NCI CCRAT: https://ccrisktool.cancer.gov/
- SEER: https://seer.cancer.gov/

---


