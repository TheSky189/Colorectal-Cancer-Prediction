from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from scipy.ndimage import zoom

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_PYTHON = BASE_DIR / "src" / "python"
ASSETS_DIR = BASE_DIR / "app" / "assets"
REPORTS_DIR = BASE_DIR / "reports"
sys.path.insert(0, str(SRC_PYTHON))

from utils.paciente_schema import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    OPTION_LABELS,
    validar_paciente,
)
from utils.calculadora_riesgo_literatura import (
    calcular_riesgo_literatura,
    calcular_riesgo_lifetime_aprox,
)
from utils.decision_combinada import calcular_decision_combinada
from utils.model_downloader import ensure_models_present


CLINICAL_MODEL_PATH = BASE_DIR / "models" / "calibrated_model.joblib"
CT_MODEL_PATH = BASE_DIR / "models" / "msd_colon_unet_cached_best.pth"
CSS_PATH = ASSETS_DIR / "styles.css"

# Asegurar que los modelos están disponibles (descarga desde GitHub Releases
# la primera vez que se ejecuta la app si no están localmente)
ensure_models_present(BASE_DIR / "models", verbose=True)

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
DEFAULT_THRESHOLD_CLASS = 0.60



# Glosario de términos técnicos

GLOSARIO = {
    "NCI CCRAT": (
        "Colorectal Cancer Risk Assessment Tool: calculadora del Instituto "
        "Nacional del Cáncer de EE.UU. (NCI) que estima el riesgo de "
        "cáncer colorrectal a 5 y 10 años. Disponible en "
        "https://ccrisktool.cancer.gov/"
    ),
    "SEER": (
        "Surveillance, Epidemiology and End Results: programa del NCI que "
        "recopila estadísticas de cáncer de la población de EE.UU. desde "
        "1973. Es la fuente de referencia para tasas basales de incidencia."
    ),
    "AUC": (
        "Area Under the ROC Curve. Mide la capacidad del modelo para "
        "distinguir entre pacientes con y sin cáncer. Va de 0.5 (azar) a "
        "1.0 (perfecto). 0.7-0.8 es 'aceptable', 0.8-0.9 'excelente'."
    ),
    "Brier score": (
        "Mide la calidad de las probabilidades predichas. Va de 0 "
        "(perfecto) a 0.25 (azar). Valores bajos indican que cuando el "
        "modelo dice '70% de riesgo', realmente esa proporción de "
        "pacientes resulta tener cáncer."
    ),
    "Random Forest": (
        "Algoritmo de Machine Learning que combina muchos árboles de "
        "decisión (300 en este caso) para obtener una predicción más "
        "robusta que cualquier árbol individual."
    ),
    "Calibración isotónica": (
        "Técnica que ajusta las probabilidades del modelo para que sean "
        "realistas. Sin calibrar, los modelos suelen ser sobreconfiados."
    ),
    "Riesgo Relativo (RR)": (
        "Cuántas veces aumenta o disminuye el riesgo respecto a alguien "
        "sin ese factor. RR=2 significa que el riesgo se duplica. RR<1 "
        "indica factor protector."
    ),
    "Odds Ratio (OR)": (
        "Métrica similar al Riesgo Relativo, usada en regresión "
        "logística. Aproximadamente equivalente al RR cuando la "
        "enfermedad es poco frecuente (<10%)."
    ),
    "IC 95%": (
        "Intervalo de Confianza al 95%: rango en el que está el valor "
        "real con un 95% de probabilidad. Si el IC NO incluye el 1.0, "
        "el factor es estadísticamente significativo."
    ),
    "U-Net": (
        "Arquitectura de red neuronal diseñada específicamente para "
        "segmentar imágenes médicas. Tiene forma de 'U': una rama que "
        "comprime la imagen para extraer características y otra que la "
        "expande para localizar las regiones de interés píxel a píxel."
    ),
    "ResNet-34": (
        "Red neuronal preentrenada en millones de imágenes naturales "
        "(ImageNet). Aporta capacidad de reconocimiento visual general "
        "que acelera la detección de tumores."
    ),
    "Tversky loss": (
        "Función de pérdida que penaliza más los falsos negativos (no "
        "detectar un tumor real) que los falsos positivos. Crítico en "
        "oncología."
    ),
    "Dice score": (
        "Mide cuánto se solapan la predicción del modelo y la verdad "
        "anotada por radiólogos. Va de 0 (sin solape) a 1 (perfecto). "
        "Para segmentación tumoral, 0.5+ se considera bueno."
    ),
    "Índice de Youden": (
        "Criterio para elegir el umbral óptimo del modelo que maximiza "
        "simultáneamente la sensibilidad y la especificidad."
    ),
    "MSD Task10 Colon": (
        "Medical Segmentation Decathlon: conjunto de datos público con "
        "miles de imágenes CT/MRI anotadas por radiólogos. La Task10 "
        "contiene 126 volúmenes CT con tumores colorrectales anotados. "
        "http://medicaldecathlon.com/"
    ),
}

FUENTES_DETALLADAS = {
    "Freedman2009": {
        "titulo": "Freedman AN et al. (2009)",
        "revista": "Journal of Clinical Oncology",
        "tipo": "Artículo metodológico",
        "descripcion": (
            "Artículo original donde el NCI publicó el algoritmo del CCRAT "
            "(Colorectal Cancer Risk Assessment Tool). Define el cálculo "
            "de riesgo absoluto a 5 años combinando tasas basales y "
            "riesgos relativos."
        ),
    },
    "Butterworth2006": {
        "titulo": "Butterworth AS et al. (2006)",
        "revista": "European Journal of Cancer",
        "tipo": "Meta-análisis",
        "descripcion": (
            "Meta-análisis de 47 estudios sobre antecedentes familiares "
            "de cáncer colorrectal. Estima que tener un familiar de "
            "primer grado afectado multiplica el riesgo por 1.79."
        ),
    },
    "Ma2013": {
        "titulo": "Ma Y et al. (2013)",
        "revista": "PLoS One",
        "tipo": "Meta-análisis",
        "descripcion": (
            "Meta-análisis sobre obesidad y cáncer colorrectal con más "
            "de 1.5 millones de participantes. Estima RR de 1.10 para "
            "sobrepeso y 1.33 para obesidad."
        ),
    },
    "Johnson2013": {
        "titulo": "Johnson CM et al. (2013)",
        "revista": "Cancer Causes & Control",
        "tipo": "Revisión sistemática",
        "descripcion": (
            "Revisión sistemática de los principales factores de riesgo "
            "modificables: tabaco, alcohol, dieta, actividad física, "
            "diabetes. Fuente de la mayoría de los RR usados en este modelo."
        ),
    },
    "SEER": {
        "titulo": "SEER Cancer Statistics",
        "revista": "National Cancer Institute",
        "tipo": "Base de datos epidemiológica",
        "descripcion": (
            "Programa de vigilancia del cáncer que cubre ~28% de la "
            "población estadounidense desde 1973. Usado para las tasas "
            "basales de incidencia por edad y sexo."
        ),
    },
    "MSD": {
        "titulo": "Medical Segmentation Decathlon",
        "revista": "Antonelli M, Reinke A et al. (2022) Nature Comm.",
        "tipo": "Dataset público",
        "descripcion": (
            "Conjunto de datos público con 10 tareas de segmentación "
            "médica. La Task10 (Colon) contiene 126 volúmenes CT con "
            "tumores anotados por radiólogos. Es el benchmark estándar "
            "para evaluar modelos de segmentación de cáncer colorrectal."
        ),
    },
}



# UNet pequeña

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNetPequena(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        u3 = self.up3(b); d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out_conv(d1)




# CT utils

def normalizar_ct_2d(image):
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
    if p99 > p1:
        image = np.clip(image, p1, p99)
        image = (image - p1) / (p99 - p1)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


def resize_2d(arr, output_shape, order=1):
    zy = output_shape[0] / arr.shape[0]
    zx = output_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=order)


def preparar_imagen_ct_desde_upload(uploaded_file, target_size=(192, 192)):
    nombre = uploaded_file.name.lower()
    if nombre.endswith(".npz"):
        data = np.load(uploaded_file)
        if "image" in data:
            image = data["image"].astype(np.float32)
        elif "img" in data:
            arr = data["img"]
            image = arr[:, :, arr.shape[2] // 2].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
        else:
            raise ValueError("El archivo .npz no contiene 'image' ni 'img'.")
    elif nombre.endswith(".npy"):
        image = np.load(uploaded_file).astype(np.float32)
        if image.ndim == 3: image = image[:, :, image.shape[2] // 2]
    elif nombre.endswith((".png", ".jpg", ".jpeg")):
        image = np.array(Image.open(uploaded_file).convert("L")).astype(np.float32)
    else:
        raise ValueError("Formato no soportado.")
    image = normalizar_ct_2d(image)
    image = resize_2d(image, target_size, 1).astype(np.float32)
    tensor = torch.tensor(np.stack([image]*3, axis=0), dtype=torch.float32).unsqueeze(0)
    return image, tensor


def calcular_mascara_adaptativa(probs_tensor, prob_global, umbral_clas):
    probs_np = probs_tensor.cpu().numpy()[0, 0]
    if prob_global < 0.70 * umbral_clas:
        return np.zeros_like(probs_np), None
    umbral = max(float(np.percentile(probs_np, 99.5)), 0.30)
    return (probs_np > umbral).astype(np.float32), umbral


def render_overlay_mascara(image_np, mascara_np):
    img_rgb = np.stack([image_np]*3, axis=-1)
    img_rgb = np.clip(img_rgb, 0, 1)
    alpha = 0.45
    overlay = img_rgb.copy()
    overlay[mascara_np > 0.5, 0] = 1.0
    overlay[mascara_np > 0.5, 1] = 0.0
    overlay[mascara_np > 0.5, 2] = 0.0
    return np.clip((1 - alpha) * img_rgb + alpha * overlay, 0, 1)



# Carga de modelos

@st.cache_resource
def cargar_modelo_clinico():
    return joblib.load(CLINICAL_MODEL_PATH) if CLINICAL_MODEL_PATH.exists() else None


@st.cache_resource
def cargar_modelo_ct():
    if not CT_MODEL_PATH.exists(): return None, None
    # PyTorch 2.6+ exige weights_only=False explicitamente
    checkpoint = torch.load(CT_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_keys = list(checkpoint["model_state_dict"].keys())
    # Detectar arquitectura por las claves del state_dict
    es_efficientnet = any("_blocks" in k or "_conv_stem" in k for k in state_keys)
    es_resnet       = any("encoder.layer" in k or "encoder.conv1" in k for k in state_keys)
    es_smp          = es_efficientnet or es_resnet
    if es_smp:
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            return None, None
        if es_efficientnet:
            encoder_name, arq, default_size = "efficientnet-b0", "smp_efficientnet_b0", 320
        else:
            encoder_name, arq, default_size = "resnet34", "smp_resnet34", 256
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=None,
                        in_channels=3, classes=1).to(DEVICE)
        cfg_saved = checkpoint.get("config", {})
        size = cfg_saved.get("image_size", default_size)
    else:
        model = UNetPequena().to(DEVICE)
        arq, size = "unet_small", 192
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    umbral_info = checkpoint.get("umbral_optimo_youden")
    metrics = checkpoint.get("metrics", {})
    if umbral_info and "umbral" in umbral_info:
        threshold = float(umbral_info["umbral"])
    elif "threshold_opt" in metrics:
        threshold = float(metrics["threshold_opt"])
    else:
        threshold = DEFAULT_THRESHOLD_CLASS
    # Si el threshold es absurdo (Youden=0), usar 0.5 sano por defecto
    if threshold < 0.05:
        threshold = 0.5
    return model, {"arquitectura": arq, "image_size": (size, size), "threshold_class": threshold}



# Helpers UI

def icono(nombre: str, extra_class: str = "") -> str:
    """Devuelve un Material Icon como span HTML."""
    return f'<span class="material-icons {extra_class}">{nombre}</span>'


def badge_nivel(nivel: str) -> str:
    """Badge de color para el nivel de riesgo (sin emoji)."""
    cls = {"Alto": "badge-high", "Medio": "badge-medium", "Bajo": "badge-low"}.get(nivel, "")
    return f'<span class="risk-badge {cls}">{nivel.upper()}</span>'


def clase_nivel_css(nivel: str) -> str:
    return {"Alto": "level-high", "Medio": "level-medium", "Bajo": "level-low"}.get(nivel, "")


def render_glosario_inline(claves_usadas):
    """Muestra un glosario de los términos usados en la sección actual."""
    if not claves_usadas:
        return
    st.markdown("##### Glosario de términos")
    for clave in claves_usadas:
        if clave in GLOSARIO:
            st.markdown(
                f'<div class="glosario-item">'
                f'<span class="glosario-termino">{clave}</span>'
                f'<span class="glosario-def">{GLOSARIO[clave]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def cargar_css():
    if CSS_PATH.exists():
        with open(CSS_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Cargar Material Icons y Roboto desde Google Fonts
    st.markdown("""
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div class="app-header">
        <div class="header-icon-large">{icono('medical_services')}</div>
        <h1>Simulador de Diagnóstico de Cáncer de Colon</h1>
        <div class="subtitle">
            Sistema de apoyo clínico que combina evidencia científica publicada,
            Machine Learning y análisis de imagen CT mediante red neuronal U-Net.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="app-footer">
        Simulador desarrollado como proyecto.
        No sustituye al juicio clínico profesional.
    </div>
    """, unsafe_allow_html=True)


def construir_formulario_paciente(prefix: str = "") -> dict:
    col1, col2, col3 = st.columns(3)
    def _label(f): return FEATURE_LABELS.get(f, f)
    def _opts(f): return CATEGORICAL_FEATURES[f]
    def _fmt(f):
        labels = OPTION_LABELS.get(f, {})
        return lambda x: labels.get(x, x)

    with col1:
        edad = st.slider(_label("Age"), 18, 90, 50, key=f"{prefix}edad",
                          help=FEATURE_DESCRIPTIONS["Age"])
        genero = st.selectbox(_label("Gender"), _opts("Gender"), format_func=_fmt("Gender"),
                              key=f"{prefix}gen", help=FEATURE_DESCRIPTIONS["Gender"])
        antecedentes = st.selectbox(_label("Family_History"), _opts("Family_History"),
                                     format_func=_fmt("Family_History"), key=f"{prefix}fh",
                                     help=FEATURE_DESCRIPTIONS["Family_History"])
        tabaco = st.selectbox(_label("Smoking_History"), _opts("Smoking_History"),
                              format_func=_fmt("Smoking_History"), key=f"{prefix}sm",
                              help=FEATURE_DESCRIPTIONS["Smoking_History"])
        alcohol = st.selectbox(_label("Alcohol_Consumption"), _opts("Alcohol_Consumption"),
                               format_func=_fmt("Alcohol_Consumption"), key=f"{prefix}al",
                               help=FEATURE_DESCRIPTIONS["Alcohol_Consumption"])
    with col2:
        imc = st.selectbox(_label("Obesity_BMI"), _opts("Obesity_BMI"),
                           format_func=_fmt("Obesity_BMI"), key=f"{prefix}bmi",
                           help=FEATURE_DESCRIPTIONS["Obesity_BMI"])
        dieta = st.selectbox(_label("Diet_Risk"), _opts("Diet_Risk"),
                             format_func=_fmt("Diet_Risk"), key=f"{prefix}diet",
                             help=FEATURE_DESCRIPTIONS["Diet_Risk"])
        actividad = st.selectbox(_label("Physical_Activity"), _opts("Physical_Activity"),
                                  format_func=_fmt("Physical_Activity"), key=f"{prefix}act",
                                  help=FEATURE_DESCRIPTIONS["Physical_Activity"])
        diabetes = st.selectbox(_label("Diabetes"), _opts("Diabetes"),
                                format_func=_fmt("Diabetes"), key=f"{prefix}db",
                                help=FEATURE_DESCRIPTIONS["Diabetes"])
        eii = st.selectbox(_label("Inflammatory_Bowel_Disease"),
                            _opts("Inflammatory_Bowel_Disease"),
                            format_func=_fmt("Inflammatory_Bowel_Disease"),
                            key=f"{prefix}eii",
                            help=FEATURE_DESCRIPTIONS["Inflammatory_Bowel_Disease"])
    with col3:
        mutacion = st.selectbox(_label("Genetic_Mutation"), _opts("Genetic_Mutation"),
                                format_func=_fmt("Genetic_Mutation"), key=f"{prefix}mut",
                                help=FEATURE_DESCRIPTIONS["Genetic_Mutation"])
        zona = st.selectbox(_label("Urban_or_Rural"), _opts("Urban_or_Rural"),
                            format_func=_fmt("Urban_or_Rural"), key=f"{prefix}zon",
                            help=FEATURE_DESCRIPTIONS["Urban_or_Rural"])
        acceso = st.selectbox(_label("Healthcare_Access"), _opts("Healthcare_Access"),
                              format_func=_fmt("Healthcare_Access"), key=f"{prefix}hac",
                              help=FEATURE_DESCRIPTIONS["Healthcare_Access"])

    return {
        "Age": edad, "Gender": genero, "Family_History": antecedentes,
        "Smoking_History": tabaco, "Alcohol_Consumption": alcohol,
        "Obesity_BMI": imc, "Diet_Risk": dieta, "Physical_Activity": actividad,
        "Diabetes": diabetes, "Inflammatory_Bowel_Disease": eii,
        "Genetic_Mutation": mutacion, "Urban_or_Rural": zona,
        "Healthcare_Access": acceso,
    }


def predecir_ml(modelo_clinico, paciente: dict) -> float:
    columnas = NUMERIC_FEATURES + CATEGORICAL_FEATURE_NAMES
    paciente_df = pd.DataFrame([{c: paciente[c] for c in columnas}])
    return float(modelo_clinico.predict_proba(paciente_df)[0][1])


def render_motor_card(icono_nombre, titulo, valor_mostrar, nivel, descripcion_corta):
    cls = clase_nivel_css(nivel)
    st.markdown(f"""
    <div class="motor-card">
        <div class="motor-card-title">
            {icono(icono_nombre, 'motor-icon')}
            <span>{titulo}</span>
        </div>
        <div class="motor-card-value">{valor_mostrar}</div>
        <div class="motor-card-level {cls}">{badge_nivel(nivel)}</div>
        <div class="motor-card-desc">{descripcion_corta}</div>
    </div>
    """, unsafe_allow_html=True)


def render_gauge(score: float, nivel: str):
    pct = int(score * 100)
    fill = {"Alto": "gauge-fill-high", "Medio": "gauge-fill-medium",
            "Bajo": "gauge-fill-low"}[nivel]
    st.markdown(f"""
    <div class="gauge-container">
        <div class="gauge-labels">
            <span>Bajo (0%)</span>
            <span>Medio (~40%)</span>
            <span>Alto (&gt;55%)</span>
        </div>
        <div class="gauge-bar">
            <div class="gauge-fill {fill}" style="width: {pct}%;"></div>
        </div>
        <div class="gauge-score">Score combinado: <b>{pct}%</b></div>
    </div>
    """, unsafe_allow_html=True)


def render_interpretacion_niveles(rangos):
    html = '<div class="niveles-tabla">'
    for rango, badge_cls, nivel in rangos:
        html += (f'<div class="nivel-row">'
                 f'<span class="nivel-rango">{rango}</span>'
                 f'<span class="nivel-arrow">→</span>'
                 f'<span class="risk-badge {badge_cls}">{nivel}</span>'
                 f'</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_chips_factores(riesgo, protectores):
    if riesgo:
        chips = " ".join([
            f'<span class="factor-chip chip-risk">{f["nombre"]} (RR x{f["rr"]:.2f})</span>'
            for f in riesgo
        ])
        st.markdown(
            f'<div class="chips-title">{icono("warning", "chip-icon")} '
            f'Factores de riesgo detectados</div>',
            unsafe_allow_html=True,
        )
        st.markdown(chips, unsafe_allow_html=True)
    if protectores:
        chips = " ".join([
            f'<span class="factor-chip chip-protect">{f["nombre"]} (RR x{f["rr"]:.2f})</span>'
            for f in protectores
        ])
        st.markdown(
            f'<div class="chips-title">{icono("shield", "chip-icon")} '
            f'Factores protectores</div>',
            unsafe_allow_html=True,
        )
        st.markdown(chips, unsafe_allow_html=True)


def render_fuente_card(clave: str):
    f = FUENTES_DETALLADAS[clave]
    st.markdown(f"""
    <div class="fuente-card">
        <div class="fuente-titulo">{f['titulo']}</div>
        <div class="fuente-meta">
            <span class="fuente-tipo">{f['tipo']}</span>
            <span class="fuente-separator">•</span>
            <span class="fuente-revista">{f['revista']}</span>
        </div>
        <div class="fuente-desc">{f['descripcion']}</div>
    </div>
    """, unsafe_allow_html=True)


def buscar_imagenes_reportes():
    resultado = {"eda": [], "regresion": [], "memoria": []}
    if not REPORTS_DIR.exists():
        return resultado
    for subdir, key in [("eda", "eda"),
                        ("regresion_logistica", "regresion"),
                        ("memoria", "memoria")]:
        carpeta = REPORTS_DIR / subdir
        if carpeta.exists():
            resultado[key] = sorted(carpeta.glob("*.png"))
    return resultado


def render_imagen_reporte(img_path, descripcion=""):
    """Muestra una imagen del reporte directamente (sin expander), tamaño medio."""
    st.markdown(f"**{img_path.stem.replace('_', ' ').title()}**")
    # Usar width fijo en lugar de use_container_width=True
    st.image(str(img_path), width=700)
    if descripcion:
        st.caption(descripcion)
    st.markdown("<hr class='report-separator'/>", unsafe_allow_html=True)



# Streamlit

st.set_page_config(
    page_title="Simulador CCR",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)

cargar_css()
render_header()

modelo_clinico = cargar_modelo_clinico()
modelo_ct, config_ct = cargar_modelo_ct()
imagenes_reportes = buscar_imagenes_reportes()

tab1, tab2, tab3, tab4 = st.tabs([
    "Riesgo clínico",
    "Imagen CT",
    "Análisis estadístico",
    "Información",
])


# TAB 1 — Riesgo clínico

with tab1:
    st.markdown("### Evaluación clínica del riesgo")
    st.caption(
        "Introduzca los datos del paciente. El sistema aplica DOS motores "
        "independientes y combina sus resultados con criterios clínicos "
        "para emitir una decisión robusta."
    )

    paciente = construir_formulario_paciente(prefix="t1_")

    if st.button("Evaluar riesgo clínico", type="primary"):
        errores = validar_paciente(paciente)
        if errores:
            st.error("Errores de validación:")
            for e in errores: st.write(f"- {e}")
            st.stop()

        if modelo_clinico is None:
            st.error("No se encontró el modelo calibrado.")
            st.stop()

        prob_ml = predecir_ml(modelo_clinico, paciente)
        resultado_lit = calcular_riesgo_literatura(paciente)
        riesgo_lifetime = calcular_riesgo_lifetime_aprox(paciente)
        decision = calcular_decision_combinada(paciente, prob_ml, resultado_lit)

        # Análisis por motor 
        st.markdown("## Análisis por motor")
        st.caption(
            "Cada motor analiza al paciente de forma independiente. "
            "Despliegue los detalles para entender cómo se calcula cada valor."
        )

        col_ml, col_lit = st.columns(2)

        with col_ml:
            render_motor_card(
                icono_nombre="memory",
                titulo="Modelo 1A · Machine Learning",
                valor_mostrar=f"{prob_ml*100:.2f}%",
                nivel=decision.nivel_ml,
                descripcion_corta=(
                    "Probabilidad estadística derivada de un modelo "
                    "de Machine Learning entrenado con datos clínicos."
                ),
            )
            with st.expander("¿Cómo se calcula este valor?"):
                st.markdown(f"""
**Algoritmo**: Random Forest de 300 árboles, calibrado mediante regresión isotónica con validación cruzada (CV=3).

**Entrada**: las 13 variables del paciente (edad + 12 categóricas).

**Resultado bruto**: `{prob_ml:.4f}` ({prob_ml*100:.2f}%)
""")
                st.markdown("**Interpretación del valor:**")
                render_interpretacion_niveles([
                    ("< 30%", "badge-low", "Bajo"),
                    ("30% - 70%", "badge-medium", "Medio"),
                    ("> 70%", "badge-high", "Alto"),
                ])

                st.markdown("""
**Limitación importante**: este modelo se entrenó parcialmente sobre
controles sintéticos. El análisis estadístico (pestaña "Análisis estadístico")
reveló que algunos factores aparecen invertidos (ej. sedentarismo como
protector). Por eso este motor tiene **peso secundario (30%)** en la
decisión final, dejando la prioridad a la literatura clínica.

**Métricas en validación**: AUC = 0.716 · Brier score = 0.215
""")
                render_glosario_inline([
                    "Random Forest", "Calibración isotónica",
                    "AUC", "Brier score"
                ])

        with col_lit:
            render_motor_card(
                icono_nombre="menu_book",
                titulo="Modelo 1B · Calculadora de literatura",
                valor_mostrar=f"{resultado_lit['riesgo_5_anos']*100:.2f}%",
                nivel=decision.nivel_literatura,
                descripcion_corta=(
                    f"Riesgo absoluto a 5 años. A largo plazo (~30 años): "
                    f"{riesgo_lifetime*100:.1f}%."
                ),
            )
            with st.expander("¿Cómo se calcula este valor?"):
                st.markdown("""
**Metodología**: estilo NCI CCRAT.

**Fórmula**:
""")
                st.code("Riesgo = Tasa_basal_SEER × ∏(RR_i)", language="text")
                st.markdown("""
donde **RR** es el Riesgo Relativo de cada factor según meta-análisis
publicados en revistas médicas.

**Aplicado a este paciente:**
""")
                for linea in resultado_lit["explicacion"]:
                    st.write(f"- {linea}")

                st.markdown("**Interpretación según guías clínicas:**")
                render_interpretacion_niveles([
                    ("< 1.5% a 5 años", "badge-low", "Bajo"),
                    ("1.5% - 5%", "badge-medium", "Medio"),
                    ("> 5%", "badge-high", "Alto"),
                ])

                st.markdown("""
**Ventaja**: basado en evidencia clínica validada sobre cohortes reales
de cientos de miles de pacientes. Por eso es el **motor primario (70%)**
en la decisión final.

**Fuentes principales** (ver pestaña "Información" para detalles):
SEER, Freedman 2009, Butterworth 2006, Ma 2013, Johnson 2013.
""")
                render_glosario_inline([
                    "NCI CCRAT", "SEER", "Riesgo Relativo (RR)"
                ])

        st.divider()

        # Decisión combinada 
        st.markdown("## Decisión combinada del sistema")

        cls = clase_nivel_css(decision.nivel_final)
        st.markdown(f"""
        <div class="decision-card">
            <div class="decision-title">RESULTADO FINAL</div>
            <div class="decision-level {cls}">
                Riesgo {decision.nivel_final}
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_gauge(decision.score_combinado, decision.nivel_final)

        st.markdown("#### Motivo de la decisión")
        criterios_aplicados = (
            "aplicados (criterios clínicos específicos para este paciente)"
            if decision.override_aplicado
            else "no aplicados (no fue necesario reforzar el resultado)"
        )
        st.markdown(f"""
1. **Motor de literatura (peso 70%)** indica nivel **{decision.nivel_literatura}**
   ({decision.riesgo_literatura_5a*100:.2f}% de riesgo a 5 años).
2. **Motor ML (peso 30%)** indica nivel **{decision.nivel_ml}**
   ({decision.prob_ml*100:.2f}%).
3. **Score ponderado** = `0.70 × literatura_normalizada + 0.30 × ML` =
   **{decision.score_combinado:.3f}** ({int(decision.score_combinado*100)}%).
4. **Concordancia entre motores**: {decision.concordancia_descripcion}.
5. **Criterios clínicos adicionales**: {criterios_aplicados}.
""")

        if decision.override_aplicado:
            st.info(f"**Criterio clínico aplicado:** {decision.razon_override}")

        if decision.discrepancia:
            st.warning(
                f"**{decision.concordancia_descripcion}**. "
                "El sistema priorizó la evidencia clínica publicada."
            )

        st.markdown("#### Recomendación clínica")
        if decision.nivel_final == "Alto":
            st.error(decision.recomendacion)
        elif decision.nivel_final == "Medio":
            st.warning(decision.recomendacion)
        else:
            st.success(decision.recomendacion)

        st.divider()

        # Factores 
        st.markdown("## Factores identificados en este paciente")
        render_chips_factores(decision.factores_riesgo, decision.factores_protectores)

        if decision.nivel_final == "Alto":
            st.info(
                "**Siguiente paso recomendado:** pase a la pestaña "
                "**Imagen CT** para complementar con análisis radiológico."
            )


# TAB 2 — Imagen CT

with tab2:
    st.markdown("### Análisis de imagen CT")
    st.caption(
        "Suba una slice CT en formato .npz, .npy, .png, .jpg o .jpeg. "
        "Para resultados consistentes, use imágenes del mismo dominio que "
        "el conjunto de entrenamiento (MSD Task10 Colon)."
    )

    with st.expander("¿Cómo detecta el modelo el tumor? (explicación para no expertos)"):
        st.markdown("""
El sistema usa una red neuronal llamada **U-Net** entrenada sobre
cientos de imágenes CT donde radiólogos especialistas marcaron
**píxel a píxel** dónde estaban los tumores.

**Cómo funciona en 3 pasos:**

1. **Compresión**: la imagen se procesa por capas que extraen
   características visuales (bordes, texturas, formas, contrastes).
   Cada capa "ve" la imagen con menos detalle pero más contexto global.

2. **Decisión por píxel**: el modelo asigna a cada píxel una
   probabilidad entre 0 y 1 de pertenecer a tejido tumoral. Lo aprende
   comparando lo que ve con miles de ejemplos anotados durante el
   entrenamiento.

3. **Localización (máscara roja)**: los píxeles con probabilidad más
   alta forman la zona resaltada en rojo. **No es que el modelo
   "sepa" de anatomía**, sino que reconoce patrones visuales similares
   a tumores que vio durante el entrenamiento.

**¿Por qué el rojo aparece donde aparece?**
La red identifica zonas con características típicas de tumores
colorrectales: tejido más denso, contornos irregulares, contraste
diferente al intestino normal. La zona roja **es la predicción del
modelo, no una verdad absoluta**: requiere confirmación por radiólogo.

**Limitaciones honestas:**
- El modelo se entrenó solo con imágenes del dataset MSD
  (126 pacientes). Imágenes de otros equipos o protocolos pueden dar
  resultados poco fiables.
- Solo analiza **una slice** (corte 2D) por vez. Un radiólogo evalúa
  el volumen completo en 3D.
- Puede confundir tumores con otras estructuras densas en algunos
  casos. Es una herramienta de apoyo, no un diagnóstico definitivo.
""")
        render_glosario_inline(["U-Net", "ResNet-34", "MSD Task10 Colon"])

    if modelo_ct is None:
        st.error("No se encontró el modelo CT entrenado.")
    else:
        with st.expander("Configuración del modelo cargado"):
            st.markdown(f"""
- **Arquitectura**: `{config_ct['arquitectura']}` (ResNet-34 preentrenado en ImageNet, decodificador de U-Net).
- **Tamaño de entrada**: {config_ct['image_size']}.
- **Función de pérdida**: combinada 0.5·BCE + 0.5·Tversky (α=0.3, β=0.7).
- **Umbral de clasificación**: {config_ct['threshold_class']:.4f} (calculado con el índice de Youden).
- **Umbral de máscara**: adaptativo por imagen (percentil 99.5 del mapa de probabilidades).

**Métricas en validación**:
AUC = 0.796 · Dice (positivos) = 0.470 ·
Sensibilidad = 0.732 · Especificidad = 0.742
""")
            render_glosario_inline(["Tversky loss", "Dice score", "Índice de Youden", "AUC"])

        uploaded_file = st.file_uploader(
            "Subir slice CT", type=["npz", "npy", "png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            try:
                image_np, input_tensor = preparar_imagen_ct_desde_upload(
                    uploaded_file, target_size=config_ct["image_size"]
                )
                input_tensor = input_tensor.to(DEVICE)

                with torch.no_grad():
                    logits = modelo_ct(input_tensor)
                    probs = torch.sigmoid(logits)

                probabilidad = float(probs.max().cpu().item())
                mascara_np, umbral_usado = calcular_mascara_adaptativa(
                    probs, probabilidad, config_ct["threshold_class"]
                )

                col_img, col_mask = st.columns(2)
                with col_img:
                    st.markdown("**Imagen CT procesada**")
                    st.image(image_np, clamp=True, use_container_width=True)
                with col_mask:
                    st.markdown("**Región sospechosa identificada**")
                    if mascara_np.sum() == 0:
                        st.image(image_np, clamp=True, use_container_width=True)
                        st.caption("Slice clasificado como NO tumoral.")
                    else:
                        overlay = render_overlay_mascara(image_np, mascara_np)
                        st.image(overlay, clamp=True, use_container_width=True)
                        st.caption(
                            f"Zona roja = píxeles con mayor probabilidad de "
                            f"ser tumor según el modelo (umbral dinámico: {umbral_usado:.3f}). "
                            "Requiere confirmación por radiólogo."
                        )

                st.metric("Probabilidad estimada de tumor", f"{probabilidad*100:.2f}%")

                umbral_clas = config_ct["threshold_class"]
                if probabilidad >= umbral_clas:
                    st.error(
                        f"**Sospecha tumoral** (probabilidad ≥ {umbral_clas:.3f}). "
                        "Recomendar revisión por radiólogo y posible colonoscopia."
                    )
                elif probabilidad >= umbral_clas * 0.7:
                    st.warning("**Sospecha intermedia.** Revisar slice.")
                else:
                    st.info("**Baja sospecha** tumoral en esta slice.")

            except Exception as e:
                st.error(f"No se pudo procesar el archivo: {e}")


# TAB 3 — Análisis estadístico

with tab3:
    st.markdown("### Análisis estadístico avanzado")
    st.caption(
        "Esta sección presenta el análisis estadístico inferencial del "
        "Modelo 1, realizado en R con las librerías tidyverse, ggplot2 y pROC. "
        "R complementa a Python aportando interpretabilidad estadística clásica "
        "(OR con IC 95%, tests de hipótesis) y gráficos de calidad publicación."
    )

    if not any(imagenes_reportes.values()):
        st.warning(
            "No se encontraron gráficos generados por R en `reports/`. "
            "Para generarlos, ejecute desde PowerShell:\n\n"
            "```powershell\n"
            "Rscript src/r/01_analisis_exploratorio.R\n"
            "Rscript src/r/02_regresion_logistica.R\n"
            "Rscript src/r/03_visualizaciones_memoria.R\n"
            "```"
        )
    else:
        # EDA 
        if imagenes_reportes["eda"]:
            st.markdown("#### Análisis exploratorio de datos (EDA)")
            st.caption(
                "Visualización de distribuciones, frecuencias y correlaciones "
                "del dataset clínico."
            )

            descripciones_eda = {
                "distribucion_edad": (
                    "Densidad de edad de casos (con cáncer) vs controles. "
                    "Verifica si hay sesgo de edad entre cohortes."
                ),
                "boxplot_edad_cancer": (
                    "Comparación estadística de la distribución de edad "
                    "entre ambos grupos."
                ),
                "factores_riesgo_frecuencias": (
                    "Prevalencia de cada factor de riesgo (% de 'Sí') en "
                    "cada cohorte."
                ),
                "correlograma": (
                    "Matriz de correlaciones entre todas las variables. "
                    "Útil para detectar redundancias."
                ),
            }
            for img_path in imagenes_reportes["eda"]:
                render_imagen_reporte(img_path, descripciones_eda.get(img_path.stem, ""))

        # Regresión logística 
        if imagenes_reportes["regresion"]:
            st.markdown("#### Regresión logística con Odds Ratios")
            st.caption(
                "Análisis estadístico de los factores de riesgo con Odds Ratios "
                "e Intervalos de Confianza al 95%."
            )

            descripciones_reg = {
                "forest_plot_OR": (
                    "Forest plot de los Odds Ratios. Cada punto es un "
                    "factor; las barras horizontales son su intervalo "
                    "de confianza del 95%. Si la barra NO toca la línea "
                    "vertical en 1.0, el factor es estadísticamente "
                    "significativo."
                ),
                "comparacion_OR_vs_RR": (
                    "Comparación clave: OR del modelo (azul) vs RR de "
                    "literatura (rojo). Sirve para validar si los datos "
                    "reflejan la realidad clínica."
                ),
            }
            for img_path in imagenes_reportes["regresion"]:
                render_imagen_reporte(img_path, descripciones_reg.get(img_path.stem, ""))

            render_glosario_inline(["Odds Ratio (OR)", "IC 95%",
                                    "Riesgo Relativo (RR)"])

        # Validación 
        if imagenes_reportes["memoria"]:
            st.markdown("#### Validación del modelo")
            st.caption(
                "Gráficos de evaluación del rendimiento del Modelo 1 con "
                "intervalos de confianza."
            )

            descripciones_mem = {
                "roc_con_bootstrap": (
                    "Curva ROC con intervalos de confianza vía bootstrap. "
                    "Mide la capacidad global del modelo."
                ),
                "calibration_plot": (
                    "Gráfico de calibración. La línea diagonal = calibración "
                    "perfecta. Más cerca = probabilidades más realistas."
                ),
                "forest_plot_rr_literatura": (
                    "Forest plot de los Riesgos Relativos del Modelo 1B."
                ),
                "distribucion_predicciones": (
                    "Densidad de probabilidades predichas por cohorte. "
                    "Zonas sombreadas = umbrales Bajo/Medio/Alto."
                ),
                "curvas_metricas": (
                    "Sensibilidad, especificidad, precisión y F1 según "
                    "umbral. Para elegir el punto de corte óptimo."
                ),
            }
            for img_path in imagenes_reportes["memoria"]:
                render_imagen_reporte(img_path, descripciones_mem.get(img_path.stem, ""))


# TAB 4 — Información

with tab4:
    st.markdown("### Acerca del simulador")

    arch_info = "modelo no cargado"
    if config_ct is not None:
        arch_info = f"{config_ct['arquitectura']} ({config_ct['image_size'][0]}x{config_ct['image_size'][1]})"

    st.markdown(f"""
#### Arquitectura del sistema

El simulador integra **tres motores** complementarios:

1. **Modelo 1A — ML clínico** (Random Forest calibrado).
   Aporta estratificación basada en interacciones no lineales entre factores.

2. **Modelo 1B — Calculadora de literatura**.
   Cálculo determinista basado en SEER + meta-análisis publicados,
   metodología inspirada en NCI CCRAT.
   Es el **motor primario** de la decisión combinada.

3. **Modelo 2 — Segmentación CT** (U-Net con ResNet-34).
   Modelo cargado actualmente: `{arch_info}`.

---

#### Decisión combinada

La decisión final aplica:
- **Ponderación 70/30**: literatura (mayor peso) + ML.
- **Criterios clínicos específicos**: presencia de mutación genética
  conocida o enfermedad inflamatoria intestinal eleva el nivel a Alto
  (basado en guías NCCN/ESMO).
- **Detección de discrepancias** entre ambos motores con alerta al usuario.

---

#### Limitaciones importantes

- El **Modelo 1A** está entrenado parcialmente sobre datos sintéticos.
  El análisis estadístico reveló inversiones de algunos factores. Por eso
  pesa solo el 30% en la decisión final.
- La **calculadora de literatura** asume independencia multiplicativa
  entre factores y usa tasas SEER (población EE.UU.).
- El **modelo CT** está entrenado sobre un dominio específico (MSD Task10).
  Imágenes de otros equipos o protocolos pueden dar resultados poco fiables.
- Solo analiza **slices 2D**. Un radiólogo evalúa el volumen completo en 3D.

---

#### Fuentes científicas
""")

    for clave in ["SEER", "Freedman2009", "Butterworth2006", "Ma2013",
                   "Johnson2013", "MSD"]:
        render_fuente_card(clave)

    st.markdown("""
---

#### Glosario completo de términos técnicos
""")

    for termino, definicion in sorted(GLOSARIO.items()):
        st.markdown(f"""
<div class="glosario-item">
    <span class="glosario-termino">{termino}</span>
    <span class="glosario-def">{definicion}</span>
</div>
""", unsafe_allow_html=True)

    
render_footer()