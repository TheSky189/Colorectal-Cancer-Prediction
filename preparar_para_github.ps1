# =====================================================================
# preparar_para_github.ps1
# =====================================================================
# Script de limpieza para dejar el proyecto listo para subir a GitHub.
# Mueve archivos viejos/innecesarios a una carpeta _archivados_ y
# muestra un resumen de lo que se va a subir.
#
# IMPORTANTE: este script NO borra nada. Solo mueve a _archivados_
# para que puedas revisar antes de eliminar.
#
# Uso:
#   .\preparar_para_github.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  PREPARAR PROYECTO PARA GITHUB" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archivoDir = "_archivados_pre_github_$timestamp"

# Crear carpeta de archivados
New-Item -ItemType Directory -Path $archivoDir -Force | Out-Null
Write-Host "[1/5] Carpeta de archivados creada: $archivoDir" -ForegroundColor Green

# ===== Mover archivos obsoletos =====
$obsoletos = @(
    "src/python/preprocesar_msd_colon_slices.py",
    "src/python/msd_colon_slice_dataset_cached.py",
    "src/python/17_comparar_ct_todas_versiones.py",
    "README_MEJORAS.md",
    "README_SCRIPTS_R.md",
    "README_v4.md",
    "limpiar_proyecto.ps1",
    "backup_colon_cancer_ai.zip"
)

Write-Host ""
Write-Host "[2/5] Archivando versiones antiguas y archivos locales..." -ForegroundColor Yellow

foreach ($f in $obsoletos) {
    if (Test-Path $f) {
        Move-Item $f $archivoDir -Force
        Write-Host "    - $f -> archivado" -ForegroundColor DarkGray
    }
}

# Carpetas obsoletas
$carpetasObsoletas = @(
    "src/python/research"
)

foreach ($c in $carpetasObsoletas) {
    if (Test-Path $c) {
        Move-Item $c $archivoDir -Force
        Write-Host "    - $c/ -> archivado" -ForegroundColor DarkGray
    }
}

# ===== Limpiar pycache =====
Write-Host ""
Write-Host "[3/5] Eliminando carpetas __pycache__..." -ForegroundColor Yellow

Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
        Remove-Item $_.FullName -Recurse -Force
        Write-Host "    - $($_.FullName) -> eliminado" -ForegroundColor DarkGray
    }

# ===== Verificar archivos esenciales =====
Write-Host ""
Write-Host "[4/5] Verificando archivos esenciales..." -ForegroundColor Yellow

$esenciales = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "app/app_streamlit.py",
    "app/assets/styles.css",
    "src/python/utils/paciente_schema.py",
    "src/python/utils/calculadora_riesgo_literatura.py",
    "src/python/utils/decision_combinada.py",
    "src/r/01_analisis_exploratorio.R",
    "src/r/02_regresion_logistica.R",
    "src/r/03_visualizaciones_memoria.R"
)

$faltantes = @()
foreach ($f in $esenciales) {
    if (Test-Path $f) {
        Write-Host "    OK $f" -ForegroundColor Green
    } else {
        Write-Host "    FALTA $f" -ForegroundColor Red
        $faltantes += $f
    }
}

# ===== Resumen final =====
Write-Host ""
Write-Host "[5/5] Resumen del proyecto listo para GitHub:" -ForegroundColor Cyan
Write-Host ""

$tam_total = (Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "_archivados_|\.venv|\.git|data\\|models\\" } |
    Measure-Object -Property Length -Sum).Sum / 1MB

Write-Host "    Tamano aproximado a subir: $([math]::Round($tam_total, 2)) MB" -ForegroundColor White
Write-Host ""

if ($faltantes.Count -gt 0) {
    Write-Host "ATENCION: Faltan archivos esenciales:" -ForegroundColor Red
    $faltantes | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host ""
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  SIGUIENTES PASOS:" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Revisa la carpeta '$archivoDir' por si quieres recuperar algo" -ForegroundColor White
Write-Host "  2. Ejecuta los comandos git:" -ForegroundColor White
Write-Host ""
Write-Host "     git init" -ForegroundColor Yellow
Write-Host "     git add ." -ForegroundColor Yellow
Write-Host "     git status                  # revisa que nada raro vaya" -ForegroundColor Yellow
Write-Host "     git commit -m 'Initial commit: TFM Simulador CCR'" -ForegroundColor Yellow
Write-Host "     git branch -M main" -ForegroundColor Yellow
Write-Host "     git remote add origin https://github.com/[usuario]/colon_cancer_ai.git" -ForegroundColor Yellow
Write-Host "     git push -u origin main" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. Si una vez subido quieres BORRAR DEFINITIVAMENTE los archivados:" -ForegroundColor White
Write-Host "     Remove-Item -Recurse -Force '$archivoDir'" -ForegroundColor Yellow
Write-Host ""