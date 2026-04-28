from pathlib import Path
from urllib.request import urlretrieve
import sys


# Configuración del Release que contiene los modelos
GITHUB_USER  = "TheSky189"
GITHUB_REPO  = "Colorectal-Cancer-Prediction"
RELEASE_TAG  = "v1.0"  # Tag del release con los modelos

# Lista de modelos a descargar (filename -> tamaño aproximado en MB para info)
MODELS = {
    "calibrated_model.joblib":            "Modelo clínico (Random Forest calibrado)",
    "msd_colon_unet_cached_best.pth":     "Modelo CT (U-Net + ResNet-34, v4)",
}


def _release_url(filename: str) -> str:
    """Construye la URL de descarga de un asset de Release."""
    return (
        f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"
        f"/releases/download/{RELEASE_TAG}/{filename}"
    )


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """Descarga un archivo con barra de progreso simple en consola."""
    print(f"  Descargando {label}...")
    print(f"    URL: {url}")
    print(f"    Destino: {dest}")

    def _hook(blocks_done, block_size, total_size):
        if total_size > 0:
            done = blocks_done * block_size
            pct = min(100, 100 * done / total_size)
            mb_done = done / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progreso: {pct:5.1f}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
            sys.stdout.flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dest), reporthook=_hook)
    print()  # newline tras la barra


def ensure_models_present(models_dir: Path, verbose: bool = True) -> dict:
    """
    Comprueba que todos los modelos están en models/. Si falta alguno,
    lo descarga desde GitHub Releases.

    Devuelve un dict {filename: bool} indicando si cada modelo está
    disponible (True = OK, False = falló la descarga).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    status = {}
    missing = [(name, desc) for name, desc in MODELS.items()
               if not (models_dir / name).exists()]

    if not missing:
        if verbose:
            print("[INFO] Todos los modelos están disponibles localmente.")
        return {name: True for name in MODELS}

    if verbose:
        print(f"[INFO] Faltan {len(missing)} modelo(s). Descargando desde GitHub Releases...")
        print(f"[INFO] Esto solo ocurre la primera vez que ejecutas la app.\n")

    for filename, description in missing:
        dest = models_dir / filename
        url = _release_url(filename)
        try:
            _download_with_progress(url, dest, description)
            status[filename] = True
        except Exception as exc:
            print(f"\n[ERROR] No se pudo descargar {filename}: {exc}")
            print(f"[ERROR] Descárgalo manualmente desde:")
            print(f"        https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}")
            print(f"[ERROR] Y colócalo en: {models_dir / filename}")
            status[filename] = False

    # Modelos que ya estaban
    for name in MODELS:
        if name not in status:
            status[name] = True

    if verbose:
        print()
        if all(status.values()):
            print("[INFO] Descarga completada con éxito.")
        else:
            print("[WARN] Algunos modelos no se han podido descargar. Ver mensajes arriba.")

    return status


if __name__ == "__main__":
    # Permite descargar manualmente desde la línea de comandos:
    #   python src/python/utils/model_downloader.py
    base_dir = Path(__file__).resolve().parents[3]
    models_dir = base_dir / "models"
    print(f"[INFO] Directorio destino: {models_dir}\n")
    ensure_models_present(models_dir, verbose=True)


"""
src/python/utils/model_downloader.py

Descarga automática de modelos pesados desde GitHub Releases.

Cuando alguien hace git clone del repo, las carpetas models/ está vacía
porque los .pth y .joblib están en .gitignore. Este módulo se encarga de
descargarlos automáticamente la primera vez que se ejecuta la app.

Los modelos están publicados como assets de un Release en:
https://github.com/TheSky189/Colorectal-Cancer-Prediction/releases
"""