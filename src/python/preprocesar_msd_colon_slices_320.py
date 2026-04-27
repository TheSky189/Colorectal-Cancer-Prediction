import os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# Configuración
RAW_DIR     = Path('data/raw/Task10_Colon')
OUTPUT_DIR  = Path('data/processed/msd_colon_slices_320')
TARGET_SIZE = 320
VAL_FRACTION = 0.2     # 20% de pacientes para validación
RANDOM_SEED = 42

# Filtros
PCTL_LOW  = 1
PCTL_HIGH = 99


def normalize_volume(volume):
    """Normaliza al rango [0, 1] mediante percentiles 1 y 99."""
    p_low  = np.percentile(volume, PCTL_LOW)
    p_high = np.percentile(volume, PCTL_HIGH)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume.astype(np.float32)


def resize_slice(slice_2d, target_size, order):
    """Redimensiona una slice 2D al tamaño objetivo."""
    h, w = slice_2d.shape
    zh = target_size / h
    zw = target_size / w
    return zoom(slice_2d, (zh, zw), order=order, mode='constant', cval=0.0)


def process_volume(image_path, label_path):
    """Procesa un volumen CT + máscara, devuelve listas de slices."""
    img_nii = nib.load(str(image_path))
    lbl_nii = nib.load(str(label_path))

    image = img_nii.get_fdata().astype(np.float32)
    label = lbl_nii.get_fdata().astype(np.float32)

    image = normalize_volume(image)

    slices_img = []
    slices_msk = []

    n_slices = image.shape[2]
    for z in range(n_slices):
        img_slice = image[:, :, z]
        msk_slice = label[:, :, z]

        # Redimensionar
        img_resized = resize_slice(img_slice, TARGET_SIZE, order=1)
        msk_resized = resize_slice(msk_slice, TARGET_SIZE, order=0)
        msk_resized = (msk_resized > 0.5).astype(np.float32)

        # Asegurar shape
        img_resized = np.clip(img_resized, 0.0, 1.0)
        if img_resized.shape != (TARGET_SIZE, TARGET_SIZE):
            continue
        if msk_resized.shape != (TARGET_SIZE, TARGET_SIZE):
            continue

        slices_img.append(img_resized.astype(np.float32))
        slices_msk.append(msk_resized.astype(np.float32))

    return slices_img, slices_msk


def main():
    print("=" * 70)
    print("Preprocesado MSD Colon -> slices 320x320")
    print("=" * 70)

    images_dir = RAW_DIR / 'imagesTr'
    labels_dir = RAW_DIR / 'labelsTr'

    if not images_dir.exists():
        print(f"[ERROR] No existe {images_dir}")
        print("[ERROR] Descarga el dataset MSD Task10 desde http://medicaldecathlon.com/")
        return

    image_files = sorted(images_dir.glob('colon_*.nii.gz'))
    print(f"[INFO] Volúmenes encontrados: {len(image_files)}")

    # Split por paciente (no por slice)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(image_files))
    n_val = int(len(image_files) * VAL_FRACTION)
    val_idx   = set(indices[:n_val].tolist())
    train_idx = set(indices[n_val:].tolist())

    train_dir = OUTPUT_DIR / 'train'
    val_dir   = OUTPUT_DIR / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_train_slices = 0
    n_val_slices   = 0

    for i, img_path in enumerate(image_files):
        case_id = img_path.stem.replace('.nii', '')  # colon_001
        lbl_path = labels_dir / img_path.name

        if not lbl_path.exists():
            print(f"[WARN] Sin máscara: {case_id}")
            continue

        is_val = i in val_idx
        out_dir = val_dir if is_val else train_dir

        try:
            slices_img, slices_msk = process_volume(img_path, lbl_path)
        except Exception as e:
            print(f"[ERROR] {case_id}: {e}")
            continue

        for z, (img_s, msk_s) in enumerate(zip(slices_img, slices_msk)):
            out_path = out_dir / f"{case_id}_z{z:03d}.npz"
            np.savez_compressed(out_path, image=img_s, mask=msk_s)

        n = len(slices_img)
        if is_val:
            n_val_slices += n
        else:
            n_train_slices += n

        print(f"  [{i+1:3d}/{len(image_files)}] {case_id} -> {n} slices ({'val' if is_val else 'train'})")

    print()
    print("=" * 70)
    print(f"[DONE] Train: {n_train_slices} slices  ({len(train_idx)} pacientes)")
    print(f"[DONE] Val:   {n_val_slices} slices  ({len(val_idx)} pacientes)")
    print(f"[DONE] Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()


"""
================================================================================
preprocesar_msd_colon_slices_320.py
================================================================================
Preprocesado del dataset MSD Task10 Colon a slices 320x320 para el modelo v5.

DIFERENCIA CON v4 (256x256): mayor resolución para preservar detalles finos
del tumor que en 256x256 se perdían.

ENTRADA:
  data/raw/Task10_Colon/
    ├── imagesTr/  (volúmenes CT, .nii.gz)
    └── labelsTr/  (máscaras de segmentación, .nii.gz)

SALIDA:
  data/processed/msd_colon_slices_320/
    ├── train/  (slices .npz)
    └── val/    (slices .npz)

Cada .npz contiene:
  image: float32 (320, 320), normalizado [0, 1]
  mask:  float32 (320, 320), binario {0, 1}

USO:
  python src/python/preprocesar_msd_colon_slices_320.py
================================================================================
"""