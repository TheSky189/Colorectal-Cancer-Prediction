from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_CSV = BASE_DIR / "data" / "processed" / "msd_colon" / "msd_colon_train.csv"
VAL_CSV = BASE_DIR / "data" / "processed" / "msd_colon" / "msd_colon_val.csv"

# Carpeta NUEVA — no sobreescribe la existente
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "msd_colon_slices_256"
TRAIN_OUT_CSV = OUTPUT_DIR / "train_slices.csv"
VAL_OUT_CSV = OUTPUT_DIR / "val_slices.csv"

IMAGE_SIZE = (256, 256)  # antes: (192, 192)
SEED = 42


def normalizar_ct(vol):
    vol = vol.astype(np.float32)
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)

    if p99 > p1:
        vol = np.clip(vol, p1, p99)
        vol = (vol - p1) / (p99 - p1)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)

    return vol


def resize_2d(arr, output_shape, order):
    zy = output_shape[0] / arr.shape[0]
    zx = output_shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx), order=order)


def procesar_split(csv_path, split_name, negative_ratio):
    df = pd.read_csv(csv_path)
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    filas = []
    rng = np.random.default_rng(SEED)

    for _, row in df.iterrows():
        case_id = row["case_id"]
        image_path = row["image_path"]
        label_path = row["label_path"]

        vol = nib.load(image_path).get_fdata()
        lbl = nib.load(label_path).get_fdata()

        vol = normalizar_ct(vol)
        lbl = (lbl > 0).astype(np.uint8)

        profundidad = vol.shape[2]

        for z in range(profundidad):
            image = vol[:, :, z]
            mask = lbl[:, :, z]

            label = 1 if mask.sum() > 0 else 0

            # Muestreo de negativos
            if label == 0 and rng.random() > negative_ratio:
                continue

            # Resize a 256×256
            image = resize_2d(image, IMAGE_SIZE, order=1).astype(np.float32)
            mask = resize_2d(mask, IMAGE_SIZE, order=0).astype(np.uint8)

            out_path = split_dir / f"{case_id}_z{z:03d}.npz"

            np.savez_compressed(
                out_path,
                image=image,
                mask=mask,
                label=np.array(label, dtype=np.uint8),
                case_id=case_id,
                slice_idx=np.array(z, dtype=np.int32),
            )

            filas.append({
                "path": str(out_path),
                "case_id": case_id,
                "slice_idx": z,
                "label": label,
            })

    out_csv = TRAIN_OUT_CSV if split_name == "train" else VAL_OUT_CSV
    out_df = pd.DataFrame(filas)
    out_df.to_csv(out_csv, index=False)

    print(f"\nSplit procesado: {split_name}")
    print(f"Total slices guardados: {len(out_df)}")
    print(out_df["label"].value_counts(dropna=False))
    print(f"CSV guardado en: {out_csv}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"No existe: {TRAIN_CSV}")
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"No existe: {VAL_CSV}")

    procesar_split(TRAIN_CSV, "train", negative_ratio=0.20)
    procesar_split(VAL_CSV, "val", negative_ratio=0.50)

    print("\nPreprocesamiento a 256×256 completado.")
    print("Carpeta original (192×192) intacta.")


if __name__ == "__main__":
    main()

"""
Preprocesado de slices MSD Colon a resolución 256×256.

Versión 2 del preprocesado. Diferencias con el original
(preprocesar_msd_colon_slices.py):
- Resolución 256×256 en lugar de 192×192 (más detalle espacial,
  crítico para tumores pequeños).
- Output en carpeta NUEVA: data/processed/msd_colon_slices_256/
- NO toca la carpeta msd_colon_slices/ existente. Eso permite
  entrenar v4 en paralelo sin romper v1/v2/v3.

Ratios de muestreo de negativos mantenidos igual que el preprocesado
original (train=0.20, val=0.50) para que la comparación con v1-v3
sea justa.
"""