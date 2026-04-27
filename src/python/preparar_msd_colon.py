from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "data" / "raw" / "Task10_Colon"

DATASET_JSON = DATASET_DIR / "dataset.json"
IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"
IMAGES_TS = DATASET_DIR / "imagesTs"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "msd_colon"
INDEX_CSV = OUTPUT_DIR / "msd_colon_training_index.csv"
TRAIN_CSV = OUTPUT_DIR / "msd_colon_train.csv"
VAL_CSV = OUTPUT_DIR / "msd_colon_val.csv"
TEST_INFER_CSV = OUTPUT_DIR / "msd_colon_test_inference.csv"

SEED = 42

def preparar_msd_colon():
    if not DATASET_JSON.exists():
        raise FileNotFoundError(f"No existe: {DATASET_JSON}")
    if not IMAGES_TR.exists():
        raise FileNotFoundError(f"No existe: {IMAGES_TR}")
    if not LABELS_TR.exists():
        raise FileNotFoundError(f"No existe: {LABELS_TR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    filas_train = []
    for item in meta["training"]:
        image_rel = item["image"].replace("./", "")
        label_rel = item["label"].replace("./", "")

        image_path = DATASET_DIR / image_rel
        label_path = DATASET_DIR / label_rel

        case_id = Path(image_rel).name.replace(".nii.gz", "")

        if not image_path.exists():
            raise FileNotFoundError(f"No existe la imagen: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"No existe la máscara: {label_path}")

        filas_train.append({
            "case_id": case_id,
            "image_path": str(image_path),
            "label_path": str(label_path)
        })

    df = pd.DataFrame(filas_train).sort_values("case_id").reset_index(drop=True)
    df.to_csv(INDEX_CSV, index=False)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, groups=df["case_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)

    filas_test = []
    if IMAGES_TS.exists():
        for p in sorted(IMAGES_TS.glob("*.nii.gz")):
            filas_test.append({
                "case_id": p.name.replace(".nii.gz", ""),
                "image_path": str(p)
            })

    df_test = pd.DataFrame(filas_test)
    df_test.to_csv(TEST_INFER_CSV, index=False)

    print("Preparación del dataset MSD Colon completada.")
    print(f"Casos de entrenamiento totales: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Validación: {len(val_df)}")
    print(f"Test para inferencia: {len(df_test)}")
    print(f"Índice global guardado en: {INDEX_CSV}")
    print(f"Train guardado en: {TRAIN_CSV}")
    print(f"Validación guardado en: {VAL_CSV}")
    print(f"Test guardado en: {TEST_INFER_CSV}")

if __name__ == "__main__":
    preparar_msd_colon()