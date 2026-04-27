from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CANCER_PATH = BASE_DIR / "data" / "processed" / "cancer_cleaned.csv"
CONTROL_PATH = BASE_DIR / "data" / "processed" / "control_population.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"

def construir_dataset() -> None:
    if not CANCER_PATH.exists():
        raise FileNotFoundError(f"No existe: {CANCER_PATH}")
    if not CONTROL_PATH.exists():
        raise FileNotFoundError(f"No existe: {CONTROL_PATH}")

    cancer_df = pd.read_csv(CANCER_PATH)
    control_df = pd.read_csv(CONTROL_PATH)

    df = pd.concat([cancer_df, control_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop(columns=["Screening_History"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset de entrenamiento construido correctamente.")
    print("\nDistribución de clases:")
    print(df["Cancer"].value_counts())
    print(f"\nArchivo guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    construir_dataset()