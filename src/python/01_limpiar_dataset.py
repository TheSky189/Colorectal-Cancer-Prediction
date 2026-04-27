from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "raw" / "colorectal_cancer_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "cancer_cleaned.csv"

FEATURES = [
    "Age",
    "Gender",
    "Family_History",
    "Smoking_History",
    "Alcohol_Consumption",
    "Obesity_BMI",
    "Diet_Risk",
    "Physical_Activity",
    "Diabetes",
    "Inflammatory_Bowel_Disease",
    "Genetic_Mutation",
    "Screening_History",  
    "Urban_or_Rural",
    "Healthcare_Access",
]


def limpiar_dataset() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("Forma original del dataset:", df.shape)
    print("\nColumnas originales:")
    print(df.columns.tolist())

    cancer_df = df[FEATURES].copy()
    cancer_df["Age"] = pd.to_numeric(cancer_df["Age"], errors="coerce")
    cancer_df = cancer_df.dropna().copy()
    cancer_df = cancer_df[(cancer_df["Age"] >= 18) & (cancer_df["Age"] <= 100)]
    cancer_df["Cancer"] = 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cancer_df.to_csv(OUTPUT_PATH, index=False)

    print("\nForma del dataset limpio (cohorte cáncer):", cancer_df.shape)
    print(f"Archivo guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    limpiar_dataset()