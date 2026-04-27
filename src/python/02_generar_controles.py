from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CANCER_INPUT_PATH = BASE_DIR / "data" / "processed" / "cancer_cleaned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "control_population.csv"

SEED = 42


def sample_from_distribution(values, probabilities, size):
    return np.random.choice(values, size=size, p=probabilities)


def generar_controles() -> None:
    if not CANCER_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo previo: {CANCER_INPUT_PATH}. "
            "Ejecuta antes 01_limpiar_dataset.py"
        )

    np.random.seed(SEED)

    cancer_df = pd.read_csv(CANCER_INPUT_PATH)
    n = len(cancer_df)

    control_df = pd.DataFrame()

    # 1. Edad: misma distribución que la cohorte cáncer (evita age bias)
    control_df["Age"] = cancer_df["Age"].sample(frac=1, replace=True).values

    # 2. Variables base
    control_df["Gender"] = sample_from_distribution(["M", "F"], [0.49, 0.51], n)

    control_df["Family_History"] = sample_from_distribution(
        ["Yes", "No"], [0.18, 0.82], n
    )

    control_df["Smoking_History"] = sample_from_distribution(
        ["Yes", "No"], [0.40, 0.60], n
    )

    control_df["Obesity_BMI"] = sample_from_distribution(
        ["Normal", "Overweight", "Obese"], [0.35, 0.42, 0.23], n
    )

    control_df["Physical_Activity"] = sample_from_distribution(
        ["Low", "Moderate", "High"], [0.28, 0.47, 0.25], n
    )

    control_df["Inflammatory_Bowel_Disease"] = sample_from_distribution(
        ["Yes", "No"], [0.04, 0.96], n
    )

    control_df["Urban_or_Rural"] = sample_from_distribution(
        ["Urban", "Rural"], [0.68, 0.32], n
    )

    control_df["Healthcare_Access"] = sample_from_distribution(
        ["Low", "Moderate", "High"], [0.22, 0.50, 0.28], n
    )

    # 3. Variables dependientes

    # Smoking -> Alcohol
    alcohol = []
    for sm in control_df["Smoking_History"]:
        if sm == "Yes":
            alcohol.append(np.random.choice(["Yes", "No"], p=[0.65, 0.35]))
        else:
            alcohol.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
    control_df["Alcohol_Consumption"] = alcohol

    # Smoking -> Diet
    diet = []
    for sm in control_df["Smoking_History"]:
        if sm == "Yes":
            diet.append(np.random.choice(["Low", "Moderate", "High"],
                                          p=[0.15, 0.35, 0.50]))
        else:
            diet.append(np.random.choice(["Low", "Moderate", "High"],
                                          p=[0.40, 0.40, 0.20]))
    control_df["Diet_Risk"] = diet

    # Obesity -> Diabetes
    diabetes = []
    for bmi in control_df["Obesity_BMI"]:
        if bmi == "Obese":
            diabetes.append(np.random.choice(["Yes", "No"], p=[0.30, 0.70]))
        elif bmi == "Overweight":
            diabetes.append(np.random.choice(["Yes", "No"], p=[0.18, 0.82]))
        else:
            diabetes.append(np.random.choice(["Yes", "No"], p=[0.08, 0.92]))
    control_df["Diabetes"] = diabetes

    # Family History -> Genetic Mutation
    genetic = []
    for fh in control_df["Family_History"]:
        if fh == "Yes":
            genetic.append(np.random.choice(["Yes", "No"], p=[0.25, 0.75]))
        else:
            genetic.append(np.random.choice(["Yes", "No"], p=[0.05, 0.95]))
    control_df["Genetic_Mutation"] = genetic

    # Age -> Screening History
    screening = []
    for age in control_df["Age"]:
        if age > 60:
            screening.append(np.random.choice(
                ["Regular", "Irregular", "Never"], p=[0.60, 0.25, 0.15]
            ))
        elif age > 45:
            screening.append(np.random.choice(
                ["Regular", "Irregular", "Never"], p=[0.40, 0.35, 0.25]
            ))
        else:
            screening.append(np.random.choice(
                ["Regular", "Irregular", "Never"], p=[0.20, 0.30, 0.50]
            ))
    control_df["Screening_History"] = screening

    # 4. Etiqueta
    control_df["Cancer"] = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    control_df.to_csv(OUTPUT_PATH, index=False)

    print("Controles sintéticos generados correctamente.")
    print("Forma del dataset control:", control_df.shape)
    print(f"Archivo guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    generar_controles()