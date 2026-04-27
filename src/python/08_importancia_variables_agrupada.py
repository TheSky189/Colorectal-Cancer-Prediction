from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "clinical_model_bundle.joblib"
OUTPUT_PNG_PATH = BASE_DIR / "models" / "feature_importance_grouped.png"
OUTPUT_CSV_PATH = BASE_DIR / "models" / "feature_importance_grouped.csv"

ORIGINAL_FEATURES = [
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
    "Urban_or_Rural",
    "Economic_Classification",
    "Healthcare_Access",
    "Insurance_Status"
]

def agrupar_variable(feature_name: str) -> str:
    if feature_name == "Age":
        return "Age"
    for original in ORIGINAL_FEATURES:
        if feature_name == original or feature_name.startswith(original + "_"):
            return original
    return feature_name

def obtener_importancia_agrupada() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model_pipeline = bundle["model"]

    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    numeric_features = bundle["numeric_features"]

    categorical_transformer = preprocessor.named_transformers_["cat"]
    onehot = categorical_transformer.named_steps["onehot"]
    categorical_features = bundle["categorical_features"]
    categorical_feature_names = onehot.get_feature_names_out(categorical_features)

    feature_names = list(numeric_features) + list(categorical_feature_names)

    if not hasattr(classifier, "feature_importances_"):
        raise ValueError("Este script está pensado para modelos tipo árbol con feature_importances_.")

    importances = classifier.feature_importances_

    df = pd.DataFrame({
        "Feature_Encoded": feature_names,
        "Importance": importances
    })

    df["Feature_Original"] = df["Feature_Encoded"].apply(agrupar_variable)

    grouped = (
        df.groupby("Feature_Original", as_index=False)["Importance"]
        .sum()
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    grouped.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Importancia agrupada por variable original:")
    print(grouped)

    top_df = grouped.iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["Feature_Original"], top_df["Importance"])
    plt.xlabel("Importancia agregada")
    plt.ylabel("Variable original")
    plt.title("Importancia agrupada de variables - Modelo Clínico")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráfico guardado en: {OUTPUT_PNG_PATH}")
    print(f"CSV guardado en: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    obtener_importancia_agrupada()