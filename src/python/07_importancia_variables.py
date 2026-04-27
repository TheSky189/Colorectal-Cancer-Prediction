from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "clinical_model_bundle.joblib"
OUTPUT_PNG_PATH = BASE_DIR / "models" / "feature_importance.png"
OUTPUT_CSV_PATH = BASE_DIR / "models" / "feature_importance.csv"

def obtener_nombres_variables(bundle, model_pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]

    numeric_features = bundle["numeric_features"]

    categorical_transformer = preprocessor.named_transformers_["cat"]
    onehot = categorical_transformer.named_steps["onehot"]

    categorical_features = bundle["categorical_features"]
    categorical_feature_names = onehot.get_feature_names_out(categorical_features)

    feature_names = list(numeric_features) + list(categorical_feature_names)
    return feature_names

def obtener_importancias(classifier):
    # Caso 1: modelos tipo árbol
    if hasattr(classifier, "feature_importances_"):
        return classifier.feature_importances_, "feature_importances"

    # Caso 2: Logistic Regression
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_

        # Clasificación binaria → shape (1, n_features)
        if coef.ndim == 2 and coef.shape[0] == 1:
            importancias = np.abs(coef[0])
        else:
            importancias = np.mean(np.abs(coef), axis=0)

        return importancias, "coef_abs"

    raise ValueError(
        "El clasificador seleccionado no soporta ni feature_importances_ ni coef_."
    )

def obtener_importancia_variables() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {MODEL_PATH}. Ejecuta antes 04_entrenar_modelo.py"
        )

    bundle = joblib.load(MODEL_PATH)
    model_pipeline = bundle["model"]

    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]

    feature_names = obtener_nombres_variables(bundle, model_pipeline)
    importances, metodo = obtener_importancias(classifier)

    importance_df = pd.DataFrame({
        "Variable": feature_names,
        "Importancia": importances
    })

    importance_df = importance_df.sort_values(by="Importancia", ascending=False).reset_index(drop=True)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Método de importancia utilizado: {metodo}")
    print("\nTop 15 variables más importantes:")
    print(importance_df.head(15))

    top_n = 15
    top_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["Variable"], top_df["Importancia"])
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.title("Importancia de variables - Modelo Clínico")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGráfico guardado en: {OUTPUT_PNG_PATH}")
    print(f"CSV guardado en: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    obtener_importancia_variables()