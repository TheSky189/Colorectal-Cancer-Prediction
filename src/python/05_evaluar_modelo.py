from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "clinical_model_bundle.joblib"

def evaluar_modelo() -> None:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset de test: {TEST_DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    X = df.drop(columns=["Cancer"])
    y = df["Cancer"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\nClassification Report (solo test set):")
    print(classification_report(y, y_pred))

    print("\nConfusion Matrix (solo test set):")
    print(confusion_matrix(y, y_pred))

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC - Modelo Clínico (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluar_modelo()