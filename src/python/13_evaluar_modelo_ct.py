from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from msd_colon_slice_dataset_cached import MSDColonSliceDatasetCached
from entrenar_msd_colon_unet_cached import UNetPequena

BASE_DIR = Path(__file__).resolve().parents[2]
VAL_CSV = BASE_DIR / "data" / "processed" / "msd_colon_slices" / "val_slices.csv"
MODEL_PATH = BASE_DIR / "models" / "msd_colon_unet_cached_best.pth"
OUTPUT_DIR = BASE_DIR / "models" / "ct_evaluacion"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Umbrales
THRESHOLD_MASK = 0.35      # binarización de la máscara
THRESHOLD_CLASS = 0.60     # clasificación slice-level


def calcular_dice(pred, true, smooth=1e-6):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    inter = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true)
    return (2 * inter + smooth) / (union + smooth)


def calcular_iou(pred, true, smooth=1e-6):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    inter = np.sum(pred * true)
    union = np.sum((pred + true) > 0)
    return (inter + smooth) / (union + smooth)


def evaluar() -> None:
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"No existe: {VAL_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = MSDColonSliceDatasetCached(VAL_CSV, mode="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNetPequena().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dices_todos, ious_todos = [], []
    dices_pos, ious_pos = [], []
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            label = int(batch["label"].cpu().numpy()[0])

            logits = model(image)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > THRESHOLD_MASK).float()

            pred_np = pred_mask.cpu().numpy()[0, 0]
            true_np = mask.cpu().numpy()[0, 0]

            dice = calcular_dice(pred_np, true_np)
            iou = calcular_iou(pred_np, true_np)
            dices_todos.append(dice)
            ious_todos.append(iou)

            if true_np.sum() > 0:
                dices_pos.append(dice)
                ious_pos.append(iou)

            y_true.append(label)
            y_prob.append(float(probs.max().cpu().item()))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= THRESHOLD_CLASS).astype(int)

    # Métricas de clasificación
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else None

    # Mejor umbral por índice de Youden
    if auc is not None:
        fpr, tpr, ths = roc_curve(y_true, y_prob)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        mejor_umbral = float(ths[best_idx])
        mejor_tpr = float(tpr[best_idx])
        mejor_fpr = float(fpr[best_idx])
    else:
        fpr = tpr = ths = None
        mejor_umbral = mejor_tpr = mejor_fpr = None

    metricas = {
        "configuracion": {
            "threshold_mask": THRESHOLD_MASK,
            "threshold_class": THRESHOLD_CLASS,
            "device": DEVICE,
            "n_slices_val": int(len(y_true)),
            "n_slices_positivos": int(np.sum(y_true == 1)),
            "n_slices_negativos": int(np.sum(y_true == 0)),
        },
        "segmentacion": {
            "dice_global": float(np.mean(dices_todos)),
            "iou_global": float(np.mean(ious_todos)),
            "dice_positivos": float(np.mean(dices_pos)) if dices_pos else None,
            "iou_positivos": float(np.mean(ious_pos)) if ious_pos else None,
            "n_slices_con_tumor": len(dices_pos),
        },
        "clasificacion_slice_level": {
            "auc": float(auc) if auc is not None else None,
            "sensibilidad": float(sens),
            "especificidad": float(spec),
            "valor_predictivo_positivo": float(ppv),
            "valor_predictivo_negativo": float(npv),
            "matriz_confusion": {
                "TN": int(tn), "FP": int(fp),
                "FN": int(fn), "TP": int(tp),
            },
        },
        "youden_optimo": {
            "umbral_optimo": mejor_umbral,
            "tpr_en_umbral": mejor_tpr,
            "fpr_en_umbral": mejor_fpr,
        },
    }

    # Guardar JSON de métricas
    json_path = OUTPUT_DIR / "metricas_modelo_ct.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print(f"\nMétricas guardadas en: {json_path}")

    # Curva ROC
    if fpr is not None:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.scatter(
            mejor_fpr, mejor_tpr, color="red", s=80, zorder=5,
            label=f"Youden óptimo @ {mejor_umbral:.3f}",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC - Modelo CT (clasificación slice-level)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = OUTPUT_DIR / "roc_curve_ct.png"
        plt.savefig(roc_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Curva ROC guardada en: {roc_path}")

    # Matriz de confusión
    if cm.size == 4:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sano", "Tumor"])
        ax.set_yticklabels(["Sano", "Tumor"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"Matriz de confusión (umbral={THRESHOLD_CLASS})")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14,
                )
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = OUTPUT_DIR / "confusion_matrix_ct.png"
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Matriz de confusión guardada en: {cm_path}")

    # Resumen por consola
    print("\n=== Resumen de evaluación CT ===")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    evaluar()

"""
Evaluación completa del Modelo 2 (CT - UNet) sobre el conjunto de validación.

Produce:
- metricas_modelo_ct.json con todas las métricas (Dice, IoU, AUC, sensibilidad,
  especificidad, mejor umbral por Youden, etc.).
- roc_curve_ct.png con la curva ROC.
- confusion_matrix_ct.png con la matriz de confusión.
"""