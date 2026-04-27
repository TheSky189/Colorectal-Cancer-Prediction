from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from msd_colon_slice_dataset_cached import MSDColonSliceDatasetCached
from entrenar_msd_colon_unet_cached import UNetPequena

BASE_DIR = Path(__file__).resolve().parents[2]

VAL_CSV = BASE_DIR / "data" / "processed" / "msd_colon_slices" / "val_slices.csv"
MODEL_PATH = BASE_DIR / "models" / "msd_colon_unet_cached_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
THRESHOLD_MASK = 0.35
THRESHOLD_CLASS = 0.60

def calcular_dice(pred_mask, true_mask, smooth=1e-6):
    pred = pred_mask.reshape(-1)
    true = true_mask.reshape(-1)

    inter = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true)

    return (2 * inter + smooth) / (union + smooth)

def calcular_iou(pred_mask, true_mask, smooth=1e-6):
    pred = pred_mask.reshape(-1)
    true = true_mask.reshape(-1)

    inter = np.sum(pred * true)
    union = np.sum((pred + true) > 0)

    return (inter + smooth) / (union + smooth)

def evaluar():
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"No existe: {VAL_CSV}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe: {MODEL_PATH}")

    dataset = MSDColonSliceDatasetCached(VAL_CSV, mode="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNetPequena().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dices_todos = []
    ious_todos = []
    dices_positivos = []
    ious_positivos = []

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            label = int(batch["label"].cpu().numpy()[0])

            logits = model(image)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > THRESHOLD_MASK).float()

            pred_mask_np = pred_mask.cpu().numpy()[0, 0]
            true_mask_np = mask.cpu().numpy()[0, 0]

            dice = calcular_dice(pred_mask_np, true_mask_np)
            iou = calcular_iou(pred_mask_np, true_mask_np)

            dices_todos.append(dice)
            ious_todos.append(iou)

            if true_mask_np.sum() > 0:
                dices_positivos.append(dice)
                ious_positivos.append(iou)

            prob_slice = float(probs.max().cpu().item())
            pred_label = 1 if prob_slice >= THRESHOLD_CLASS else 0

            y_true.append(label)
            y_pred.append(pred_label)
            y_prob.append(prob_slice)

    print("Resultados de segmentación (todos los slices):")
    print(f"Dice medio global: {np.mean(dices_todos):.4f}")
    print(f"IoU media global: {np.mean(ious_todos):.4f}")

    if len(dices_positivos) > 0:
        print("\nResultados de segmentación (solo slices positivos):")
        print(f"Dice medio positivos: {np.mean(dices_positivos):.4f}")
        print(f"IoU media positivos: {np.mean(ious_positivos):.4f}")

    print("\nResultados de clasificación derivada:")
    print(f"Umbral de máscara: {THRESHOLD_MASK:.2f}")
    print(f"Umbral de clasificación: {THRESHOLD_CLASS:.2f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print("Matriz de confusión:")
    print(confusion_matrix(y_true, y_pred))

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC AUC (clasificación derivada): {auc:.4f}")

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)

        print(f"Mejor umbral sugerido por Youden: {thresholds[best_idx]:.4f}")
        print(f"TPR en mejor umbral: {tpr[best_idx]:.4f}")
        print(f"FPR en mejor umbral: {fpr[best_idx]:.4f}")

if __name__ == "__main__":
    evaluar()
