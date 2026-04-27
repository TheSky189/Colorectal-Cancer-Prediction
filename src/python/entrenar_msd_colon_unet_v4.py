from pathlib import Path
import copy
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "Falta segmentation-models-pytorch. Instala con:\n"
        "    pip install segmentation-models-pytorch"
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msd_colon_slice_dataset_cached_v3 import MSDColonSliceDatasetCachedV3

BASE_DIR = Path(__file__).resolve().parents[2]

# Slices a 256×256 — cuidado, carpeta distinta de la original
TRAIN_CSV = BASE_DIR / "data" / "processed" / "msd_colon_slices_256" / "train_slices.csv"
VAL_CSV = BASE_DIR / "data" / "processed" / "msd_colon_slices_256" / "val_slices.csv"

MODEL_PATH = BASE_DIR / "models" / "msd_colon_unet_v4_best.pth"

SEED = 42
BATCH_SIZE = 8            # Menor que v1-v3 (16) porque 256×256 + ResNet34 usa más memoria
EPOCHS = 50
LR = 1e-3                 # ResNet34 admite LR más alta al inicio
LR_ENCODER_FACTOR = 0.1   # encoder LR más bajo — es preentrenado, no queremos destruirlo

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Hiperparámetros
POS_WEIGHT = 3.0
TARGET_POSITIVE_RATIO = 0.40
PATIENCE = 12
TVERSKY_ALPHA = 0.3   # peso FP
TVERSKY_BETA = 0.7    # peso FN (más alto = más sensibilidad)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Pérdidas
def tversky_loss(logits, targets, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=1e-6):
    """
    Tversky index loss. Generalización de Dice:
    - alpha controla peso de FP
    - beta controla peso de FN
    - alpha=beta=0.5 es equivalente a Dice
    - beta>alpha → más sensibilidad (penaliza más FN)

    Calculada solo sobre slices con tumor (igual que dice_loss_positivos v1-v3).
    """
    probs = torch.sigmoid(logits)
    batch_size = targets.shape[0]
    probs = probs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    positivos = (targets.sum(dim=1) > 0)
    if positivos.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    probs = probs[positivos]
    targets = targets[positivos]

    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def calcular_dice_batch(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    batch_size = targets.shape[0]
    preds = preds.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    positivos = (targets.sum(dim=1) > 0)
    if positivos.sum() == 0:
        return None

    preds = preds[positivos]
    targets = targets[positivos]

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + smooth) / (union + smooth)
    return float(dice.mean().item())


def crear_sampler_ponderado(dataset, target_pos_ratio=TARGET_POSITIVE_RATIO):
    labels = dataset.df["label"].values
    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)

    peso_0 = 1.0 / max(n0, 1)
    peso_1 = (target_pos_ratio / (1 - target_pos_ratio)) * (n1 / max(n0, 1)) * peso_0

    pesos = np.array([peso_1 if y == 1 else peso_0 for y in labels], dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=pesos, num_samples=len(pesos), replacement=True
    )
    print(f"Sampler v4: ratio efectivo ~{target_pos_ratio:.2f} positivos")
    print(f"  n_pos_real: {n1}, n_neg_real: {n0}")
    return sampler


def construir_modelo():
    """
    UNet con encoder ResNet34 preentrenado en ImageNet.
    Input: 3 canales (el dataset ya replica 1→3).
    Output: 1 canal (máscara binaria logits).
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def buscar_mejor_umbral(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return {
        "umbral": float(thresholds[best_idx]),
        "sensibilidad": float(tpr[best_idx]),
        "especificidad": float(1 - fpr[best_idx]),
    }


def entrenar():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"No existe: {TRAIN_CSV}\n"
            f"Ejecuta primero: python src/python/preprocesar_msd_colon_slices_256.py"
        )
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"No existe: {VAL_CSV}")

    train_ds = MSDColonSliceDatasetCachedV3(TRAIN_CSV, mode="train")
    val_ds = MSDColonSliceDatasetCachedV3(VAL_CSV, mode="val")

    sampler = crear_sampler_ponderado(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = construir_modelo().to(DEVICE)

    # Optimizador con LR distinto para encoder y decoder
    # (técnica estándar en fine-tuning: encoder preentrenado aprende más despacio)
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.")]
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": LR * LR_ENCODER_FACTOR},
        {"params": decoder_params, "lr": LR},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    bce_seg = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mejor_score = -1.0
    mejor_modelo = None
    mejor_epoch = 0
    sin_mejora = 0
    historial = []
    mejores_predicciones = None

    print(f"\nDispositivo: {DEVICE}")
    print(f"Arquitectura: UNet + ResNet34 preentrenado ImageNet")
    print(f"Resolución: 256×256")
    print(f"Slices train: {len(train_ds)}   val: {len(val_ds)}")
    print(f"pos_weight: {POS_WEIGHT}")
    print(f"Tversky: α={TVERSKY_ALPHA}, β={TVERSKY_BETA}")
    print(f"Epochs máx: {EPOCHS}, paciencia: {PATIENCE}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)

            loss_bce = bce_seg(logits, masks)
            loss_tversky = tversky_loss(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_tversky

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validación
        model.eval()
        val_losses = []
        dices_val = []
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                logits = model(images)
                loss_bce = bce_seg(logits, masks)
                loss_tversky = tversky_loss(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_tversky
                val_losses.append(loss.item())

                dice = calcular_dice_batch(logits, masks, threshold=0.5)
                if dice is not None:
                    dices_val.append(dice)

                score_cls = torch.amax(
                    torch.sigmoid(logits).view(logits.size(0), -1), dim=1
                )
                y_true.extend(labels.cpu().numpy().tolist())
                y_prob.extend(score_cls.cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        dice_pos_val = float(np.mean(dices_val)) if dices_val else 0.0
        auc_val = (float(roc_auc_score(y_true, y_prob))
                   if len(np.unique(y_true)) > 1 else 0.0)

        # Score combinado: da igual peso a AUC (clasificación) y Dice (segmentación)
        score = 0.5 * auc_val + 0.5 * dice_pos_val

        scheduler.step(score)
        lr_actual = optimizer.param_groups[-1]["lr"]

        registro = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice_pos_val": dice_pos_val,
            "auc_val": auc_val,
            "score": score,
            "lr_decoder": lr_actual,
        }
        historial.append(registro)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"dice_pos={dice_pos_val:.4f} | auc={auc_val:.4f} | "
            f"score={score:.4f} | lr={lr_actual:.1e}"
        )

        if score > mejor_score:
            mejor_score = score
            mejor_modelo = copy.deepcopy(model.state_dict())
            mejor_epoch = epoch
            mejores_predicciones = (np.array(y_true), np.array(y_prob))
            sin_mejora = 0
        else:
            sin_mejora += 1

        if sin_mejora >= PATIENCE:
            print("\nParada temprana activada.")
            break

    # Threshold sweep
    print(f"\nMejor epoch: {mejor_epoch} (score={mejor_score:.4f})")
    info_umbral = None
    if mejores_predicciones is not None:
        y_t, y_p = mejores_predicciones
        info_umbral = buscar_mejor_umbral(y_t, y_p)
        print(f"\n--- Threshold óptimo (Youden) ---")
        print(f"Umbral:          {info_umbral['umbral']:.4f}")
        print(f"Sensibilidad:    {info_umbral['sensibilidad']:.4f}")
        print(f"Especificidad:   {info_umbral['especificidad']:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": mejor_modelo,
            "best_score": mejor_score,
            "best_epoch": mejor_epoch,
            "umbral_optimo_youden": info_umbral,
            "arquitectura": "smp.Unet(resnet34, imagenet)",
            "image_size": 256,
            "hyperparams": {
                "pos_weight": POS_WEIGHT,
                "target_positive_ratio": TARGET_POSITIVE_RATIO,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "lr_decoder": LR,
                "lr_encoder": LR * LR_ENCODER_FACTOR,
                "epochs_trained": len(historial),
            },
            "historial": historial,
        },
        MODEL_PATH,
    )

    print(f"\nModelo v4 guardado en: {MODEL_PATH}")
    print("Compara con: python src/python/17_comparar_ct_todas_versiones.py")


if __name__ == "__main__":
    entrenar()

"""
Entrenamiento del Modelo 2 (CT) — versión 4.

Diferencias clave respecto a v1-v3
===================================
v1-v3 usaban una UNet pequeña entrenada desde cero sobre 192×192. Todas
se estancaron en AUC 0.68-0.73 y Dice positivos < 0.25.

v4 ataca los tres cuellos de botella reales:

1. **Encoder preentrenado ResNet34 (ImageNet)**
   Los pesos iniciales del encoder ya saben extraer features visuales
   básicas (bordes, texturas, gradientes). Esto es oro con datasets
   pequeños como el nuestro (~1000 slices positivos).

2. **Resolución 256×256**
   Los tumores colorrectales son lesiones pequeñas y sutiles. Con 192×192
   se pierde detalle. 256 es el sweet spot en memoria/detalle para GPU
   de gama media.

3. **Pérdida compuesta: 0.5·BCE + 0.5·Tversky (α=0.3, β=0.7)**
   - Tversky con β=0.7 penaliza FN más que FP (orientación a sensibilidad
     pero controlada, no como pos_weight=6 de v1).
   - BCE mantiene estabilidad de gradientes.
   - Sumadas con pesos iguales: el modelo aprende a segmentar bien Y a
     clasificar bien sin colapsar en ninguna dirección.

Otros cambios
=============
- Scheduler ReduceLROnPlateau (factor=0.5, paciencia=4) para que el LR
  baje automáticamente cuando deja de mejorar.
- Criterio de selección: AUC_val * 0.5 + Dice_pos_val * 0.5.
- 50 epochs máximo con paciencia 12 (más margen por arranque con pesos
  preentrenados que adaptan en pocos epochs).
- pos_weight=3.0 y sampler ratio=0.40 (v3 funcionaba en esa zona).

Dependencias nuevas
===================
- segmentation-models-pytorch (SMP)
  pip install segmentation-models-pytorch
"""