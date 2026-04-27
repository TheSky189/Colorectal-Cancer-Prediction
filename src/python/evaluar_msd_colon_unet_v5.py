import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)

# Importar Dataset del script de entrenamiento
sys.path.insert(0, str(Path(__file__).parent))
from entrenar_msd_colon_unet_v5 import MSDColonDataset, CONFIG, DEVICE


def load_model_v5(checkpoint_path):
    """Carga el modelo v5 desde checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get('config', CONFIG)

    model = smp.Unet(
        encoder_name=cfg['encoder'],
        encoder_weights=None,  # Ya cargamos pesos del checkpoint
        in_channels=cfg['in_channels'],
        classes=cfg['classes'],
        activation=None,
    ).to(DEVICE)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg, ckpt.get('metrics', {})


@torch.no_grad()
def predict_all(model, loader):
    """Devuelve (probs, targets) flatten + dice por slice."""
    all_probs   = []
    all_targets = []
    dice_per_slice = []

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        probs  = torch.sigmoid(logits)

        for b in range(images.shape[0]):
            target = masks[b, 0].numpy()
            if target.sum() > 0:
                pred = (probs[b, 0].cpu().numpy() > 0.5).astype(np.float32)
                inter = (pred * target).sum()
                denom = pred.sum() + target.sum()
                if denom > 0:
                    dice_per_slice.append(2 * inter / denom)

        all_probs.append(probs.cpu().numpy().ravel())
        all_targets.append(masks.numpy().ravel())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_targets),
        np.array(dice_per_slice),
    )


def find_youden_threshold(targets, probs):
    """Umbral que maximiza sensibilidad + especificidad - 1."""
    fpr, tpr, thresholds = roc_curve(targets, probs)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return float(thresholds[idx]), float(tpr[idx]), float(1 - fpr[idx])


def evaluate(model_path, name='v5'):
    print(f"\n{'='*70}")
    print(f"Evaluación del modelo {name}: {model_path}")
    print(f"{'='*70}\n")

    if not Path(model_path).exists():
        print(f"[WARN] No existe: {model_path}")
        return None

    model, cfg, train_metrics = load_model_v5(model_path)

    # Dataset de validación
    data_dir = Path(cfg['data_dir'])
    val_files = sorted((data_dir / 'val').glob('*.npz'))
    print(f"[INFO] Slices de validación: {len(val_files)}")

    val_ds = MSDColonDataset(val_files, augment=False, config=cfg)
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Predicciones
    print("[INFO] Calculando predicciones...")
    probs, targets, dice_array = predict_all(model, val_loader)

    # Métricas globales
    auc = roc_auc_score(targets, probs)
    ap  = average_precision_score(targets, probs)

    # Umbral óptimo
    thr_opt, sens_opt, spec_opt = find_youden_threshold(targets, probs)

    # Dice
    dice_mean = float(dice_array.mean()) if len(dice_array) > 0 else 0.0
    dice_median = float(np.median(dice_array)) if len(dice_array) > 0 else 0.0

    results = {
        'model':         name,
        'checkpoint':    str(model_path),
        'auc':           float(auc),
        'ap':            float(ap),
        'dice_mean':     dice_mean,
        'dice_median':   dice_median,
        'n_pos_slices':  int(len(dice_array)),
        'threshold_opt': thr_opt,
        'sens_opt':      sens_opt,
        'spec_opt':      spec_opt,
    }

    print(f"AUC                       : {auc:.4f}")
    print(f"Average Precision (AP)    : {ap:.4f}")
    print(f"Dice (media slices+)      : {dice_mean:.4f}")
    print(f"Dice (mediana slices+)    : {dice_median:.4f}")
    print(f"Slices+ evaluadas         : {len(dice_array)}")
    print(f"Umbral óptimo (Youden)    : {thr_opt:.4f}")
    print(f"Sensibilidad @ umbral opt : {sens_opt:.4f}")
    print(f"Especificidad @ umbral opt: {spec_opt:.4f}")

    return results


def main():
    out_dir = Path('reports/ct_v5_evaluation')
    out_dir.mkdir(parents=True, exist_ok=True)

    results_all = {}

    # Evaluar v5
    res_v5 = evaluate('models/msd_colon_unet_v5_best.pth', name='v5')
    if res_v5:
        results_all['v5'] = res_v5

    # Si existe v4, comparar
    v4_path = Path('models/msd_colon_unet_cached_best.pth')
    if v4_path.exists():
        try:
            res_v4 = evaluate(v4_path, name='v4_referencia')
            if res_v4:
                results_all['v4'] = res_v4
        except Exception as e:
            print(f"[WARN] No se pudo evaluar v4: {e}")

    # Guardar
    output_json = out_dir / 'metricas_v5.json'
    with open(output_json, 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f"\n[DONE] Resultados guardados en: {output_json}")

    # Imprimir comparación si hay ambas
    if 'v4' in results_all and 'v5' in results_all:
        v4 = results_all['v4']
        v5 = results_all['v5']
        print("\n" + "=" * 70)
        print("COMPARACIÓN v4 vs v5")
        print("=" * 70)
        print(f"{'Métrica':<25} {'v4':>10} {'v5':>10} {'Δ':>10}")
        print("-" * 60)
        for key in ['auc', 'ap', 'dice_mean', 'sens_opt', 'spec_opt']:
            d = v5[key] - v4[key]
            arrow = '↑' if d > 0 else '↓' if d < 0 else '='
            print(f"{key:<25} {v4[key]:>10.4f} {v5[key]:>10.4f} {d:>+10.4f} {arrow}")


if __name__ == '__main__':
    main()

"""
================================================================================
evaluar_msd_colon_unet_v5.py
================================================================================
Evaluación del modelo CT v5 sobre el conjunto de validación.

GENERA:
  - Métricas globales: AUC, AP, Dice, Sensibilidad, Especificidad
  - Búsqueda del umbral óptimo por índice de Youden
  - Comparación lado a lado con v4 si existe el modelo anterior
  - Guarda predicciones de ejemplo para análisis visual

USO:
  python src/python/evaluar_msd_colon_unet_v5.py
================================================================================
"""