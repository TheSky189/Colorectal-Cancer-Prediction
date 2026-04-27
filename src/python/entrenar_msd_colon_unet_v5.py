import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, average_precision_score

# CONFIGURACIÓN GLOBAL
CONFIG = {
    # Datos
    'data_dir':         'data/processed/msd_colon_slices_320',
    'image_size':       320,
    'batch_size':       8,             # Reducir si OOM en GPU pequeña
    'num_workers':      4,

    # Arquitectura
    'encoder':          'efficientnet-b0',
    'encoder_weights':  'imagenet',
    'in_channels':      3,
    'classes':          1,

    # Entrenamiento
    'epochs':           80,
    'lr_encoder':       5e-5,          # Más bajo que v4 (preserva preentrenado)
    'lr_decoder':       5e-4,
    'weight_decay':     1e-4,
    'warmup_epochs':    5,
    'patience':         15,            # Early stopping

    # Loss
    'tversky_alpha':    0.5,           # Más simétrico que v4 (0.3)
    'tversky_beta':     0.5,
    'focal_gamma':      1.5,
    'bce_weight':       0.3,           # Reducido vs v4 (0.5)
    'tversky_weight':   0.7,

    # Augmentation
    'aug_prob':         0.7,
    'rot_max_deg':      30,
    'noise_std':        0.02,

    # Optimización
    'use_amp':          True,          # Mixed precision (1.5-2x speedup)
    'grad_clip':        1.0,

    # Output
    'output_dir':       'models',
    'log_dir':          'reports/ct_v5_logs',
    'save_best_only':   True,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# DATASET CON AUGMENTATION AGRESIVO
class MSDColonDataset(Dataset):
    """Dataset de slices CT preprocesadas (320x320, normalizadas, 3 canales)."""

    def __init__(self, file_list, augment=False, config=None):
        self.files = file_list
        self.augment = augment
        self.cfg = config or CONFIG

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        data = np.load(npz_path)
        image = data['image'].astype(np.float32)  # (320, 320)
        mask  = data['mask'].astype(np.float32)   # (320, 320)

        # Replicar canal único en 3 canales
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=0)  # (3, 320, 320)

        # Augmentation
        if self.augment and np.random.rand() < self.cfg['aug_prob']:
            image, mask = self._augment(image, mask)

        return torch.from_numpy(image), torch.from_numpy(mask).unsqueeze(0)

    def _augment(self, image, mask):
        """Augmentation agresivo dirigido a problemas observados."""
        # 1. Flip horizontal
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, :, ::-1])
            mask  = np.ascontiguousarray(mask[:, ::-1])

        # 2. Flip vertical (anatómicamente raro pero ayuda)
        if np.random.rand() < 0.3:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask  = np.ascontiguousarray(mask[::-1, :])

        # 3. Rotación aleatoria pequeña
        if np.random.rand() < 0.5:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-self.cfg['rot_max_deg'], self.cfg['rot_max_deg'])
            image = np.stack([
                rotate(image[c], angle, reshape=False, order=1, mode='constant', cval=0.0)
                for c in range(image.shape[0])
            ])
            mask = rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0.0)

        # 4. Cambio de contraste (simula distintos protocolos CT)
        if np.random.rand() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = (image - mean) * factor + mean
            image = np.clip(image, 0.0, 1.0)

        # 5. Brillo
        if np.random.rand() < 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            image = np.clip(image + shift, 0.0, 1.0)

        # 6. Ruido gaussiano (simula distintos equipos CT)
        if np.random.rand() < 0.4:
            noise = np.random.randn(*image.shape).astype(np.float32) * self.cfg['noise_std']
            image = np.clip(image + noise, 0.0, 1.0)

        return image, mask


# LOSS: FOCAL TVERSKY
class FocalTverskyLoss(nn.Module):
    """
    Tversky con factor focal para enfocar en ejemplos difíciles.
    Combina mejor que BCE+Tversky cuando hay falsos positivos dispersos.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky


class CombinedLoss(nn.Module):
    """BCE + Focal Tversky con pesos configurables."""
    def __init__(self, bce_weight, tversky_weight, alpha, beta, gamma):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ftv = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_w = bce_weight
        self.tv_w  = tversky_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.tv_w * self.ftv(logits, targets)


# MÉTRICAS
@torch.no_grad()
def compute_metrics(model, loader, device):
    """Calcula AUC, AP, Dice (en píxeles positivos) sobre todo el loader."""
    model.eval()

    all_probs   = []
    all_targets = []
    dice_per_slice = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with autocast(enabled=CONFIG['use_amp']):
            logits = model(images)
            probs  = torch.sigmoid(logits)

        # Dice por slice (solo slices con tumor real)
        for b in range(images.shape[0]):
            target = masks[b, 0].cpu().numpy()
            if target.sum() == 0:
                continue
            pred = (probs[b, 0].cpu().numpy() > 0.5).astype(np.float32)
            inter = (pred * target).sum()
            denom = pred.sum() + target.sum()
            if denom > 0:
                dice_per_slice.append(2 * inter / denom)

        all_probs.append(probs.cpu().numpy().ravel())
        all_targets.append(masks.cpu().numpy().ravel())

    all_probs   = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    auc = roc_auc_score(all_targets, all_probs) if all_targets.sum() > 0 else 0.0
    ap  = average_precision_score(all_targets, all_probs) if all_targets.sum() > 0 else 0.0
    dice = np.mean(dice_per_slice) if dice_per_slice else 0.0

    return {'auc': auc, 'ap': ap, 'dice_pos': dice, 'n_pos_slices': len(dice_per_slice)}


# SCHEDULER: WARMUP + COSINE ANNEALING
def warmup_cosine_schedule(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


# ENTRENAMIENTO
def train():
    # Cargar listas de archivos 
    data_dir = Path(CONFIG['data_dir'])
    if not data_dir.exists():
        print(f"[ERROR] No existe {data_dir}")
        print("[ERROR] Primero ejecuta: python src/python/preprocesar_msd_colon_slices_320.py")
        sys.exit(1)

    train_files = sorted((data_dir / 'train').glob('*.npz'))
    val_files   = sorted((data_dir / 'val').glob('*.npz'))

    print(f"[INFO] Train slices: {len(train_files)}")
    print(f"[INFO] Val slices:   {len(val_files)}")

    if len(train_files) == 0:
        print("[ERROR] No hay archivos de entrenamiento")
        sys.exit(1)

    train_ds = MSDColonDataset(train_files, augment=True,  config=CONFIG)
    val_ds   = MSDColonDataset(val_files,   augment=False, config=CONFIG)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True,
    )

    # Modelo
    print(f"[INFO] Construyendo modelo: U-Net + {CONFIG['encoder']}")
    model = smp.Unet(
        encoder_name=CONFIG['encoder'],
        encoder_weights=CONFIG['encoder_weights'],
        in_channels=CONFIG['in_channels'],
        classes=CONFIG['classes'],
        activation=None,  # Sigmoid se aplica en el loss
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parámetros totales: {n_params:,}")

    # Optimizer (LR diferenciado)
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if not n.startswith('encoder')]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': CONFIG['lr_encoder']},
        {'params': decoder_params, 'lr': CONFIG['lr_decoder']},
    ], weight_decay=CONFIG['weight_decay'])

    # Loss 
    criterion = CombinedLoss(
        bce_weight=CONFIG['bce_weight'],
        tversky_weight=CONFIG['tversky_weight'],
        alpha=CONFIG['tversky_alpha'],
        beta=CONFIG['tversky_beta'],
        gamma=CONFIG['focal_gamma'],
    ).to(DEVICE)

    # AMP scaler 
    scaler = GradScaler(enabled=CONFIG['use_amp'])

    # Logging 
    Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = Path(CONFIG['log_dir']) / f'training_log_{timestamp}.json'

    history = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                   for k, v in CONFIG.items()},
        'epochs': [],
    }

    # Loop de entrenamiento 
    best_dice = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n[INFO] Iniciando entrenamiento ({CONFIG['epochs']} épocas)\n")
    t_start = time.time()

    for epoch in range(CONFIG['epochs']):
        # Aplicar scheduler
        lr_factor = warmup_cosine_schedule(epoch, CONFIG['warmup_epochs'], CONFIG['epochs'])
        for i, pg in enumerate(optimizer.param_groups):
            base_lr = CONFIG['lr_encoder'] if i == 0 else CONFIG['lr_decoder']
            pg['lr'] = base_lr * lr_factor

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=CONFIG['use_amp']):
                logits = model(images)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(1, n_batches)
        train_time = time.time() - t0

        # Validación 
        t0 = time.time()
        val_metrics = compute_metrics(model, val_loader, DEVICE)
        val_time = time.time() - t0

        elapsed = time.time() - t_start
        eta_total = elapsed / (epoch + 1) * CONFIG['epochs']
        eta_remain = eta_total - elapsed

        log_line = (
            f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
            f"loss={avg_loss:.4f} | "
            f"AUC={val_metrics['auc']:.4f} | "
            f"AP={val_metrics['ap']:.4f} | "
            f"Dice+={val_metrics['dice_pos']:.4f} | "
            f"lr={optimizer.param_groups[1]['lr']:.2e} | "
            f"t_train={train_time:.0f}s t_val={val_time:.0f}s | "
            f"ETA={eta_remain/60:.0f}min"
        )
        print(log_line)

        history['epochs'].append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_auc':    val_metrics['auc'],
            'val_ap':     val_metrics['ap'],
            'val_dice':   val_metrics['dice_pos'],
            'lr':         optimizer.param_groups[1]['lr'],
            'train_time': train_time,
        })

        # Guardar checkpoint si mejora
        if val_metrics['dice_pos'] > best_dice:
            best_dice = val_metrics['dice_pos']
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            ckpt_path = Path(CONFIG['output_dir']) / 'msd_colon_unet_v5_best.pth'
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'metrics': val_metrics,
            }, ckpt_path)
            print(f"  ★ Mejor modelo guardado: Dice={best_dice:.4f} @ epoch {best_epoch}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CONFIG['patience']:
                print(f"\n[INFO] Early stopping: {CONFIG['patience']} épocas sin mejora")
                break

        # Guardar log cada época
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    # Resumen final 
    total_time = time.time() - t_start
    print(f"\n[DONE] Entrenamiento completado en {total_time/60:.1f} min")
    print(f"[DONE] Mejor Dice+: {best_dice:.4f} en epoch {best_epoch}")
    print(f"[DONE] Modelo guardado en: {CONFIG['output_dir']}/msd_colon_unet_v5_best.pth")
    print(f"[DONE] Log: {log_path}")


if __name__ == '__main__':
    train()


    """
================================================================================
entrenar_msd_colon_unet_v5.py
================================================================================
Entrenamiento del modelo de segmentación de tumores colorrectales en imagen CT.
VERSIÓN 5 — optimización dirigida a corregir los problemas observados en v4:

PROBLEMAS DE v4 (Dice 0.47):
  - Falsos positivos dispersos por bordes vasculares
  - A veces marca todo menos el tumor real
  - Tumor segmentado con forma irregular vs anillo limpio del ground truth

CAMBIOS PRINCIPALES v5:
  1. Encoder: ResNet-34 -> EfficientNet-B0 (mejor relación calidad/parámetros)
  2. Resolución: 256x256 -> 320x320 (sweet spot memoria/detalle)
  3. Loss: BCE+Tversky -> Focal Tversky (gamma=1.5) más simétrico
  4. Augmentation más agresivo (rotaciones, contraste, ruido gaussiano)
  5. Más épocas (80 vs 50) con cosine annealing + warmup
  6. Validación con AP (Average Precision) además de AUC
  7. Mixed precision training (autocast) para acelerar 1.5-2x con GPU

REQUISITOS:
  - GPU NVIDIA con CUDA 11.8+ (8GB VRAM mínimo, 12GB recomendado)
  - PyTorch >= 2.0 + segmentation_models_pytorch >= 0.3
  - Dataset preprocesado en data/processed/msd_colon_slices_320/

USO:
  python src/python/entrenar_msd_colon_unet_v5.py

Tiempo estimado: 2-4h en GPU NVIDIA RTX 3060+ (12GB)
================================================================================
"""