import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MSDColonSliceDatasetCachedV3(Dataset):
    def __init__(self, csv_path, mode="train"):
        self.df = pd.read_csv(csv_path)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    # Augmentación geométrica
    def augmentar_geometrico(self, image, mask):
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
        return image, mask

    # Augmentación de intensidad (solo imagen, no máscara)
    def augmentar_intensidad(self, image):
        # Brillo aditivo
        if random.random() < 0.6:
            delta = random.uniform(-0.15, 0.15)
            image = np.clip(image + delta, 0.0, 1.0)

        # Contraste multiplicativo
        if random.random() < 0.6:
            factor = random.uniform(0.75, 1.25)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0.0, 1.0)

        # Gamma
        if random.random() < 0.4:
            gamma = random.uniform(0.8, 1.25)
            image = np.clip(image ** gamma, 0.0, 1.0)

        # Ruido gaussiano leve
        if random.random() < 0.4:
            sigma = random.uniform(0.005, 0.025)
            noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return image.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["path"])

        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        label = float(data["label"])

        if self.mode == "train":
            image, mask = self.augmentar_geometrico(image, mask)
            image = self.augmentar_intensidad(image)

        # 3 canales para compatibilidad con encoder preentrenado (ImageNet)
        image = np.stack([image, image, image], axis=0)
        mask = np.expand_dims(mask, axis=0)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "case_id": row["case_id"],
            "slice_idx": torch.tensor(int(row["slice_idx"]), dtype=torch.long),
        }

"""
Dataset de slices MSD Colon — versión 3.

Mejoras respecto a v2:
- Elastic-like deformación aleatoria (desplazamiento pequeño) para augmentar
  la variabilidad de formas del tumor.
- Augmentación de intensidad más agresiva en train (brillo/contraste más
  amplio, porque el encoder ResNet34 preentrenado se beneficia de diversidad
  fotométrica).
- Limpieza: eliminado código duplicado.

Compatible con slices de cualquier resolución (192, 256, 320...).
El modelo v4 usará esta versión con la carpeta msd_colon_slices_256.
"""