from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import CLASSES, IMAGES_DIR


class RefleXDataset(Dataset):
    """Multilabel dataset for the RefleX X-ray diffraction images."""

    def __init__(self, labels_csv: Path, images_dir: Path = IMAGES_DIR, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_ext = self._detect_extension()

    def _detect_extension(self) -> str:
        for ext in (".png", ".jpg", ".jpeg"):
            if (self.images_dir / f"{self.df.iloc[0]['image']}{ext}").exists():
                return ext
        raise FileNotFoundError(
            f"Could not find images for '{self.df.iloc[0]['image']}' in {self.images_dir} "
            "with extension .png/.jpg/.jpeg"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.images_dir / f"{row['image']}{self.image_ext}"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(row[CLASSES].values.astype("float32"))
        return image, labels
