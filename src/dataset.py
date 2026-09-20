import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """Dataset that loads a face image per identity folder."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()
        self.samples = self._load_samples()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            label = int(class_dir.name.split("_")[-1]) if class_dir.name.startswith("person_") else len(samples)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((str(img_path), label))

        if not samples:
            raise ValueError(f"No usable images found in dataset: {self.root_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
