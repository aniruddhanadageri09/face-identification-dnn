import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    """Load images from data/train/<identity>/ and assign contiguous labels."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        self.transform = transform or self._default_transform()
        class_names = sorted(p.name for p in self.root_dir.iterdir() if p.is_dir())
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.samples = self._load_samples()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = self.class_to_index[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((str(image_path), label))
        if not samples:
            raise ValueError(f"No usable images found in dataset: {self.root_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label
