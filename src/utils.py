from pathlib import Path

from PIL import Image
from torchvision import transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_image(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return image


def get_identity_folders(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")
    return sorted([p.name for p in root.iterdir() if p.is_dir()])
