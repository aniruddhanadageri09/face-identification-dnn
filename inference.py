import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import torch
from PIL import Image

from src.model import FaceIDModel
from src.utils import get_identity_folders, get_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_recognizer(model_path: str, train_dir: str, device: torch.device):
    identities = get_identity_folders(train_dir)
    if not identities:
        raise ValueError(f"No identity folders found in {train_dir}")

    model = FaceIDModel(num_classes=len(identities), embedding_dim=512).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    gallery = build_gallery(model, train_dir, device)
    return {"model": model, "gallery": gallery, "device": device}


def image_to_embedding(image, recognizer):
    model = recognizer["model"]
    device = recognizer["device"]
    tensor = get_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
        return torch.nn.functional.normalize(embedding, p=2, dim=1).squeeze(0).cpu()


def build_gallery(model, train_dir: str, device: torch.device) -> Dict[str, torch.Tensor]:
    gallery = {}
    transform = get_transform()
    for identity in get_identity_folders(train_dir):
        identity_dir = Path(train_dir) / identity
        embeddings = []
        for image_file in sorted(identity_dir.iterdir()):
            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                image = Image.open(image_file).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = torch.nn.functional.normalize(model(tensor), p=2, dim=1)
                embeddings.append(embedding.squeeze(0).cpu())
            except (OSError, RuntimeError) as exc:
                print(f"Skipping {image_file}: {exc}")
        if embeddings:
            gallery[identity] = torch.stack(embeddings)
    if not gallery:
        raise ValueError(f"No usable gallery images found in {train_dir}")
    return gallery


def identify_array(image, recognizer, threshold: float = 0.60) -> Tuple[str, float]:
    if image is None or image.size == 0:
        raise ValueError("Empty face image")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    query_embedding = image_to_embedding(Image.fromarray(image_rgb), recognizer)

    best_identity, best_score = "Unknown", -1.0
    for identity, embeddings in recognizer["gallery"].items():
        score = float((embeddings @ query_embedding).max())
        if score > best_score:
            best_identity, best_score = identity, score
    return (best_identity if best_score >= threshold else "Unknown", best_score)


def identify(query_path: str, model_path: str = "checkpoints/face_id_model.pth", train_dir: str = "data/train", threshold: float = 0.60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = load_recognizer(model_path, train_dir, device)
    image = cv2.imread(query_path)
    if image is None:
        raise ValueError(f"Could not read query image: {query_path}")
    return identify_array(image, recognizer, threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="data/test/query.jpg")
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    identity, score = identify(args.query, threshold=args.threshold)
    print(f"Predicted identity: {identity}")
    print(f"Similarity score: {score:.4f}")
