import os
from pathlib import Path

import torch
from torchvision import transforms

from src.model import FaceIDModel
from src.utils import get_identity_folders, get_transform, load_image


def build_gallery(model, train_dir: str, device):
    gallery = {}
    transform = get_transform()

    for identity in get_identity_folders(train_dir):
        identity_dir = Path(train_dir) / identity
        embeddings = []

        for image_file in sorted(identity_dir.iterdir()):
            if image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            image = load_image(str(image_file))
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(tensor)
                embedding = embedding / embedding.norm(dim=1, keepdim=True)
                embeddings.append(embedding.squeeze(0).cpu())

        if embeddings:
            gallery[identity] = torch.stack(embeddings)

    return gallery


def identify(query_path: str, model_path: str = "checkpoints/face_id_model.pth", train_dir: str = "data/train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    identities = get_identity_folders(train_dir)
    model = FaceIDModel(num_classes=len(identities), embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gallery = build_gallery(model, train_dir, device)

    transform = get_transform()
    image = load_image(query_path)
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(x)
        query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
        query_embedding = query_embedding.squeeze(0).cpu()

    best_identity = "Unknown"
    best_score = -1.0

    for identity, embeddings in gallery.items():
        similarities = embeddings @ query_embedding
        top_score = float(similarities.max())
        if top_score > best_score:
            best_score = top_score
            best_identity = identity

    threshold = 0.6
    if best_score < threshold:
        return "Unknown", best_score
    return best_identity, best_score


if __name__ == "__main__":
    query = "data/test/query.jpg"
    identity, score = identify(query)
    print(f"Predicted identity: {identity}")
    print(f"Similarity score: {score:.4f}")
