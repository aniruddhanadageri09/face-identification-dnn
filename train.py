import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import FaceDataset
from src.model import FaceIDModel


def train_model(dataset_dir: str = "data/train", checkpoint_path: str = "checkpoints/face_id_model.pth", epochs: int = 20, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    identities = sorted([p.name for p in Path(dataset_dir).iterdir() if p.is_dir()])
    if not identities:
        raise ValueError(f"No identity folders found in {dataset_dir}")

    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FaceDataset(dataset_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(identities)
    model = FaceIDModel(num_classes=num_classes, embedding_dim=512).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} => loss: {avg_loss:.4f}")

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    return model


if __name__ == "__main__":
    train_model()
