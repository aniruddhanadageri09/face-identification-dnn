# Face Identification DNN

This project builds a face identification system using a convolutional neural network with ArcFace loss. The model learns person-specific embeddings from facial images and identifies a query face by comparing its embedding to an enrolled gallery.

## Features
- ResNet-50 backbone for face representation learning
- ArcFace margin loss for identity separation
- L2-normalized embedding vectors
- Cosine similarity matching for recognition
- Training and inference scripts
- Folder-based dataset structure for multiple identities

## Project structure

```text
face-identification-dnn/
├── .gitignore
├── requirements.txt
├── README.md
├── train.py
├── inference.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── data/
│   ├── train/
│   │   ├── person_001/
│   │   ├── person_002/
│   │   └── ...
│   └── test/
│       └── query.jpg
└── checkpoints/
    └── face_id_model.pth
```

## Requirements

```bash
pip install -r requirements.txt
```

## Dataset structure

Create the dataset in this format:

```text
data/
├── train/
│   ├── person_001/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── person_002/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── ...
└── test/
    └── query.jpg
```

Each folder under `data/train` represents one identity.

## Training

```bash
python train.py
```

This trains a `ResNet-50 + ArcFace` face recognition model.

## Identification

```bash
python inference.py
```

The script loads the saved model and compares the query face embedding against embeddings from the training identities.

## Model design

```text
Input face image (112x112x3)
         |
         v
ResNet-50 backbone
         |
         v
Embedding head (512-d)
         |
         v
L2 normalization
         |
         v
ArcFace loss
```

## Notes
- This is a closed-set identification system.
- For an open-set deployment, use a similarity threshold to reject unknown persons.
- In real deployments, add face detection and landmark alignment before recognition.
- The current project is a strong baseline for research, learning, and demo use.

## References
- ArcFace: https://arxiv.org/abs/1801.07698
- ResNet: https://arxiv.org/abs/1512.03385

## License
MIT
