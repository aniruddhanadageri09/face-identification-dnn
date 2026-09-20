# Face Identification DNN

This repository contains a consent-based research/demo system for identifying an enrolled person from a face image. It uses a ResNet-50 embedding model with ArcFace training and cosine similarity matching.

## Setup

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Put consented face images in one folder per identity:

```text
data/train/
├── alice/
│   ├── image1.jpg
│   └── image2.jpg
└── bob/
    ├── image1.jpg
    └── image2.jpg
```

Put a query image at `data/test/query.jpg`. Use several varied images per identity; the dataset loader automatically assigns contiguous labels, regardless of folder names.

## Train

```bash
python train.py
```

For a quick smoke test, edit `train.py` to use fewer epochs and a smaller batch size. Training on CPU with ResNet-50 can be slow.

## Identify one image

```bash
python inference.py --query data/test/query.jpg --threshold 0.60
```

The output is an enrolled identity or `Unknown`. Calibrate the threshold on a held-out validation set before relying on results.

## Webcam mode

```bash
python webcam.py --camera 0 --threshold 0.60
```

Press `q` to stop. The webcam path uses OpenCV's bundled Haar cascade for a lightweight demo. For higher accuracy, replace it with RetinaFace or SCRFD.

## Flask API

Start the API:

```bash
python app.py
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Identify an uploaded image:

```bash
curl -X POST -F "image=@data/test/query.jpg" http://127.0.0.1:8000/identify
```

Example response:

```json
{"identity":"alice","similarity":0.81}
```

## Important limitations

- The repository does not include a trained checkpoint or images; those must be supplied by the user.
- Images should contain a single, reasonably clear face. The webcam/API detects and crops the largest face.
- This is not a production biometric system. Obtain consent, protect embeddings, restrict access, and evaluate false matches and demographic performance before any deployment.
- Do not use the output as the sole basis for decisions about people.
