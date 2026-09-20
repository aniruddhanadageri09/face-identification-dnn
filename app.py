from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request

from inference import identify_array, load_recognizer
from src.face_detection import FaceDetector


def create_app(checkpoint: str = "checkpoints/face_id_model.pth", train_dir: str = "data/train", threshold: float = 0.60):
    app = Flask(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = load_recognizer(checkpoint, train_dir, device)
    detector = FaceDetector()

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "device": str(device)})

    @app.post("/identify")
    def identify_endpoint():
        uploaded = request.files.get("image")
        if uploaded is None:
            return jsonify({"error": "Upload an image using the 'image' form field"}), 400

        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image file"}), 400

        try:
            face = detector.crop_largest(image)
            identity, score = identify_array(face, recognizer, threshold)
            return jsonify({"identity": identity, "similarity": round(score, 6)})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)
