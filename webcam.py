import argparse
from pathlib import Path

import cv2
import torch

from inference import identify_array, load_recognizer
from src.face_detection import FaceDetector


def main():
    parser = argparse.ArgumentParser(description="Real-time face identification from a webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--checkpoint", default="checkpoints/face_id_model.pth")
    parser.add_argument("--train-dir", default="data/train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = load_recognizer(args.checkpoint, args.train_dir, device)
    detector = FaceDetector()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    print("Press q to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            for x, y, w, h in detector.detect(frame):
                crop = frame[y:y + h, x:x + w]
                try:
                    identity, score = identify_array(crop, recognizer, args.threshold)
                    label = f"{identity} ({score:.2f})"
                except ValueError as exc:
                    label = str(exc)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Identification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
