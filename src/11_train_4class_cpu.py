from __future__ import annotations
from ultralytics import YOLO

def main():
    data_dir = "data/processed/soy_4class"

    model = YOLO("yolov8s-cls.pt")

    model.train(
        data=data_dir,
        imgsz=224,
        epochs=20,
        batch=8,
        device="cpu",
        workers=2,
        project="models",
        name="soy_4class_yolov8s",
        pretrained=True,
    )

    print("OK: models/soy_4class_yolov8s/")

if __name__ == "__main__":
    main()