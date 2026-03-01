from __future__ import annotations

from ultralytics import YOLO


def main():
    data_dir = "data/processed/soy_cls"  # deve conter train/ e val/

    # Modelo leve para começar
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data=data_dir,
        imgsz=224,
        epochs=30,
        batch=32,        # se der erro de memória, baixe para 16 ou 8
        device="cpu",    # se tiver GPU NVIDIA, troque para 0
        workers=4,
        project="models",
        name="soy_cls_yolov8s",
        pretrained=True,
    )

    print("Treino finalizado. Pesos e logs em models/soy_cls_yolov8s/")


if __name__ == "__main__":
    main()