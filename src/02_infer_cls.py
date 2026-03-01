from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def main():
    weights = Path("models/soy_cls_yolov8s/weights/best.pt")
    assert weights.exists(), f"Não achei os pesos: {weights}"

    model = YOLO(str(weights))

    # Pode inferir no test set para demonstrar
    source_dir = Path("data/processed/soy_cls/test")
    assert source_dir.exists(), f"Não achei: {source_dir}"

    imgs = list_images(source_dir)
    assert len(imgs) > 0, "Sem imagens no test."

    rows = []
    for img_path in imgs:
        res = model.predict(str(img_path), verbose=False)[0]
        probs = res.probs  # vetor de probabilidades por classe

        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf)
        top_name = res.names[top_idx]

        # classe verdadeira = nome da pasta (para avaliar acerto)
        true_label = img_path.parent.name

        rows.append(
            {
                "image": img_path.name,
                "path": str(img_path).replace("\\", "/"),
                "true_label": true_label,
                "pred_label": top_name,
                "pred_conf": round(top_conf, 6),
                "is_correct": int(top_name == true_label),
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "soy_cls_predictions_test.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    acc = df["is_correct"].mean() if len(df) else 0.0
    print(f"OK: {out_csv}")
    print(f"Acurácia (test): {acc:.4f}")


if __name__ == "__main__":
    main()