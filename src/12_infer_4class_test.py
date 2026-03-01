from __future__ import annotations
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def main():
    weights = Path("models/soy_4class_yolov8s/weights/best.pt")
    assert weights.exists(), f"Não achei: {weights}"

    model = YOLO(str(weights))

    test_dir = Path("data/processed/soy_4class/test")
    imgs = list_images(test_dir)
    assert len(imgs) > 0, "Sem imagens no test."

    rows = []
    for p in imgs:
        res = model.predict(str(p), verbose=False)[0]
        probs = res.probs
        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf)
        pred = res.names[top_idx]
        true = p.parent.name

        rows.append({
            "image": p.name,
            "true_label": true,
            "pred_label": pred,
            "pred_conf": round(top_conf, 6),
            "is_correct": int(pred == true),
        })

    df = pd.DataFrame(rows)
    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "soy_4class_predictions_test.csv"
    df.to_csv(out_csv, index=False, sep=";", decimal=",", encoding="utf-8")

    acc = df["is_correct"].mean() if len(df) else 0.0
    print(f"OK: {out_csv}")
    print(f"Acurácia (test): {acc:.4f}")

if __name__ == "__main__":
    main()