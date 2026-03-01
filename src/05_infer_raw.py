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

    raw_dir = Path("data/raw")
    assert raw_dir.exists(), f"Não achei: {raw_dir}"

    imgs = list_images(raw_dir)
    assert len(imgs) > 0, "Coloque fotos em data/raw/ antes de rodar."

    model = YOLO(str(weights))

    rows = []
    for img_path in imgs:
        res = model.predict(str(img_path), verbose=False)[0]
        probs = res.probs
        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf)
        pred_name = res.names[top_idx]

        rows.append({
            "image": img_path.name,
            "path": str(img_path).replace("\\", "/"),
            "pred_label": pred_name,
            "pred_conf": round(top_conf, 6),
        })

    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "soy_raw_predictions.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"OK: {out_csv}")

if __name__ == "__main__":
    main()