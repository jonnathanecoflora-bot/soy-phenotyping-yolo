from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)

def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_i = 1 + int(np.argmax(areas))
    return (labels == max_i).astype(np.uint8)

def main():
    raw_dir = Path("data/raw")
    assert raw_dir.exists(), f"Não achei: {raw_dir}"

    imgs = list_images(raw_dir)
    assert len(imgs) > 0, "Coloque fotos em data/raw/ antes de rodar."

    rows = []
    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(imgs, desc="Severidade-proxy (raw)"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove pixels muito claros (fundo branco)
        bg = gray > 245
        leaf_mask = (~bg).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k, iterations=1)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k, iterations=2)
        leaf_mask = largest_component(leaf_mask)

        leaf_area = int(np.count_nonzero(leaf_mask))
        if leaf_area < 500:
            rows.append({
                "image": img_path.name,
                "path": str(img_path).replace("\\", "/"),
                "leaf_area_px": leaf_area,
                "lesion_area_px": 0,
                "severity_proxy_pct": 0.0,
            })
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        yellow_brown = ((H >= 10) & (H <= 45) & (S >= 60) & (V >= 40)).astype(np.uint8)
        dark = ((V <= 80) & (S >= 40)).astype(np.uint8)

        lesion_mask = ((yellow_brown | dark) & (leaf_mask == 1)).astype(np.uint8)

        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k2, iterations=1)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k2, iterations=1)

        lesion_area = int(np.count_nonzero(lesion_mask))
        severity = 100.0 * (lesion_area / leaf_area)

        rows.append({
            "image": img_path.name,
            "path": str(img_path).replace("\\", "/"),
            "leaf_area_px": leaf_area,
            "lesion_area_px": lesion_area,
            "severity_proxy_pct": round(float(severity), 4),
        })

    out_csv = out_dir / "soy_raw_severity_proxy.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"OK: {out_csv}")

if __name__ == "__main__":
    main()