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
    # mantém apenas o maior componente conectado (remove ruído)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    # stats[0] é fundo
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_i = 1 + int(np.argmax(areas))
    return (labels == max_i).astype(np.uint8)


def main():
    # vamos calcular severidade no test set (demonstração)
    source_dir = Path("data/processed/soy_cls/test")
    assert source_dir.exists(), f"Não achei: {source_dir}"

    imgs = list_images(source_dir)
    assert len(imgs) > 0, "Sem imagens."

    rows = []
    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(imgs, desc="Severidade-proxy"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 1) Leaf mask (aproximação): separar “objeto” do fundo
        # Funciona bem quando o fundo é mais claro/regular.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove pixels quase brancos (fundo claro comum em datasets)
        bg = gray > 245
        leaf_mask = (~bg).astype(np.uint8)

        # limpa ruído e fecha buracos
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k, iterations=1)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k, iterations=2)

        # mantém só o maior componente (a folha)
        leaf_mask = largest_component(leaf_mask)

        leaf_area = int(np.count_nonzero(leaf_mask))
        if leaf_area < 500:  # imagem problemática
            rows.append(
                {
                    "image": img_path.name,
                    "path": str(img_path).replace("\\", "/"),
                    "true_label": img_path.parent.name,
                    "leaf_area_px": leaf_area,
                    "lesion_area_px": 0,
                    "severity_proxy_pct": 0.0,
                }
            )
            continue

        # 2) Lesion mask (proxy): combina 2 sinais
        # - regiões amareladas/marrons (HSV)
        # - regiões escuras (baixo V) dentro da folha
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # amarelo/marrom (ajuste fino depois se necessário)
        yellow_brown = ((H >= 10) & (H <= 45) & (S >= 60) & (V >= 40)).astype(np.uint8)

        # manchas escuras
        dark = ((V <= 80) & (S >= 40)).astype(np.uint8)

        lesion_mask = ((yellow_brown | dark) & (leaf_mask == 1)).astype(np.uint8)

        # pós-processamento
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k2, iterations=1)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k2, iterations=1)

        lesion_area = int(np.count_nonzero(lesion_mask))
        severity = 100.0 * (lesion_area / leaf_area)

        rows.append(
            {
                "image": img_path.name,
                "path": str(img_path).replace("\\", "/"),
                "true_label": img_path.parent.name,
                "leaf_area_px": leaf_area,
                "lesion_area_px": lesion_area,
                "severity_proxy_pct": round(float(severity), 4),
            }
        )

    df = pd.DataFrame(rows).sort_values(["true_label", "image"])
    out_csv = out_dir / "soy_severity_proxy_test.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"OK: {out_csv}")


if __name__ == "__main__":
    main()