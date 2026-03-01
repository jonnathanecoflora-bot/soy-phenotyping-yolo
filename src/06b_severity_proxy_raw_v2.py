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

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_i = 1 + int(np.argmax(areas))
    return (labels == max_i).astype(np.uint8)

def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove componentes menores que min_area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out

def main():
    raw_dir = Path("data/raw")
    assert raw_dir.exists(), f"Não achei: {raw_dir}"
    imgs = list_images(raw_dir)
    assert len(imgs) > 0, "Coloque fotos em data/raw/ antes de rodar."

    out_dir = Path("data/phenotypes")
    out_dir.mkdir(parents=True, exist_ok=True)

    # pasta para auditoria visual (opcional)
    debug_dir = out_dir / "debug_masks_v2"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for img_path in tqdm(imgs, desc="Severity-proxy v2 (raw)"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 1) Leaf mask (verde) - mais robusto para fotos reais
        green = ((H >= 25) & (H <= 95) & (S >= 25) & (V >= 25)).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        leaf_mask = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k, iterations=2)
        leaf_mask = keep_largest_component(leaf_mask)

        leaf_area = int(np.count_nonzero(leaf_mask))
        if leaf_area < 5000:
            rows.append({
                "image": img_path.name,
                "leaf_area_px": leaf_area,
                "lesion_area_px": 0,
                "severity_proxy_pct": 0.0,
            })
            continue

        # 2) Lesion mask (mais conservador)
        # Ferrugem/lesões: tende a ser amarelado/marrom com saturação alta ou pontos escuros localizados.
        yellow_brown = ((H >= 8) & (H <= 40) & (S >= 80) & (V >= 50)).astype(np.uint8)
        dark_spots   = ((V <= 70) & (S >= 60)).astype(np.uint8)

        lesion = ((yellow_brown | dark_spots) & (leaf_mask == 1)).astype(np.uint8)

        # 3) Limpeza morfológica (remove “chuvisco”)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k2, iterations=1)
        lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, k2, iterations=1)

        # 4) Remove componentes pequenos
        # Ajuste min_area: se suas lesões são pequenas, use 30–80. Se pegar muito ruído, suba para 150–300.
        lesion = remove_small_components(lesion, min_area=120)

        lesion_area = int(np.count_nonzero(lesion))
        severity = 100.0 * (lesion_area / leaf_area)

        rows.append({
            "image": img_path.name,
            "leaf_area_px": leaf_area,
            "lesion_area_px": lesion_area,
            "severity_proxy_pct": round(float(severity), 4),
        })

        # Auditoria: salva overlay para as primeiras 10 imagens
        # (ajuda a validar se está marcando lesão certo)
        if len(rows) <= 10:
            overlay = img.copy()
            overlay[lesion == 1] = (0, 0, 255)
            cv2.putText(overlay, f"sev={severity:.1f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_overlay.png"), overlay)

    df = pd.DataFrame(rows)
    out_csv = out_dir / "soy_raw_severity_proxy_v2.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8", sep=";")  # separador ; para Excel PT-BR
    print(f"OK: {out_csv}")
    print(f"Overlays (amostra) em: {debug_dir}")

if __name__ == "__main__":
    main()