from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

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
    out_dir = Path("data/phenotypes/debug_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(raw_dir)
    assert len(imgs) > 0, "Coloque fotos em data/raw/"

    # Só as primeiras 8 para não lotar
    for img_path in imgs[:8]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # NOVO: leaf_mask por verde (ajustável)
        green = ((H >= 25) & (H <= 95) & (S >= 30) & (V >= 30)).astype(np.uint8)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        leaf_mask = cv2.morphologyEx(green, cv2.MORPH_OPEN, k, iterations=1)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k, iterations=2)
        leaf_mask = largest_component(leaf_mask)

        leaf_area = int(np.count_nonzero(leaf_mask))

        # lesão: amarelo/marrom + escuro, dentro da folha
        yellow_brown = ((H >= 10) & (H <= 45) & (S >= 60) & (V >= 40)).astype(np.uint8)
        dark = ((V <= 80) & (S >= 40)).astype(np.uint8)
        lesion_mask = ((yellow_brown | dark) & (leaf_mask == 1)).astype(np.uint8)

        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k2, iterations=1)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k2, iterations=1)

        lesion_area = int(np.count_nonzero(lesion_mask))
        severity = 0.0 if leaf_area == 0 else 100.0 * (lesion_area / leaf_area)

        # salvar máscaras e overlay
        leaf_vis = (leaf_mask * 255).astype(np.uint8)
        lesion_vis = (lesion_mask * 255).astype(np.uint8)

        overlay = img.copy()
        # pinta lesão de vermelho
        overlay[lesion_mask == 1] = (0, 0, 255)

        # texto na imagem
        cv2.putText(overlay, f"leaf={leaf_area} lesion={lesion_area} sev={severity:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        stem = img_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_leafmask.png"), leaf_vis)
        cv2.imwrite(str(out_dir / f"{stem}_lesionmask.png"), lesion_vis)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    print(f"OK: debug salvo em {out_dir}")

if __name__ == "__main__":
    main()