from __future__ import annotations

import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASSES = [
    "ferrugem_asiatica",
    "mancha_alvo",
    "dfc_septoria",
    "dfc_cercospora",
]

def list_images(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    random.seed(42)

    src_root = Path("data/external_curated")
    assert src_root.exists(), f"Crie a pasta: {src_root}"

    out_root = Path("data/processed/soy_4class")
    if out_root.exists():
        print(f"ATENÇÃO: {out_root} já existe. Apague manualmente se quiser recriar.")
        return

    # valida classes
    for c in CLASSES:
        cdir = src_root / c
        assert cdir.exists(), f"Falta pasta da classe: {cdir}"
        imgs = list_images(cdir)
        assert len(imgs) >= 20, f"Classe '{c}' tem poucas imagens ({len(imgs)}). Recomendo 100+."

    # cria estrutura
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            ensure_dir(out_root / split / c)

    # split por classe e copia
    total = 0
    for c in CLASSES:
        imgs = list_images(src_root / c)
        train_imgs, test_imgs = train_test_split(imgs, test_size=0.10, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.22, random_state=42)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dst = out_root / split_name / c
            for img in split_imgs:
                shutil.copy2(img, dst / img.name)
                total += 1

        print(f"{c}: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    print(f"\nOK: dataset pronto em {out_root}")
    print(f"Total copiado: {total}")

if __name__ == "__main__":
    main()