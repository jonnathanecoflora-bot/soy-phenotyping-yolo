from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    random.seed(42)

    external = Path("data/external")
    assert external.exists(), f"Não achei: {external}"

    out_root = Path("data/processed/soy_cls")
    if out_root.exists():
        # segurança: não destruir nada sem querer
        print(f"ATENÇÃO: {out_root} já existe. Apague manualmente se quiser recriar.")
        return

    class_dirs = [d for d in external.iterdir() if d.is_dir()]
    assert len(class_dirs) > 1, "Não encontrei pastas de classes em data/external/"

    # Ignorar pastas que não sejam classes (se existirem)
    ignore_names = {"archive", "archive.zip", "__MACOSX"}
    class_dirs = [d for d in class_dirs if d.name not in ignore_names]

    print("Classes encontradas:")
    for d in class_dirs:
        print(" -", d.name)

    # cria estrutura
    for split in ["train", "val", "test"]:
        for cd in class_dirs:
            ensure_dir(out_root / split / cd.name)

    # split por classe
    total_copied = 0
    for cd in class_dirs:
        imgs = list_images(cd)
        if len(imgs) < 10:
            print(f"Pulando classe {cd.name} (poucas imagens: {len(imgs)})")
            continue

        # 10% test, 20% val, 70% train (aprox)
        train_imgs, test_imgs = train_test_split(imgs, test_size=0.10, random_state=42)
        train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.22, random_state=42)

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dst_dir = out_root / split_name / cd.name
            for img in split_imgs:
                shutil.copy2(img, dst_dir / img.name)
                total_copied += 1

        print(f"{cd.name}: train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    print(f"\nOK. Dataset criado em: {out_root}")
    print(f"Total de imagens copiadas: {total_copied}")


if __name__ == "__main__":
    main()