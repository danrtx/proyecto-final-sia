"""Prepara el dataset: mapea las 12 clases originales a 6, balancea,
redimensiona y divide en train/val/test (70/15/15)."""
import random
import shutil
from collections import defaultdict

from PIL import Image

from config import DATA_PROC, DATA_RAW, MAPEO_DATASET, MAX_POR_CLASE

random.seed(42)


def main():
    if DATA_PROC.exists():
        shutil.rmtree(DATA_PROC)

    por_clase = defaultdict(list)
    for carpeta, clase in MAPEO_DATASET.items():
        por_clase[clase] += sorted((DATA_RAW / carpeta).glob("*.*"))

    for clase, archivos in por_clase.items():
        random.shuffle(archivos)
        archivos = archivos[:MAX_POR_CLASE] if MAX_POR_CLASE else archivos
        n = len(archivos)
        cortes = {"train": archivos[: int(n * 0.70)],
                  "val": archivos[int(n * 0.70): int(n * 0.85)],
                  "test": archivos[int(n * 0.85):]}
        for split, lista in cortes.items():
            destino = DATA_PROC / split / clase
            destino.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(lista):
                try:
                    img = Image.open(f).convert("RGB")
                    img.thumbnail((320, 320))
                    img.save(destino / f"{clase}_{i:04d}.jpg", quality=92)
                except Exception as e:  # imágenes corruptas
                    print("omitida", f, e)
        print(f"{clase:10s} total={n:4d}  " + "  ".join(f"{k}={len(v)}" for k, v in cortes.items()))


if __name__ == "__main__":
    main()
