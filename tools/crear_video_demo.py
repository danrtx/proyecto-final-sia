"""Crea un video de demostración tipo "banda transportadora" con imágenes del
conjunto de PRUEBA (nunca vistas en el entrenamiento)."""
import random
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROC, RAIZ  # noqa: E402

random.seed(7)
W, H, FPS, SEG = 1280, 720, 25, 20
VEL = 6  # píxeles por cuadro


def fondo():
    img = np.full((H, W, 3), 55, np.uint8)
    cv2.rectangle(img, (0, 90), (W, H - 90), (75, 75, 80), -1)
    for x in range(0, W, 80):
        cv2.line(img, (x, 90), (x, H - 90), (65, 65, 70), 2)
    return img


def main():
    archivos = sorted((DATA_PROC / "test").glob("*/*.jpg"))
    random.shuffle(archivos)
    objetos, siguiente, k = [], 0, 0
    salida = RAIZ / "outputs" / "video_demo_banda.mp4"
    salida.parent.mkdir(exist_ok=True)
    w = imageio.get_writer(salida, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    base = fondo()
    for f in range(FPS * SEG):
        if f >= siguiente:
            img = cv2.imread(str(archivos[k % len(archivos)])); k += 1
            lado = random.randint(170, 230)
            esc = lado / max(img.shape[:2])
            img = cv2.resize(img, None, fx=esc, fy=esc)
            carriles = [100, 300, 500 - img.shape[0] // 2]
            y = min(random.choice(carriles) + random.randint(-10, 10), H - 95 - img.shape[0])
            objetos.append([-img.shape[1], max(95, y), img])
            siguiente = f + random.randint(18, 30)
        frame = base.copy()
        for o in objetos:
            o[0] += VEL
            x, y, img = o
            h, ww = img.shape[:2]
            x0, x1 = max(0, x), min(W, x + ww)
            if x1 > x0:
                frame[y:y + h, x0:x1] = img[:, x0 - x:x1 - x]
        objetos = [o for o in objetos if o[0] < W]
        w.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w.close()
    print("Video creado:", salida)


if __name__ == "__main__":
    main()
