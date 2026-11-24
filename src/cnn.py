"""EcoNet: red neuronal convolucional (CNN) construida y entrenada desde cero.

Arquitectura (entrada 3x128x128):
  Bloque 1: [Conv 3x3 -> BatchNorm -> ReLU] x2, 32 filtros  -> MaxPool  (64x64)
  Bloque 2: [Conv 3x3 -> BatchNorm -> ReLU] x2, 64 filtros  -> MaxPool  (32x32)
  Bloque 3: [Conv 3x3 -> BatchNorm -> ReLU] x2, 128 filtros -> MaxPool  (16x16)
  Bloque 4: [Conv 3x3 -> BatchNorm -> ReLU] x2, 256 filtros -> MaxPool  (8x8)
  Global Average Pooling -> Dropout(0.4) -> Linear(256 -> 6)
Todos los pesos se inicializan aleatoriamente (inicialización de Kaiming).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2 as T

from config import CLASES, TAM_IMAGEN

MEDIA = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def bloque(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class EcoNet(nn.Module):
    def __init__(self, n_clases: int = len(CLASES), canales=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        capas, c = [], 3
        for c_out in canales:
            capas.append(bloque(c, c_out))
            c = c_out
        self.features = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_clases))
        for m in self.modules():  # inicialización aleatoria (NO pre-entrenada)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.clasificador(self.pool(self.features(x)))


# ------------------------------------------------------------- transformaciones
tf_entrenamiento = T.Compose([
    T.ToImage(), T.RandomResizedCrop(TAM_IMAGEN, scale=(0.5, 1.0), antialias=True),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(25),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA, STD),
])
tf_evaluacion = T.Compose([
    T.ToImage(), T.Resize(TAM_IMAGEN, antialias=True), T.CenterCrop(TAM_IMAGEN),
    T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA, STD),
])


class ClasificadorCNN:
    """Envoltura para usar la CNN entrenada en la aplicación."""

    def __init__(self, ruta, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ruta, map_location=self.device)
        self.clases = ckpt["clases"]
        self.modelo = EcoNet(len(self.clases)).to(self.device).eval()
        self.modelo.load_state_dict(ckpt["modelo"])

    @torch.no_grad()
    def predecir(self, imagenes: list[Image.Image]):
        if not imagenes:
            return []
        x = torch.stack([tf_evaluacion(im.convert("RGB")) for im in imagenes]).to(self.device)
        probs = self.modelo(x).softmax(-1).cpu()
        return [(self.clases[int(p.argmax())], float(p.max()), p) for p in probs]
