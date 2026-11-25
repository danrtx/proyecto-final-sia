"""EcoNet: red neuronal convolucional (CNN) construida y entrenada desde cero.

Arquitectura (entrada 3x128x128):
  Bloque 1: [Conv 3x3 -> BatchNorm -> ReLU] x2, 32 filtros  -> MaxPool  (64x64)
  Bloque 2: [Conv 3x3 -> BatchNorm -> ReLU] x2, 64 filtros  -> MaxPool  (32x32)
  Bloque 3: [Conv 3x3 -> BatchNorm -> ReLU] x2, 128 filtros -> MaxPool  (16x16)
  Bloque 4: [Conv 3x3 -> BatchNorm -> ReLU] x2, 256 filtros -> MaxPool  (8x8)
  Global Average Pooling -> Dropout(0.4) -> Linear(256 -> 6)
Todos los pesos se inicializan aleatoriamente (inicialización de Kaiming).

¿QUÉ HACE EN EL PROYECTO?
  Recibe la imagen de UN residuo (o el recorte de un objeto del video) y dice de qué
  material es: cartón, metal, orgánico, papel, plástico o vidrio. A partir de esa clase,
  config.INFO_CLASES dice si es orgánico o inorgánico y en qué caneca va.

RESULTADOS (100 épocas, 989 imágenes de prueba nunca vistas):
  accuracy 88,7 % · F1 macro 0,885 · orgánico vs. inorgánico 97,8 % · 1 174 758 parámetros

ORGANIZACIÓN DEL ARCHIVO
  1. bloque() ................ un bloque convolucional
  2. EcoNet ................. la red completa (+ inicialización aleatoria)
  3. tf_entrenamiento ....... pre-procesamiento + AUMENTO DE DATOS
     tf_evaluacion .......... pre-procesamiento sin aumento (validación, prueba y app)
  4. Residuos ............... el dataset de imágenes
  5. ClasificadorCNN ........ uso del modelo entrenado en la aplicación (inferencia)
El entrenamiento (pérdida, optimizador, épocas) está en notebooks/01_cnn_desde_cero.ipynb.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2 as T

from config import CLASES, TAM_IMAGEN

# NORMALIZACIÓN: los píxeles (0..1) se transforman con (x - 0,5) / 0,5 y quedan entre -1 y 1.
# Las redes entrenan mejor con valores centrados en 0.
MEDIA = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


# =================================================================== 1. BLOQUE CONVOLUCIONAL
def bloque(c_in: int, c_out: int) -> nn.Sequential:
    """Un bloque de EcoNet = 2 × [Convolución -> BatchNorm -> ReLU] + MaxPool.

    - CONVOLUCIÓN 3×3: filtros de 3×3 que se deslizan sobre la imagen; cada filtro detecta
      un patrón (un borde, un color, una textura). c_out = número de filtros = número de
      "mapas de características" de salida. padding=1 conserva el tamaño de la imagen.
      bias=False porque la BatchNorm que sigue ya tiene su propio desplazamiento.
    - BATCHNORM: normaliza las activaciones de cada lote (media 0, varianza 1). El
      entrenamiento es más rápido y estable.
    - RELU: max(0, x). Es la función de activación: agrega NO LINEALIDAD (sin ella, apilar
      capas sería equivalente a una sola transformación lineal).
    - MAXPOOL 2×2: de cada cuadro de 2×2 se queda con el valor máximo, así que el alto y el
      ancho se reducen a la mitad. Menos cómputo y tolerancia a pequeños desplazamientos.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


# =================================================================== 2. LA RED COMPLETA
class EcoNet(nn.Module):
    """CNN completa. Cómo cambia la forma de los datos (canales × alto × ancho):
         entrada   3 × 128 × 128   (imagen RGB)
         bloque 1  32 × 64 × 64     -> bordes y colores
         bloque 2  64 × 32 × 32     -> combinaciones de bordes
         bloque 3 128 × 16 × 16     -> texturas (ondulación del cartón, brillo del metal)
         bloque 4 256 × 8 × 8       -> partes y formas de objetos
         GAP      256               -> un número por cada patrón aprendido
         Linear   6                 -> un puntaje (logit) por clase
    La imagen se hace MÁS PEQUEÑA y MÁS PROFUNDA: pierde detalle espacial y gana significado.
    """

    def __init__(self, n_clases: int = len(CLASES), canales=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        # Se apilan los 4 bloques: 3 -> 32 -> 64 -> 128 -> 256 canales.
        capas, c = [], 3
        for c_out in canales:
            capas.append(bloque(c, c_out))
            c = c_out
        self.features = nn.Sequential(*capas)   # "extractor de características"
        # GLOBAL AVERAGE POOLING: promedia cada mapa de 8×8 en un solo número -> vector de 256.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # CLASIFICADOR: aplanar -> DROPOUT (en entrenamiento apaga al azar el 40 % de las
        # neuronas para que la red no dependa de unas pocas, lo que reduce el sobreajuste)
        # -> capa lineal 256 -> 6 (una salida por clase).
        self.clasificador = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_clases))
        # ENTRENAMIENTO DESDE CERO: los pesos empiezan con valores ALEATORIOS (no hay pesos
        # pre-entrenados). La inicialización de Kaiming elige la escala adecuada para redes con
        # ReLU, de modo que las señales no se hagan ni demasiado grandes ni demasiado pequeñas.
        for m in self.modules():  # inicialización aleatoria (NO pre-entrenada)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        # FORWARD (propagación hacia adelante): imagen -> bloques -> GAP -> clasificador -> 6 logits.
        # El softmax que convierte los logits en probabilidades se aplica afuera
        # (en predecir() o dentro de la función de pérdida durante el entrenamiento).
        return self.clasificador(self.pool(self.features(x)))


# =================================================================== 3. TRANSFORMACIONES
# AUMENTO DE DATOS (data augmentation), SOLO EN ENTRENAMIENTO.
# Como la red parte de cero y solo hay 4 601 imágenes, en cada época cada imagen se modifica
# al azar. La red ve versiones distintas y aprende el MATERIAL, no a memorizar la foto.
tf_entrenamiento = T.Compose([
    T.ToImage(), T.RandomResizedCrop(TAM_IMAGEN, scale=(0.5, 1.0), antialias=True),  # recorte aleatorio (50-100 %) y redimensión a 128×128
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(25),        # volteos y rotación de hasta ±25°
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),                                             # brillo, contraste, saturación y tono
    T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA, STD),                  # a números entre -1 y 1
])
# Para validación, prueba y la app NO hay aumento: solo se redimensiona y normaliza,
# para medir el desempeño con la imagen tal cual.
tf_evaluacion = T.Compose([
    T.ToImage(), T.Resize(TAM_IMAGEN, antialias=True), T.CenterCrop(TAM_IMAGEN),
    T.ToDtype(torch.float32, scale=True), T.Normalize(MEDIA, STD),
])


# =================================================================== 4. DATASET
class Residuos(torch.utils.data.Dataset):
    """Imágenes de data/processed/<split>/<clase>/*.jpg cargadas en memoria.

    split = "train" (4 601 imágenes, para ajustar los pesos),
            "val"   (986, para elegir el mejor modelo y vigilar el sobreajuste) o
            "test"  (989, para el resultado final; nunca se usan para entrenar).
    La etiqueta de cada imagen es el nombre de su carpeta (cartón, metal, ...).
    """

    def __init__(self, raiz, split, tf):
        self.items = [(p, CLASES.index(p.parent.name)) for p in sorted((raiz / split).glob("*/*.jpg"))]
        self.imgs = [Image.open(p).convert("RGB").copy() for p, _ in self.items]   # todo en memoria: entrena más rápido
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        # Devuelve (imagen transformada, número de clase). El DataLoader del notebook agrupa
        # estos pares en LOTES de 64 imágenes, preparados en 6 procesos en paralelo.
        return self.tf(self.imgs[i]), self.items[i][1]


# =================================================================== 5. INFERENCIA
class ClasificadorCNN:
    """Envoltura para usar la CNN entrenada en la aplicación."""

    def __init__(self, ruta, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ruta, map_location=self.device)      # pesos aprendidos (models/econet.pt)
        self.clases = ckpt["clases"]
        # .eval() desactiva el dropout y hace que la BatchNorm use las estadísticas aprendidas.
        self.modelo = EcoNet(len(self.clases)).to(self.device).eval()
        self.modelo.load_state_dict(ckpt["modelo"])

    @torch.no_grad()   # en inferencia no se calculan gradientes (no se está entrenando)
    def predecir(self, imagenes: list[Image.Image]):
        """CÓMO CLASIFICA:
          1. Cada imagen se redimensiona a 128×128 y se normaliza (tf_evaluacion).
          2. La red produce 6 logits.
          3. SOFTMAX los convierte en probabilidades que suman 1.
          4. ARGMAX: la clase con mayor probabilidad es la predicción; esa probabilidad es la confianza.
        Devuelve una lista de (clase, confianza, probabilidades) por imagen.
        """
        if not imagenes:
            return []
        x = torch.stack([tf_evaluacion(im.convert("RGB")) for im in imagenes]).to(self.device)
        probs = self.modelo(x).softmax(-1).cpu()
        return [(self.clases[int(p.argmax())], float(p.max()), p) for p in probs]
