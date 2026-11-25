"""Configuración central de EcoVisión.

Todo se construye y entrena DESDE CERO (sin modelos pre-entrenados ni fine-tuning):
  - EcoNet: CNN propia que clasifica el material del residuo.
  - EcoTransformer: Transformer encoder-decoder propio que recibe la salida de la CNN
    (y una pregunta opcional) y genera la retroalimentación en español.

Aquí están las rutas, las 6 clases, la regla orgánico/inorgánico y los umbrales de detección.
"""
from pathlib import Path

# ------------------------------------------------------------------ rutas
RAIZ = Path(__file__).resolve().parent
DATA_RAW = RAIZ / "data" / "raw" / "garbage_classification"   # dataset original (12 clases)
DATA_PROC = RAIZ / "data" / "processed"                         # dataset preparado (6 clases, train/val/test)
MODELOS = RAIZ / "models"
CNN_PESOS = MODELOS / "econet.pt"                 # pesos aprendidos por la CNN (notebook 01)
TRANSFORMER_PESOS = MODELOS / "ecotransformer.pt" # pesos + vocabularios del Transformer (notebook 02)
OUTPUTS = RAIZ / "outputs"                        # gráficas y videos de resultados

# ---------------------------------------------------------------------------
# Clases y mapeo desde el dataset original (12 clases -> 6)
# ---------------------------------------------------------------------------
# Las 6 clases que predice la CNN (en orden alfabético: es el orden de sus 6 salidas).
CLASES = ["carton", "metal", "organico", "papel", "plastico", "vidrio"]

# Dataset público "Garbage Classification" (12 clases). Se usan 8 y se agrupan en 6.
# No se usan: battery, clothes, shoes, trash.
MAPEO_DATASET = {
    "cardboard": "carton",
    "metal": "metal",
    "biological": "organico",
    "paper": "papel",
    "plastic": "plastico",
    "brown-glass": "vidrio",
    "green-glass": "vidrio",
    "white-glass": "vidrio",
}
MAX_POR_CLASE = None  # se usan todas las imágenes (el desbalance se corrige con pesos de clase)

TAM_IMAGEN = 128  # entrada de la CNN: imágenes de 128 × 128 píxeles

# Información de cada clase. Código de colores de Colombia (Resolución 2184 de 2019).
# CLASIFICACIÓN ORGÁNICO / INORGÁNICO: es una REGLA aplicada sobre la clase que predice la CNN.
#   orgánico -> caneca verde · plástico, papel, cartón, vidrio, metal -> inorgánico, caneca blanca.
# color_bgr = color de la caja que se dibuja en el video (OpenCV usa el orden Azul-Verde-Rojo).
INFO_CLASES = {
    "plastico": {"nombre": "Plástico", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (255, 140, 0)},
    "papel":    {"nombre": "Papel", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (200, 200, 200)},
    "carton":   {"nombre": "Cartón", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (60, 120, 180)},
    "vidrio":   {"nombre": "Vidrio", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (80, 200, 80)},
    "metal":    {"nombre": "Metal", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (160, 160, 60)},
    "organico": {"nombre": "Orgánico", "tipo": "Orgánico", "caneca": "Verde", "color_bgr": (0, 160, 0)},
}

# Detección de objetos (OpenCV clásico + CNN propia)
AREA_MIN = 0.004      # fracción mínima del área del cuadro para considerar un objeto (0,4 %)
AREA_MAX = 0.60       # fracción máxima (60 %): más grande suele ser fondo o varios objetos juntos
CONF_MIN_CNN = 0.40   # objetos con confianza de la CNN menor a esto se descartan por dudosos
