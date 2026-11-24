"""Configuración central de EcoVisión.

Todo se construye y entrena DESDE CERO (sin modelos pre-entrenados ni fine-tuning):
  - EcoNet: CNN propia que clasifica el material del residuo.
  - EcoTransformer: Transformer encoder-decoder propio que recibe la salida de la CNN
    (y una pregunta opcional) y genera la retroalimentación en español.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATA_RAW = RAIZ / "data" / "raw" / "garbage_classification"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
CNN_PESOS = MODELOS / "econet.pt"
TRANSFORMER_PESOS = MODELOS / "ecotransformer.pt"
OUTPUTS = RAIZ / "outputs"

# ---------------------------------------------------------------------------
# Clases y mapeo desde el dataset original (12 clases -> 6)
# ---------------------------------------------------------------------------
CLASES = ["carton", "metal", "organico", "papel", "plastico", "vidrio"]

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

TAM_IMAGEN = 128  # entrada de la CNN

# Información de cada clase. Código de colores de Colombia (Resolución 2184 de 2019).
INFO_CLASES = {
    "plastico": {"nombre": "Plástico", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (255, 140, 0)},
    "papel":    {"nombre": "Papel", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (200, 200, 200)},
    "carton":   {"nombre": "Cartón", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (60, 120, 180)},
    "vidrio":   {"nombre": "Vidrio", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (80, 200, 80)},
    "metal":    {"nombre": "Metal", "tipo": "Inorgánico", "caneca": "Blanca", "color_bgr": (160, 160, 60)},
    "organico": {"nombre": "Orgánico", "tipo": "Orgánico", "caneca": "Verde", "color_bgr": (0, 160, 0)},
}

# Detección de objetos (OpenCV clásico + CNN propia)
AREA_MIN = 0.004      # fracción mínima del área del cuadro para considerar un objeto
AREA_MAX = 0.60
CONF_MIN_CNN = 0.40   # por debajo se marca como "dudoso"
