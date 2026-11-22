"""Configuración central del sistema de detección y clasificación de residuos."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DATA_RAW = RAIZ / "data" / "raw" / "garbage_classification"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
VIT_FINETUNED = MODELOS / "vit-residuos"

# ---------------------------------------------------------------------------
# Modelos pre-entrenados (Hugging Face). Se descargan una vez y corren LOCAL.
# ---------------------------------------------------------------------------
VIT_BASE = "google/vit-base-patch16-224-in21k"        # Encoder (Vision Transformer)
DETECTOR = "google/owlv2-base-patch16-ensemble"        # Detector Transformer (ViT + texto)
LLM = "Qwen/Qwen2.5-1.5B-Instruct"                     # Decoder (tipo GPT) para el asistente

# ---------------------------------------------------------------------------
# Clases del clasificador (fine-tuning) y mapeo desde el dataset original
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
MAX_POR_CLASE = 1000  # balancea el dataset (vidrio tiene ~2000)

# Información de cada clase. Código de colores de Colombia
# (Resolución 2184 de 2019): blanca = aprovechables, verde = orgánicos, negra = no aprovechables.
INFO_CLASES = {
    "plastico": {"nombre": "Plástico", "tipo": "Inorgánico", "caneca": "Blanca",
                 "color_bgr": (255, 140, 0),
                 "consejo": "Vacíalo, enjuágalo y aplástalo antes de depositarlo."},
    "papel":    {"nombre": "Papel", "tipo": "Inorgánico", "caneca": "Blanca",
                 "color_bgr": (200, 200, 200),
                 "consejo": "Debe estar limpio y seco; el papel con grasa o servilletas usadas va a la negra."},
    "carton":   {"nombre": "Cartón", "tipo": "Inorgánico", "caneca": "Blanca",
                 "color_bgr": (60, 120, 180),
                 "consejo": "Desármalo y aplánalo; si tiene restos de comida va a la negra."},
    "vidrio":   {"nombre": "Vidrio", "tipo": "Inorgánico", "caneca": "Blanca",
                 "color_bgr": (80, 200, 80),
                 "consejo": "Sin tapa y vacío. El vidrio roto envuélvelo en papel para proteger al reciclador."},
    "metal":    {"nombre": "Metal", "tipo": "Inorgánico", "caneca": "Blanca",
                 "color_bgr": (160, 160, 60),
                 "consejo": "Latas vacías y enjuagadas; aplástalas para ahorrar espacio."},
    "organico": {"nombre": "Orgánico", "tipo": "Orgánico", "caneca": "Verde",
                 "color_bgr": (0, 160, 0),
                 "consejo": "Restos de comida, cáscaras y residuos de jardín. Sirven para compostaje."},
}

# Consultas de texto para el detector OWLv2 (detección "open-vocabulary").
CONSULTAS_DETECTOR = [
    "a plastic bottle", "a plastic bag", "a plastic container", "a plastic cup",
    "a glass bottle", "a glass jar",
    "an aluminum can", "a metal can", "a tin can",
    "a cardboard box", "a juice carton", "a piece of cardboard",
    "a sheet of paper", "a newspaper", "a paper cup",
    "food waste", "a fruit peel", "a banana peel", "an apple", "leftover food",
]
# Pista de material que aporta cada consulta (se usa si el ViT no está seguro).
PISTA_CONSULTA = {
    "a plastic bottle": "plastico", "a plastic bag": "plastico", "a plastic container": "plastico",
    "a plastic cup": "plastico", "a glass bottle": "vidrio", "a glass jar": "vidrio",
    "an aluminum can": "metal", "a metal can": "metal", "a tin can": "metal",
    "a cardboard box": "carton", "a juice carton": "carton", "a piece of cardboard": "carton",
    "a sheet of paper": "papel", "a newspaper": "papel", "a paper cup": "papel",
    "food waste": "organico", "a fruit peel": "organico", "a banana peel": "organico",
    "an apple": "organico", "leftover food": "organico",
}
UMBRAL_CONFIANZA_VIT = 0.55
UMBRAL_DETECCION = 0.20
UMBRAL_NMS = 0.4
