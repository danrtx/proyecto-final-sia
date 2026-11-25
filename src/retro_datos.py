"""Dataset propio para entrenar el EcoTransformer desde cero.

Cada ejemplo es un par (entrada -> salida):
  entrada: lo que detectó la CNN + una pregunta opcional del usuario
           "det plastico 2 vidrio 1 sep que hago con el vidrio roto"
  salida:  la retroalimentación en español
           "El vidrio roto no se aplasta ni se lava a mano. Recoge los trozos ..."

Las respuestas están basadas en la Resolución 2184 de 2019 (código de colores de Colombia).
Las preguntas se generan con plantillas y variaciones (sinónimos, saludos, cortesías);
algunas plantillas se reservan solo para prueba, para medir si el modelo generaliza
a frases que nunca vio.

¿POR QUÉ UN DATASET PROPIO?
  No existe un dataset público de "salida de una CNN de residuos -> recomendación en español",
  así que lo creamos nosotros. Esto también es parte de entrenar el modelo DESDE CERO.

TRES TIPOS DE EJEMPLOS (proporción 5 : 3 : 4)
  1. Retroalimentación general: detección + "dame recomendaciones" (o sin pregunta)
     -> una línea por material + un mensaje de cierre.
  2. Pregunta por material: "¿dónde boto una lata de atún?" -> consejo para el metal.
  3. Pregunta de conocimiento: vidrio roto, pilas, compost, ... -> respuesta fija.

TAMAÑOS USADOS EN EL NOTEBOOK 02
  40 000 de entrenamiento · 2 000 de validación · 2 000 de prueba con plantillas vistas
  · 2 000 de prueba con plantillas NUNCA vistas (para medir la generalización: 74,5 % exacto).
"""
from __future__ import annotations

import random
import re
import unicodedata

from config import CLASES

# ------------------------------------------------------------ retroalimentación por material
# Línea de la retroalimentación GENERAL para cada material que detecte la CNN.
# {n} se reemplaza por la cantidad detectada: el Transformer aprende a COPIAR ese número
# desde la entrada (se ve en el mapa de atención). Los textos tienen espacios alrededor de
# los signos para que la tokenización separe cada signo como un token.
LINEA_MATERIAL = {
    "plastico": "Plástico ( {n} ) : es inorgánico , va en la caneca blanca . Vacíalo , enjuágalo y aplástalo .",
    "papel": "Papel ( {n} ) : es inorgánico , va en la caneca blanca . Debe estar limpio y seco .",
    "carton": "Cartón ( {n} ) : es inorgánico , va en la caneca blanca . Desármalo y aplánalo .",
    "vidrio": "Vidrio ( {n} ) : es inorgánico , va en la caneca blanca . Vacío , sin tapa y con cuidado .",
    "metal": "Metal ( {n} ) : es inorgánico , va en la caneca blanca . Enjuaga las latas y aplástalas .",
    "organico": "Orgánico ( {n} ) : va en la caneca verde . Sirve para hacer compost .",
}
# Mensaje de cierre: depende de si se detectaron orgánicos, inorgánicos o ambos.
# El Transformer tiene que "razonar" sobre toda la entrada para elegir el correcto.
CIERRE = {
    "mixto": "Recuerda : orgánicos a la caneca verde e inorgánicos limpios y secos a la blanca .",
    "inorganico": "Todo es aprovechable : llévalo limpio y seco a la caneca blanca .",
    "organico": "Todo es orgánico : llévalo a la caneca verde o úsalo para compostaje .",
}
SIN_DETECCION = "No se detectaron residuos . Acerca el objeto a la cámara o analiza otra imagen ."

# Respuesta a una PREGUNTA POR MATERIAL ("¿qué hago con una lata?" -> consejo del metal).
# Clasificación orgánico/inorgánico y caneca según la Resolución 2184 de 2019.
CONSEJO_MATERIAL = {
    "plastico": "El plástico es inorgánico y va en la caneca blanca . Vacíalo , enjuágalo y aplástalo antes de botarlo .",
    "papel": "El papel es inorgánico y va en la caneca blanca . Debe estar limpio y seco ; el papel con grasa va en la negra .",
    "carton": "El cartón es inorgánico y va en la caneca blanca . Desármalo y aplánalo ; si tiene comida va en la negra .",
    "vidrio": "El vidrio es inorgánico y va en la caneca blanca . Debe ir vacío y sin tapa ; si está roto , envuélvelo en papel .",
    "metal": "El metal es inorgánico y va en la caneca blanca . Enjuaga las latas y aplástalas para ahorrar espacio .",
    "organico": "Los residuos orgánicos van en la caneca verde . Sirven para hacer compost y abono .",
}
# SINÓNIMOS de cada material: el modelo aprende que "lata", "aluminio" o "tapa metálica"
# se refieren al metal. Los 2 últimos de cada lista solo aparecen en la prueba de frases nuevas.
SINONIMOS = {
    "plastico": ["el plastico", "una botella plastica", "la botella de plastico", "un envase plastico", "un vaso plastico", "los plasticos", "una tapa plastica", "un empaque plastico"],
    "papel": ["el papel", "una hoja", "las hojas de papel", "el periodico", "un cuaderno viejo", "una revista", "los papeles", "una factura"],
    "carton": ["el carton", "una caja", "la caja de carton", "las cajas", "un carton", "una caja de cereal", "un tubo de carton", "los cartones"],
    "vidrio": ["el vidrio", "una botella de vidrio", "un frasco", "el tarro de vidrio", "las botellas de vidrio", "un frasco de mermelada", "una botella de cerveza", "los vidrios"],
    "metal": ["el metal", "una lata", "la lata de atun", "una lata de gaseosa", "el aluminio", "las latas", "una tapa metalica", "una lata de cerveza"],
    "organico": ["la comida", "una cascara de banano", "los restos de comida", "las cascaras", "la fruta podrida", "las verduras", "el cuncho de cafe", "las cascaras de huevo"],
}
# PLANTILLAS de pregunta por material: {m} se reemplaza por un sinónimo.
PLANTILLAS_MATERIAL = ["que hago con {m}", "donde boto {m}", "en que caneca va {m}", "como reciclo {m}",
                       "donde va {m}", "como preparo {m} para reciclar", "{m} donde lo boto", "a donde va {m}",
                       "como separo {m}", "en que caneca boto {m}"]

# ------------------------------------------------------------ preguntas generales (base de conocimiento)
# BASE DE CONOCIMIENTO: 13 temas. Cada tema es (respuesta, [8 formas de preguntarlo]).
# Las 6 primeras formas se usan para entrenar y las 2 últimas solo para la prueba de frases nuevas.
CONOCIMIENTO = {
    "vidrio_roto": ("El vidrio roto no se aplasta ni se lava a mano . Recoge los trozos con escoba , envuélvelos en papel periódico o en una caja , marca el paquete como vidrio roto y deposítalo en la caneca blanca .",
                    ["que hago con el vidrio roto", "donde boto vidrio roto", "como boto un vaso roto", "se me rompio una botella que hago",
                     "donde van los vidrios rotos", "como desecho un vidrio quebrado", "que hago si se rompe un frasco", "vidrio partido donde lo boto"]),
    "pizza_grasa": ("Una caja de pizza con grasa o restos de comida va en la caneca negra , porque el cartón sucio no se puede reciclar . Si una parte está limpia , córtala y llévala a la blanca .",
                    ["donde boto una caja de pizza con grasa", "la caja de pizza donde va", "que hago con una caja de pizza sucia", "carton con grasa donde lo boto",
                     "caja de pizza usada que hago", "en que caneca va la caja de pizza", "donde va el carton sucio de comida", "como boto la caja de la pizza"]),
    "pilas": ("Las pilas y baterías son residuos peligrosos : no van en ninguna caneca . Llévalas a un punto posconsumo , como los contenedores de Pilas con el Ambiente .",
              ["donde boto las pilas", "que hago con las pilas usadas", "las baterias donde van", "como desecho una bateria",
               "donde boto pilas viejas", "en que caneca van las pilas", "que hago con una bateria de celular", "pilas gastadas donde las llevo"]),
    "servilletas": ("Las servilletas usadas , el papel higiénico y los pañales van en la caneca negra porque no son aprovechables .",
                    ["donde boto una servilleta usada", "el papel higienico donde va", "que hago con los panales", "servilletas sucias en que caneca",
                     "donde van las servilletas", "papel higienico usado donde lo boto", "en que caneca va un panal", "las toallas de papel usadas donde van"]),
    "compost": ("Para hacer compost alterna capas de residuos verdes ( cáscaras , restos de fruta , cuncho de café ) y secos ( hojas secas , cartón sin tinta ) . Mantén la mezcla húmeda y revuélvela cada semana . En dos o tres meses tendrás abono .",
                ["como hago compost", "como hago compost en casa", "que es el compostaje", "como hago abono con los organicos",
                 "como aprovecho los residuos organicos", "ensename a hacer compost", "como se hace una compostera", "que hago para tener abono casero"]),
    "diferencia": ("Los residuos orgánicos vienen de seres vivos y se descomponen rápido , como la comida y las cáscaras . Los inorgánicos son materiales fabricados que tardan años en degradarse , como el plástico , el vidrio y el metal .",
                   ["cual es la diferencia entre organico e inorganico", "que es un residuo organico", "que es un residuo inorganico", "diferencia entre organicos e inorganicos",
                    "que significa organico", "como se si algo es organico", "explicame organico e inorganico", "que residuos son organicos"]),
    "colores": ("Caneca blanca : plástico , vidrio , metal , papel y cartón limpios y secos . Caneca verde : residuos orgánicos . Caneca negra : residuos no aprovechables como servilletas y papel higiénico .",
                ["que significan los colores de las canecas", "que va en cada caneca", "cuales son los colores de las canecas", "que va en la caneca blanca",
                 "que va en la caneca negra", "que va en la caneca verde", "explicame el codigo de colores", "para que es cada caneca"]),
    "por_que": ("Separar en la fuente permite reciclar más material , reduce la basura que llega al relleno sanitario y facilita el trabajo de los recicladores .",
                ["por que es importante reciclar", "para que sirve separar la basura", "por que debo separar los residuos", "cual es el beneficio de reciclar",
                 "de que sirve reciclar", "por que separar en la fuente", "que ganamos reciclando", "es importante separar la basura"]),
    "bolsas": ("Las bolsas plásticas limpias y secas van en la caneca blanca ; si están sucias de comida van en la negra . Lo mejor es reutilizarlas .",
               ["donde boto las bolsas plasticas", "que hago con las bolsas", "las bolsas sucias donde van", "en que caneca van las bolsas",
                "como reciclo una bolsa", "bolsas de mercado donde las boto", "donde van las bolsas de plastico", "una bolsa sucia en que caneca va"]),
    "tetra": ("Los envases de jugo y leche ( Tetra Pak ) se vacían , se enjuagan , se aplastan y van en la caneca blanca .",
              ["donde boto la caja de leche", "el envase de jugo donde va", "que hago con un tetra pak", "caja de jugo en que caneca va",
               "como reciclo una caja de leche", "donde van los envases de leche", "el carton de jugo donde lo boto", "que hago con la cajita de jugo"]),
    "aceite": ("El aceite de cocina usado no se bota por el desagüe . Guárdalo en una botella cerrada y llévalo a un punto de recolección autorizado .",
               ["donde boto el aceite usado", "que hago con el aceite de cocina", "el aceite se bota por el lavaplatos", "como desecho aceite",
                "aceite de freir donde lo boto", "que hago con el aceite viejo", "donde llevo el aceite usado", "puedo botar aceite al sifon"]),
    "medicamentos": ("Los medicamentos vencidos son residuos peligrosos . Llévalos a los puntos azules de las farmacias .",
                     ["donde boto medicamentos vencidos", "que hago con las pastillas vencidas", "los remedios viejos donde van", "como desecho medicamentos",
                      "donde boto jarabes vencidos", "medicinas caducadas que hago", "en que caneca van los medicamentos", "donde llevo los medicamentos que sobran"]),
    "saludo": ("Hola , soy EcoAsistente . Puedo darte recomendaciones sobre los residuos detectados o responder preguntas sobre cómo separar la basura .",
               ["hola", "buenos dias", "quien eres", "que puedes hacer", "buenas tardes", "en que me ayudas", "como funcionas", "hola que tal"]),
}

# Frases que piden la RETROALIMENTACIÓN GENERAL de lo detectado ("" = sin pregunta, el caso del
# botón "Recomendaciones"; aparece 3 veces para que sea el caso más frecuente).
PREGUNTAS_GENERALES = ["", "", "", "que hago con esto", "dame recomendaciones", "como separo estos residuos",
                       "en que caneca va esto", "que se detecto", "recomendaciones", "que debo hacer con lo detectado",
                       "ayudame a separar", "como boto esto", "que hago con lo que detectaste", "dame retroalimentacion"]
# Saludos y cortesías que se agregan al azar para que el modelo sea robusto a variaciones.
PREFIJOS = ["", "", "", "hola ", "oye ", "una pregunta ", "disculpa ", "ecoasistente "]
SUFIJOS = ["", "", "", " por favor", " gracias", " porfa"]

N_RESERVADAS = 2  # últimas plantillas de cada intención: solo en prueba
# ^ Así se mide la GENERALIZACIÓN: si el modelo solo memorizara, fallaría con estas frases.


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y sin signos: "¿Dónde boto la LATA?" -> "donde boto la lata".
    Así el usuario puede escribir como quiera y el vocabulario de entrada es más pequeño."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", texto))


def entrada(conteo: dict[str, int], pregunta: str = "") -> str:
    """Construye la secuencia de entrada a partir de la salida de la CNN.

    PUENTE CNN -> TRANSFORMER. Ejemplo:
        conteo = {"plastico": 2, "vidrio": 1}, pregunta = "¿Qué hago con esto?"
        ->  "det plastico 2 vidrio 1 sep que hago con esto"
    "det" marca el inicio de la detección y "sep" la separa de la pregunta.
    Los materiales van ordenados de mayor a menor cantidad; las cantidades se limitan a 30.
    """
    if conteo:
        partes = [f"{c} {min(n, 30)}" for c, n in sorted(conteo.items(), key=lambda x: (-x[1], x[0]))]
        det = "det " + " ".join(partes)
    else:
        det = "det ninguno"
    return f"{det} sep {normalizar(pregunta)}".strip()


def retro_general(conteo: dict[str, int]) -> str:
    """Respuesta CORRECTA (la que el Transformer debe aprender a generar) para la
    retroalimentación general: una línea por material + el cierre adecuado."""
    if not conteo:
        return SIN_DETECCION
    orden = sorted(conteo.items(), key=lambda x: (-x[1], x[0]))
    lineas = [LINEA_MATERIAL[c].format(n=min(n, 30)) for c, n in orden]
    org = "organico" in conteo
    inorg = any(c != "organico" for c in conteo)
    lineas.append(CIERRE["mixto" if org and inorg else ("organico" if org else "inorganico")])
    return " ".join(lineas)


def _conteo_aleatorio(rng: random.Random) -> dict[str, int]:
    """Simula una salida de la CNN: de 0 a 6 materiales distintos, cada uno con una cantidad
    de 1 a 30 (las cantidades pequeñas, de 1 a 5, son las más probables)."""
    k = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[1, 4, 4, 3, 2, 1, 1])[0]
    clases = rng.sample(CLASES, k)
    return {c: rng.choices(range(1, 31), weights=[30 if i < 5 else 3 for i in range(30)])[0] for c in clases}


def _decorar(p: str, rng: random.Random) -> str:
    return (rng.choice(PREFIJOS) + p + rng.choice(SUFIJOS)).strip() if p else p


def generar(n: int, semilla: int = 0, reservadas: bool = False) -> list[tuple[str, str]]:
    """Genera n pares (entrada, salida). Con reservadas=True usa solo plantillas no vistas.
    La semilla hace que el dataset sea reproducible (siempre se generan los mismos ejemplos)."""
    rng = random.Random(semilla)
    sel = (lambda l: l[-N_RESERVADAS:]) if reservadas else (lambda l: l[:-N_RESERVADAS])
    pares = []
    for _ in range(n):
        conteo = _conteo_aleatorio(rng)
        tipo = rng.choices(["general", "material", "conocimiento"], weights=[5, 3, 4])[0]
        if tipo == "general":
            p = _decorar(rng.choice(sel(PREGUNTAS_GENERALES[3:]) if reservadas else PREGUNTAS_GENERALES[:-N_RESERVADAS]), rng)
            salida = retro_general(conteo)
        elif tipo == "material":
            m = rng.choice(CLASES)
            p = _decorar(rng.choice(sel(PLANTILLAS_MATERIAL)).format(m=rng.choice(sel(SINONIMOS[m]))), rng)
            salida = CONSEJO_MATERIAL[m]
        else:
            clave = rng.choice(list(CONOCIMIENTO))
            resp, plantillas = CONOCIMIENTO[clave]
            p = _decorar(rng.choice(sel(plantillas)), rng)
            salida = resp
        pares.append((entrada(conteo, p), salida))
    return pares
