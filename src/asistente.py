"""Asistente de recomendaciones de separación de residuos.

Usa un LLM pre-entrenado tipo GPT (Transformer solo-decoder, Qwen2.5-1.5B-Instruct)
que corre 100% local con `transformers`; no hay ninguna API externa.
Para que no "invente", se le inyecta:
  1) Una base de conocimiento propia (normas de separación en Colombia).
  2) Los fragmentos de esa base relevantes a la pregunta (recuperación por palabras clave).
  3) El resultado de la última detección del sistema de visión.
"""
from __future__ import annotations

import re
import unicodedata
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from config import LLM

REGLAS = """Código de colores en Colombia (Resolución 2184 de 2019, vigente desde 2021):
- Caneca BLANCA: residuos aprovechables, limpios y secos: plástico, vidrio, metales, papel y cartón.
- Caneca VERDE: residuos orgánicos aprovechables: restos de comida, cáscaras, residuos de jardín.
- Caneca NEGRA: residuos no aprovechables: papel higiénico, servilletas, papeles y cartones con grasa o comida, empaques metalizados contaminados, colillas.
Residuos peligrosos (pilas, baterías, medicamentos, bombillos, aparatos electrónicos, aceite usado) NO van en ninguna caneca: se llevan a puntos posconsumo."""

BASE_CONOCIMIENTO = [
    (["botella", "plastico", "pet", "envase"], "Botellas plásticas (PET): vaciar, enjuagar, aplastar y tapar. Caneca blanca. Son inorgánicas y 100% reciclables."),
    (["bolsa", "plastico"], "Bolsas plásticas: si están limpias y secas van a la blanca; si están sucias de comida van a la negra. Mejor reutilizarlas."),
    (["icopor", "poliestireno", "desechable"], "Icopor y desechables: si están limpios, blanca; con restos de comida, negra. Son difíciles de reciclar: mejor evitarlos."),
    (["vidrio", "frasco", "botella"], "Vidrio (botellas y frascos): vacío, sin tapa y enjuagado. Caneca blanca."),
    (["roto", "rota", "rotos", "vidrio", "cortante", "espejo"], "Vidrio roto: NO se aplasta ni se lava a mano (es peligroso). Recoge los trozos con escoba y recogedor, envuélvelos en papel periódico o en una caja de cartón, cierra bien, marca el paquete como 'VIDRIO ROTO – PELIGRO' y deposítalo en la caneca blanca. Así proteges al reciclador."),
    (["lata", "metal", "aluminio", "tapa"], "Latas de aluminio y metal: vaciar, enjuagar y aplastar. Caneca blanca. El aluminio se recicla infinitas veces."),
    (["carton", "caja"], "Cartón: desarmar y aplanar las cajas, debe estar seco y sin grasa. Caneca blanca. Una caja de pizza con grasa va a la negra."),
    (["tetra", "jugo", "leche", "carton"], "Envases de jugo o leche (Tetra Pak): vaciar, enjuagar y aplastar. Caneca blanca."),
    (["papel", "hoja", "periodico", "cuaderno", "revista"], "Papel (hojas, periódico, cuadernos, revistas): limpio y seco, sin arrugar. Caneca blanca. El papel mojado o con grasa va a la negra."),
    (["servilleta", "higienico", "panal", "toalla"], "Papel higiénico, servilletas usadas, pañales y toallas higiénicas: caneca negra (no aprovechables)."),
    (["comida", "organico", "cascara", "fruta", "verdura", "cafe", "huevo", "banano"], "Residuos orgánicos (cáscaras, restos de fruta y verdura, cuncho de café, cáscaras de huevo): caneca verde. Sirven para compostaje y abono."),
    (["compost", "abono", "compostaje"], "Compostaje casero en 5 pasos: 1) Usa un recipiente con huecos o un rincón del patio. 2) Alterna capas de residuos verdes (cáscaras de fruta y verdura, cuncho de café, cáscaras de huevo) y secos (hojas secas, aserrín, cartón sin tinta). 3) Mantén la mezcla húmeda como una esponja escurrida. 4) Revuélvela una vez por semana para airearla. 5) En 2 a 3 meses, cuando huela a tierra y se vea oscura, ya es abono. NO agregues carne, lácteos, aceites, papel higiénico, plásticos, vidrio ni metal."),
    (["pila", "bateria", "celular", "electronico", "bombillo", "medicamento", "aceite"], "Pilas, baterías, electrónicos, bombillos, medicamentos y aceite usado son residuos peligrosos: llevarlos a puntos posconsumo (por ejemplo, programas como Pilas con el Ambiente o Punto Azul)."),
    (["organico", "inorganico", "diferencia"], "Orgánico: proviene de seres vivos y se descompone naturalmente (comida, cáscaras, hojas). Inorgánico: materiales fabricados que tardan mucho en degradarse (plástico, vidrio, metal, papel y cartón procesados)."),
    (["por que", "importancia", "beneficio", "reciclar"], "Separar en la fuente aumenta el material que se recupera, reduce lo que llega al relleno sanitario, dignifica el trabajo de los recicladores y disminuye la contaminación."),
]

SISTEMA = f"""Eres "EcoAsistente", un asistente experto en separación de residuos y reciclaje en Colombia.
Respondes SIEMPRE en español, de forma breve, clara y práctica (máximo 6 viñetas o 120 palabras).
Reglas obligatorias:
1. Basa tu respuesta PRIMERO en los "DATOS ESPECÍFICOS" y repite sus indicaciones con fidelidad.
2. No inventes procedimientos ni pasos que no estén en la información de referencia.
3. Si la información no alcanza para responder, dilo con honestidad.
4. Escribe los nombres de los materiales y los colores de caneca correctamente.

INFORMACIÓN DE REFERENCIA:
{REGLAS}"""


def _normalizar(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _texto(contenido) -> str:
    """Gradio 6 guarda el contenido del chat como lista de partes [{'type':'text','text':...}]."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, dict):
        return contenido.get("text", "")
    if isinstance(contenido, (list, tuple)):
        return " ".join(_texto(p) for p in contenido)
    return str(contenido)


def recuperar(pregunta: str, k: int = 3) -> list[str]:
    palabras = set(re.findall(r"[a-z]+", _normalizar(pregunta)))
    texto = _normalizar(pregunta)
    puntajes = []
    for claves, frag in BASE_CONOCIMIENTO:
        p = sum(1 for c in claves if c in palabras or (" " in c and c in texto)
                or any(w.startswith(c) for w in palabras))
        if p:
            puntajes.append((p, frag))
    return [f for _, f in sorted(puntajes, key=lambda x: -x[0])[:k]]


class Asistente:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(LLM)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def _mensajes(self, pregunta: str, historial: list[dict], deteccion: str | None):
        sistema = SISTEMA
        # Primero se busca con la pregunta; la detección solo se usa si la pregunta no da resultados.
        frags = recuperar(pregunta) or recuperar(deteccion or "")
        if frags:
            sistema += "\n\nDATOS ESPECÍFICOS:\n- " + "\n- ".join(frags)
        if deteccion:
            sistema += f"\n\nÚLTIMA DETECCIÓN DEL SISTEMA DE VISIÓN:\n{deteccion}"
        msgs = [{"role": "system", "content": sistema}]
        msgs += [{"role": m["role"], "content": _texto(m["content"])} for m in historial[-6:]]
        msgs.append({"role": "user", "content": pregunta})
        return msgs

    def responder_stream(self, pregunta: str, historial: list[dict] | None = None,
                         deteccion: str | None = None, max_tokens: int = 220):
        msgs = self._mensajes(pregunta, historial or [], deteccion)
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], streamer=streamer,
                      max_new_tokens=max_tokens, do_sample=False,
                      repetition_penalty=1.05)
        Thread(target=self.model.generate, kwargs=kwargs, daemon=True).start()
        texto = ""
        for parte in streamer:
            texto += parte
            yield texto

    def responder(self, pregunta: str, historial=None, deteccion=None) -> str:
        r = ""
        for r in self.responder_stream(pregunta, historial, deteccion):
            pass
        return r
