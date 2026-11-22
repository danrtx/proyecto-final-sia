"""EcoVisión – Sistema inteligente de detección y clasificación de residuos.

Ejecutar:  .venv\\Scripts\\python app.py   ->  abrir http://127.0.0.1:7860
Todo corre localmente (modelos Transformer pre-entrenados + ViT afinado). Sin APIs externas.
"""
import tempfile
from collections import Counter

import gradio as gr
import pandas as pd

from config import INFO_CLASES
from src.asistente import Asistente
from src.detector import SistemaResiduos, resumen_texto

print("Cargando modelos (OWLv2 + ViT afinado)...")
sistema = SistemaResiduos()
_asistente = None


def asistente():
    global _asistente
    if _asistente is None:
        print("Cargando LLM local (Qwen2.5-1.5B-Instruct)...")
        _asistente = Asistente()
    return _asistente


def tabla(conteo: Counter) -> pd.DataFrame:
    filas = [{"Material": INFO_CLASES[c]["nombre"], "Cantidad": n, "Tipo": INFO_CLASES[c]["tipo"],
              "Caneca": INFO_CLASES[c]["caneca"], "Recomendación": INFO_CLASES[c]["consejo"]}
             for c, n in conteo.most_common()]
    return pd.DataFrame(filas, columns=["Material", "Cantidad", "Tipo", "Caneca", "Recomendación"])


def tabla_detalle(dets) -> pd.DataFrame:
    return pd.DataFrame([{"Objeto (detector)": d.consulta.replace("a ", "", 1).replace("an ", "", 1),
                          "Material": d.info["nombre"], "Confianza": round(d.conf_clase, 2),
                          "Decidido por": d.fuente, "Tipo": d.info["tipo"], "Caneca": d.info["caneca"]}
                         for d in dets])


# ----------------------------------------------------------------- callbacks
def analizar_imagen(img):
    if img is None:
        return None, None, None, "", ""
    anotada, dets = sistema.procesar_imagen(img)
    conteo = Counter(d.clase for d in dets)
    txt = resumen_texto(conteo)
    return anotada, tabla(conteo), tabla_detalle(dets), txt, txt


def analizar_video(ruta, cada_n, progress=gr.Progress()):
    if not ruta:
        return None, None, "", ""
    salida = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    _, conteo, stats = sistema.procesar_video(ruta, salida, cada_n=int(cada_n), progreso=progress)
    txt = resumen_texto(conteo)
    info = f"{txt}\n\n({stats['cuadros']} cuadros procesados en {stats['segundos']:.1f} s)"
    return salida, tabla(conteo), info, txt


def webcam(frame):
    if frame is None:
        return None
    anotada, _ = sistema.procesar_imagen(frame)
    return anotada


def chat(mensaje, historial, deteccion):
    historial = historial or []
    historial.append({"role": "user", "content": mensaje})
    historial.append({"role": "assistant", "content": ""})
    for parcial in asistente().responder_stream(mensaje, historial[:-2], deteccion or None):
        historial[-1]["content"] = parcial
        yield "", historial


def recomendar(historial, deteccion):
    if not deteccion:
        msg = "Analiza primero una imagen o video para darte recomendaciones."
        yield (historial or []) + [{"role": "assistant", "content": msg}]
        return
    # 1) Recomendaciones deterministas por material detectado (siempre correctas).
    lineas = ["**Recomendaciones para lo detectado:**"]
    for c, inf in INFO_CLASES.items():
        for linea in deteccion.splitlines():
            if linea.startswith(f"- {inf['nombre']}:"):
                n = linea.split(":")[1].split("(")[0].strip()
                lineas.append(f"- **{inf['nombre']}** ({n}) → caneca **{inf['caneca']}** "
                              f"({inf['tipo'].lower()}). {inf['consejo']}")
    base = "\n".join(lineas) + "\n\n**Consejo del EcoAsistente:** "
    historial = (historial or []) + [
        {"role": "user", "content": "♻️ Recomendaciones para la última detección"},
        {"role": "assistant", "content": base}]
    yield historial
    # 2) El LLM local agrega un consejo general breve.
    pregunta = ("Con base en los residuos detectados, da UN solo consejo práctico y breve (máximo 2 frases) "
                "para separarlos mejor en casa.")
    for parcial in asistente().responder_stream(pregunta, [], deteccion, max_tokens=90):
        historial[-1]["content"] = base + parcial
        yield historial


# ----------------------------------------------------------------------- UI
with gr.Blocks(title="EcoVisión – Residuos") as demo:
    ultima_det = gr.State("")
    gr.Markdown("# ♻️ EcoVisión: detección y clasificación de residuos con Transformers\n"
                "**Detector:** OWLv2 (Vision Transformer) · **Clasificador:** ViT afinado (plástico, papel, cartón, vidrio, metal, orgánico) · "
                "**Asistente:** Qwen2.5 (LLM local). Todo corre en este equipo, sin APIs.")

    with gr.Tab("🎥 Video"):
        with gr.Row():
            v_in = gr.Video(label="Video de entrada")
            v_out = gr.Video(label="Resultado")
        cada_n = gr.Slider(1, 10, value=3, step=1, label="Analizar 1 de cada N cuadros (más alto = más rápido)")
        v_btn = gr.Button("Detectar y clasificar", variant="primary")
        v_tabla = gr.Dataframe(label="Residuos únicos encontrados en el video")
        v_txt = gr.Textbox(label="Resumen", lines=6)
        v_btn.click(analizar_video, [v_in, cada_n], [v_out, v_tabla, v_txt, ultima_det])

    with gr.Tab("🖼️ Imagen"):
        with gr.Row():
            i_in = gr.Image(type="pil", label="Imagen de entrada")
            i_out = gr.Image(label="Resultado")
        i_btn = gr.Button("Detectar y clasificar", variant="primary")
        i_tabla = gr.Dataframe(label="Resumen por material")
        i_det = gr.Dataframe(label="Detalle de cada objeto")
        i_txt = gr.Textbox(label="Resumen", lines=6)
        i_btn.click(analizar_imagen, i_in, [i_out, i_tabla, i_det, i_txt, ultima_det])

    with gr.Tab("📷 Webcam en vivo"):
        with gr.Row():
            w_in = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Cámara")
            w_out = gr.Image(label="Detección en vivo")
        w_in.stream(webcam, w_in, w_out, stream_every=0.5, concurrency_limit=1)

    with gr.Tab("💬 EcoAsistente"):
        gr.Markdown("Pregunta cómo separar un residuo, o pide recomendaciones sobre lo último que se analizó.")
        bot = gr.Chatbot(height=420, label="EcoAsistente")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ej: ¿Dónde boto una caja de pizza con grasa?", scale=5, show_label=False)
            enviar = gr.Button("Enviar", variant="primary", scale=1)
        rec = gr.Button("♻️ Recomendaciones para la última detección")
        gr.Examples(["¿Cuál es la diferencia entre residuos orgánicos e inorgánicos?",
                     "¿Dónde boto las pilas usadas?",
                     "¿Cómo preparo una botella de plástico para reciclarla?",
                     "¿Cómo hago compost en casa?"], msg)
        msg.submit(chat, [msg, bot, ultima_det], [msg, bot])
        enviar.click(chat, [msg, bot, ultima_det], [msg, bot])
        rec.click(recomendar, [bot, ultima_det], bot)

if __name__ == "__main__":
    demo.queue().launch(inbrowser="--no-browser" not in __import__("sys").argv, theme=gr.themes.Soft(primary_hue="green"))
