"""EcoVisión – Detección y clasificación de residuos con modelos construidos desde cero.

  - EcoNet (CNN propia) clasifica cada residuo detectado.
  - EcoTransformer (Transformer propio) toma la salida de la CNN y genera la retroalimentación.
Ejecutar:  .venv\\Scripts\\python app.py   ->  http://127.0.0.1:7860

CÓMO CUMPLE LOS REQUISITOS DEL PROYECTO
  1. Detectar objetos en video ............. pestaña "Video"  -> analizar_video()
  2. Clasificar en plástico, papel, ... .... EcoNet (src/cnn.py), dentro de SistemaResiduos
  3. Orgánicos e inorgánicos ............... columna "Tipo" + resumen_texto()
  4. Mostrar el resultado .................. video/imagen anotados + tablas
  5. Asistente de recomendaciones .......... EcoTransformer: recuadro de retroalimentación y pestaña "EcoAsistente"

FLUJO:  video/imagen -> SistemaResiduos (OpenCV + CNN) -> conteo -> Retroalimentador (Transformer) -> texto
La interfaz web se construye con Gradio (no es un modelo de IA, solo la "pantalla").
"""
import sys
import tempfile
from collections import Counter

import gradio as gr
import pandas as pd

from config import INFO_CLASES, TRANSFORMER_PESOS
from src.detector import SistemaResiduos, resumen_texto
from src.transformer import Retroalimentador

# Los dos modelos se cargan UNA vez al iniciar la app (pesos de models/econet.pt y models/ecotransformer.pt).
print("Cargando EcoNet (CNN) y EcoTransformer...")
sistema = SistemaResiduos()                    # localización OpenCV + CNN + tracker
retro = Retroalimentador(TRANSFORMER_PESOS)    # Transformer de retroalimentación


def tabla(conteo: Counter) -> pd.DataFrame:
    """Tabla resumen por material: cantidad, tipo (orgánico/inorgánico) y caneca."""
    return pd.DataFrame([{"Material": INFO_CLASES[c]["nombre"], "Cantidad": n,
                          "Tipo": INFO_CLASES[c]["tipo"], "Caneca": INFO_CLASES[c]["caneca"]}
                         for c, n in conteo.most_common()], columns=["Material", "Cantidad", "Tipo", "Caneca"])


def tabla_detalle(dets) -> pd.DataFrame:
    """Tabla con cada objeto detectado y la confianza de la CNN."""
    return pd.DataFrame([{"Objeto": i + 1, "Material (CNN)": d.info["nombre"], "Confianza": round(d.conf, 2),
                          "Tipo": d.info["tipo"], "Caneca": d.info["caneca"]} for i, d in enumerate(dets)])


# ----------------------------------------------------------------- callbacks
# Cada función se ejecuta cuando el usuario pulsa un botón de la interfaz.

def analizar_imagen(img):
    """Pestaña Imagen: CNN sobre cada objeto de la foto -> Transformer da la retroalimentación."""
    if img is None:
        return None, None, None, "", "", {}
    anotada, dets = sistema.procesar_imagen(img)          # OpenCV + CNN
    conteo = Counter(d.clase for d in dets)
    texto, _ = retro.responder(dict(conteo))             # el Transformer TOMA LA RESPUESTA DE LA CNN
    # El último valor (conteo) se guarda en "ultima" para que el EcoAsistente lo use después.
    return anotada, tabla(conteo), tabla_detalle(dets), resumen_texto(conteo), texto, dict(conteo)


def analizar_video(ruta, cada_n, progress=gr.Progress()):
    """Pestaña Video: detección + clasificación cuadro a cuadro, conteo de objetos únicos
    con el tracker y, al final, la retroalimentación del Transformer."""
    if not ruta:
        return None, None, "", "", {}
    salida = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    _, conteo, st = sistema.procesar_video(ruta, salida, cada_n=int(cada_n), progreso=progress)
    texto, _ = retro.responder(dict(conteo))
    info = f"{resumen_texto(conteo)}\n\n({st['cuadros']} cuadros procesados en {st['segundos']:.1f} s)"
    return salida, tabla(conteo), info, texto, dict(conteo)


def webcam(frame):
    """Pestaña Webcam: analiza un cuadro de la cámara cada 0,5 s."""
    if frame is None:
        return None, ""
    anotada, dets = sistema.procesar_imagen(frame)
    texto, _ = retro.responder(dict(Counter(d.clase for d in dets)))
    return anotada, texto


def chat(mensaje, historial, conteo):
    """EcoAsistente: la pregunta del usuario + lo último que detectó la CNN -> Transformer.
    Debajo de cada respuesta se muestra la entrada exacta que recibió el Transformer."""
    historial = list(historial or [])
    if not mensaje.strip():
        return "", historial
    texto, entrada = retro.responder(conteo or {}, mensaje)
    historial += [{"role": "user", "content": mensaje},
                  {"role": "assistant", "content": f"{texto}\n\n`entrada del Transformer: {entrada}`"}]
    return "", historial


def recomendar(historial, conteo):
    """Botón "Recomendaciones": retroalimentación general sobre la última detección (sin pregunta)."""
    texto, entrada = retro.responder(conteo or {})
    return list(historial or []) + [{"role": "user", "content": "♻️ Recomendaciones para la última detección"},
                                    {"role": "assistant", "content": f"{texto}\n\n`entrada del Transformer: {entrada}`"}]


# ----------------------------------------------------------------------- UI
# Interfaz web con 4 pestañas. "ultima" guarda el conteo de la última detección
# para compartirlo entre las pestañas (así el EcoAsistente sabe qué se detectó).
with gr.Blocks(title="EcoVisión – Residuos") as demo:
    ultima = gr.State({})
    gr.Markdown("# ♻️ EcoVisión: detección y clasificación de residuos\n"
                "**EcoNet**: CNN propia que clasifica plástico, papel, cartón, vidrio, metal y orgánico · "
                "**EcoTransformer**: Transformer propio que toma la salida de la CNN y da la retroalimentación. "
                "Los dos modelos se construyeron y entrenaron **desde cero**.")

    with gr.Tab("🎥 Video"):
        with gr.Row():
            v_in = gr.Video(label="Video de entrada")
            v_out = gr.Video(label="Resultado")
        # Analizar 1 de cada N cuadros: compromiso entre velocidad y precisión (ver procesar_video).
        cada_n = gr.Slider(1, 10, value=2, step=1, label="Analizar 1 de cada N cuadros (más alto = más rápido)")
        v_btn = gr.Button("Detectar y clasificar", variant="primary")
        with gr.Row():
            v_tabla = gr.Dataframe(label="Residuos únicos encontrados")
            v_retro = gr.Textbox(label="💬 Retroalimentación del EcoTransformer", lines=8)
        v_txt = gr.Textbox(label="Resumen", lines=5)
        v_btn.click(analizar_video, [v_in, cada_n], [v_out, v_tabla, v_txt, v_retro, ultima])

    with gr.Tab("🖼️ Imagen"):
        with gr.Row():
            i_in = gr.Image(type="pil", label="Imagen de entrada")
            i_out = gr.Image(label="Resultado")
        i_btn = gr.Button("Detectar y clasificar", variant="primary")
        with gr.Row():
            i_tabla = gr.Dataframe(label="Resumen por material")
            i_retro = gr.Textbox(label="💬 Retroalimentación del EcoTransformer", lines=8)
        i_det = gr.Dataframe(label="Detalle de cada objeto")
        i_txt = gr.Textbox(label="Resumen", lines=4)
        i_btn.click(analizar_imagen, i_in, [i_out, i_tabla, i_det, i_txt, i_retro, ultima])

    with gr.Tab("📷 Webcam en vivo"):
        with gr.Row():
            w_in = gr.Image(sources=["webcam"], streaming=True, type="pil", label="Cámara")
            w_out = gr.Image(label="Detección en vivo")
        w_retro = gr.Textbox(label="💬 Retroalimentación", lines=4)
        w_in.stream(webcam, w_in, [w_out, w_retro], stream_every=0.5, concurrency_limit=1)

    with gr.Tab("💬 EcoAsistente"):
        gr.Markdown("El EcoTransformer responde usando lo último que detectó la CNN y tu pregunta.")
        bot = gr.Chatbot(height=420, label="EcoAsistente (Transformer propio)")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ej: ¿Qué hago con el vidrio roto?", scale=5, show_label=False)
            enviar = gr.Button("Enviar", variant="primary", scale=1)
        rec = gr.Button("♻️ Recomendaciones para la última detección")
        # Preguntas de ejemplo: están dentro de los temas que el Transformer aprendió.
        gr.Examples(["¿Qué hago con el vidrio roto?", "¿Dónde boto las pilas usadas?",
                     "¿Cuál es la diferencia entre orgánico e inorgánico?", "¿Cómo hago compost en casa?",
                     "¿Dónde boto una lata de atún?", "¿Qué va en cada caneca?"], msg)
        msg.submit(chat, [msg, bot, ultima], [msg, bot])
        enviar.click(chat, [msg, bot, ultima], [msg, bot])
        rec.click(recomendar, [bot, ultima], bot)

if __name__ == "__main__":
    demo.queue().launch(inbrowser="--no-browser" not in sys.argv, theme=gr.themes.Soft(primary_hue="green"))
