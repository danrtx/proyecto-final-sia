"""Detección de residuos en imágenes y video.

1) Localización de objetos con visión por computador CLÁSICA (OpenCV, sin modelos pre-entrenados):
   - Video: sustracción de fondo (MOG2), que separa lo que se mueve (los residuos en la banda)
     del fondo estático.
   - Imagen / webcam: segmentación por contraste con el color de fondo estimado en los bordes
     de la imagen, más detección de bordes (Canny).
   - Operaciones morfológicas + contornos -> cajas delimitadoras.
2) Clasificación de cada objeto con EcoNet (CNN propia entrenada desde cero).
3) Reglas: orgánico / inorgánico y caneca (Resolución 2184 de 2019).
4) Tracker por IoU para contar cada objeto una sola vez en el video.

FLUJO DE UN CUADRO DE VIDEO
  cuadro -> MOG2 (máscara de lo que se mueve) -> morfología -> contornos -> cajas
         -> recorte de cada caja -> EcoNet (material + confianza) -> tracker (ID del objeto)
         -> dibujo de cajas y etiquetas.
  Al final del video: conteo de objetos únicos -> EcoTransformer (retroalimentación, en app.py).

ORGANIZACIÓN DEL ARCHIVO
  1. Deteccion ......................... un objeto detectado (caja + clase + confianza)
  2. _iou, _contenida, _fusionar ........ utilidades de cajas
  3. _cajas_desde_mascara ............... morfología + contornos
  4. proponer_objetos_imagen ............ localización en fotos y webcam
  5. Tracker ............................ seguimiento por IoU entre cuadros
  6. SistemaResiduos .................... une todo: localización + CNN + dibujo + video
  7. resumen_texto ...................... resumen orgánico / inorgánico
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from config import AREA_MAX, AREA_MIN, CNN_PESOS, CONF_MIN_CNN, INFO_CLASES
from src.cnn import ClasificadorCNN


# =================================================================== 1. UN OBJETO DETECTADO
@dataclass
class Deteccion:
    caja: tuple[int, int, int, int]   # (x1, y1, x2, y2): esquina superior izquierda y esquina inferior derecha
    clase: str                        # material que predijo la CNN
    conf: float                       # confianza de la CNN (probabilidad de esa clase, 0 a 1)
    track_id: int | None = None       # número de seguimiento (#12, ...) en el video

    @property
    def info(self):
        # Nombre, tipo (orgánico/inorgánico), caneca y color: CLASIFICACIÓN ORGÁNICO/INORGÁNICO por regla.
        return INFO_CLASES[self.clase]


# =================================================================== 2. UTILIDADES DE CAJAS
def _iou(a, b):
    """IoU (Intersection over Union) = área de intersección / área de unión de dos cajas.
    0 = no se tocan, 1 = son idénticas. Sirve para saber si dos cajas son el mismo objeto."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _contenida(a, b):
    """Fracción de la caja a que está dentro de b."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area else 0.0


def _fusionar(cajas):
    """Elimina cajas repetidas: de mayor a menor, descarta las que están más del 70 % dentro
    de otra caja ya aceptada o que se solapan mucho con ella (IoU >= 0,4)."""
    cajas = sorted(cajas, key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
    final = []
    for c in cajas:
        if all(_contenida(c, f) < 0.7 and _iou(c, f) < 0.4 for f in final):
            final.append(c)
    return final


# =================================================================== 3. MORFOLOGÍA + CONTORNOS
def _cajas_desde_mascara(mascara, W, H):
    """Convierte una máscara blanco/negro (blanco = objeto) en cajas delimitadoras.
    - APERTURA morfológica: elimina puntos sueltos de ruido.
    - CIERRE morfológico: une las partes de un mismo objeto (rellena huecos pequeños).
    - CONTORNOS: el borde de cada mancha blanca; su rectángulo es la caja del objeto.
    - Se descartan cajas muy pequeñas (< 0,4 % del cuadro) o muy grandes (> 60 %).
    """
    k = max(3, (min(W, H) // 120) | 1)                      # tamaño del kernel según la resolución
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((2 * k, 2 * k), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        a = w * h / (W * H)                                  # fracción del cuadro que ocupa la caja
        if AREA_MIN <= a <= AREA_MAX and min(w, h) > 12:
            cajas.append((x, y, x + w, y + h))
    return _fusionar(cajas)


# =================================================================== 4. LOCALIZACIÓN EN FOTOS
def proponer_objetos_imagen(bgr: np.ndarray):
    """Segmenta objetos contra el fondo estimado en los bordes de la imagen.
    En una foto no hay movimiento, así que se asume que los bordes de la imagen son fondo:
      1. Se calcula el color mediano de los bordes (en espacio de color Lab, más parecido
         a como percibimos el color).
      2. Los píxeles cuyo color se aleja de ese fondo se marcan como objeto.
      3. Se suman los bordes detectados con Canny.
      4. Morfología + contornos -> cajas.
    """
    H, W = bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, min(W, H) // 30)                                    # grosor de la franja de borde
    borde = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    fondo = np.median(borde, axis=0)                               # color típico del fondo
    dist = np.linalg.norm(lab - fondo, axis=2)                     # qué tan distinto es cada píxel del fondo
    umbral = max(18.0, float(np.percentile(np.linalg.norm(borde - fondo, axis=1), 95)) * 1.2)
    mascara = (dist > umbral).astype(np.uint8) * 255
    bordes = cv2.dilate(cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 60, 160), np.ones((3, 3), np.uint8))
    mascara = cv2.bitwise_or(mascara, bordes)
    return _cajas_desde_mascara(mascara, W, H)


# =================================================================== 5. TRACKER
@dataclass
class _Track:
    id: int
    caja: tuple
    votos: Counter = field(default_factory=Counter)  # suma de confianzas por clase
    apariciones: int = 0
    perdido: int = 0


class Tracker:
    """Seguimiento por IoU para no contar el mismo objeto en cada cuadro.

    Un mismo residuo aparece en decenas de cuadros mientras cruza la banda. El tracker
    asocia cada caja nueva con el objeto anterior que más se le solape (IoU > 0,2), le da
    un número fijo (#12, ...) y lo cuenta UNA sola vez. Su clase final es la que acumuló
    más confianza en todos sus cuadros (una "votación" que corrige errores sueltos de la CNN).
    """

    def __init__(self, iou_min=0.2, max_perdido=5):
        self.iou_min, self.max_perdido = iou_min, max_perdido
        self.tracks: list[_Track] = []
        self.finalizados: list[_Track] = []
        self._sig = 1

    def actualizar(self, dets: list[Deteccion]):
        libres = set(range(len(self.tracks)))
        for d in sorted(dets, key=lambda d: -d.conf):
            # Se busca el objeto ya conocido con el que esta caja se solapa más.
            mejor, mejor_iou = None, self.iou_min
            for i in libres:
                v = _iou(d.caja, self.tracks[i].caja)
                if v > mejor_iou:
                    mejor, mejor_iou = i, v
            if mejor is None:
                # No se parece a ninguno: es un objeto NUEVO y recibe un número nuevo.
                t = _Track(self._sig, d.caja); self._sig += 1
                self.tracks.append(t)
            else:
                # Es el mismo objeto de antes: se actualiza su posición.
                t = self.tracks[mejor]; libres.discard(mejor)
                t.caja, t.perdido = d.caja, 0
            t.votos[d.clase] += d.conf      # voto ponderado por la confianza de la CNN
            t.apariciones += 1
            d.track_id = t.id
        # Los objetos que no se vieron en este cuadro suman un "perdido"; tras 5 se dan por terminados.
        for i in libres:
            self.tracks[i].perdido += 1
        vivos = []
        for t in self.tracks:
            (self.finalizados if t.perdido > self.max_perdido else vivos).append(t)
        self.tracks = vivos

    def resumen(self, min_apariciones=3) -> Counter:
        """Conteo final: cada objeto que apareció al menos 3 veces cuenta una vez, con su
        clase más votada. Este conteo es lo que recibe el EcoTransformer."""
        todos = self.finalizados + self.tracks
        return Counter(t.votos.most_common(1)[0][0] for t in todos if t.apariciones >= min_apariciones)


# =================================================================== 6. SISTEMA COMPLETO
class SistemaResiduos:
    def __init__(self, device: str | None = None):
        if not CNN_PESOS.exists():
            raise FileNotFoundError(f"No existe {CNN_PESOS}. Ejecuta notebooks/01_cnn_desde_cero.ipynb")
        self.cnn = ClasificadorCNN(CNN_PESOS, device)    # EcoNet entrenada (models/econet.pt)

    def clasificar_cajas(self, bgr: np.ndarray, cajas) -> list[Deteccion]:
        """UNIÓN LOCALIZACIÓN + CNN: recorta cada caja (con un margen del 6 %), le pasa todos
        los recortes juntos a la CNN y descarta los objetos con confianza menor a 0,40."""
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)        # OpenCV usa BGR; la CNN se entrenó con RGB
        recortes = []
        for x1, y1, x2, y2 in cajas:
            px, py = int(0.06 * (x2 - x1)), int(0.06 * (y2 - y1))
            recortes.append(Image.fromarray(rgb[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]))
        preds = self.cnn.predecir(recortes)
        return [Deteccion(c, clase, conf) for c, (clase, conf, _) in zip(cajas, preds) if conf >= CONF_MIN_CNN]

    def detectar_imagen(self, imagen: Image.Image) -> list[Deteccion]:
        """Fotos y webcam: localización por contraste con el fondo + CNN."""
        bgr = cv2.cvtColor(np.array(imagen.convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        cajas = proponer_objetos_imagen(bgr)
        dets = self.clasificar_cajas(bgr, cajas) if cajas else []
        if not dets:  # un solo objeto que ocupa la imagen: se clasifica la imagen completa
            clase, conf, _ = self.cnn.predecir([imagen])[0]
            dets = [Deteccion((0, 0, W - 1, H - 1), clase, conf)]
        return dets

    # ----------------------------------------------------------------- dibujo
    @staticmethod
    def dibujar(frame_bgr: np.ndarray, dets: list[Deteccion]) -> np.ndarray:
        """MOSTRAR EL RESULTADO: dibuja cada caja con el color de su material y una etiqueta
        "#ID Material (Org./Ino.) confianza", como en la imagen de ejemplo del profesor."""
        img = frame_bgr.copy()
        grosor = max(2, img.shape[1] // 400)
        escala = max(0.5, img.shape[1] / 1600)
        for d in dets:
            x1, y1, x2, y2 = d.caja
            cv2.rectangle(img, (x1, y1), (x2, y2), d.info["color_bgr"], grosor)
            nombre = d.info["nombre"].replace("á", "a").replace("ó", "o")  # OpenCV no dibuja tildes
            texto = f"{nombre} ({d.info['tipo'][:3]}.) {d.conf:.2f}"
            if d.track_id is not None:
                texto = f"#{d.track_id} " + texto
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            y0 = max(th + 6, y1)
            cv2.rectangle(img, (x1, y0 - th - 6), (x1 + tw + 6, y0), (0, 230, 255), -1)   # fondo amarillo de la etiqueta
            cv2.putText(img, texto, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 2)
        return img

    def procesar_imagen(self, imagen: Image.Image):
        dets = self.detectar_imagen(imagen)
        bgr = cv2.cvtColor(np.array(imagen.convert("RGB")), cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(self.dibujar(bgr, dets), cv2.COLOR_BGR2RGB), dets

    # ------------------------------------------------------------------ video
    def procesar_video(self, ruta_in: str, ruta_out: str, cada_n: int = 2,
                       max_segundos: float = 60, ancho_max: int = 960, progreso=None):
        """DETECTAR OBJETOS EN VIDEO (requisito 1 del proyecto).
        cada_n = analizar 1 de cada N cuadros (control de la app): con N = 2 se procesa la
        mitad de los cuadros y en los demás se mantienen las cajas del último análisis.
        Compromiso entre velocidad y precisión: el video demo (500 cuadros) tarda unos 17 s.
        """
        import imageio.v2 as imageio

        cap = cv2.VideoCapture(ruta_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9, fps * max_segundos))
        writer = imageio.get_writer(ruta_out, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        # SUSTRACCIÓN DE FONDO MOG2: modela cada píxel del fondo con una mezcla de gaussianas;
        # lo que se aparta de ese modelo (lo que se mueve) queda blanco en la máscara.
        fondo = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
        tracker, dets, i, t0 = Tracker(), [], 0, time.time()
        while i < total:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > ancho_max:          # videos muy grandes se reducen a 960 px de ancho
                f = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, None, fx=f, fy=f)
            H, W = frame.shape[:2]
            mascara = fondo.apply(frame)            # el modelo de fondo se actualiza en TODOS los cuadros
            if i % cada_n == 0:
                # Los primeros 5 cuadros se usan solo para que MOG2 aprenda el fondo.
                cajas = _cajas_desde_mascara(mascara, W, H) if i > 5 else []
                dets = self.clasificar_cajas(frame, cajas) if cajas else []
                tracker.actualizar(dets)
            writer.append_data(cv2.cvtColor(self.dibujar(frame, dets), cv2.COLOR_BGR2RGB))
            i += 1
            if progreso and i % 10 == 0:
                progreso(i / max(total, 1), desc=f"Procesando cuadro {i}/{total}")
        cap.release(); writer.close()
        return ruta_out, tracker.resumen(), {"cuadros": i, "segundos": time.time() - t0}


# =================================================================== 7. RESUMEN
def resumen_texto(conteo: Counter) -> str:
    """IDENTIFICAR ORGÁNICOS E INORGÁNICOS (requisito 3): resume el conteo por material
    y cuántos objetos son orgánicos y cuántos inorgánicos."""
    if not conteo:
        return "No se detectaron residuos."
    lineas = [f"- {INFO_CLASES[c]['nombre']}: {n} ({INFO_CLASES[c]['tipo']}, caneca {INFO_CLASES[c]['caneca']})"
              for c, n in conteo.most_common()]
    org = sum(n for c, n in conteo.items() if INFO_CLASES[c]["tipo"] == "Orgánico")
    lineas.append(f"Total: {sum(conteo.values())} objetos -> {org} orgánicos, {sum(conteo.values()) - org} inorgánicos.")
    return "\n".join(lineas)
