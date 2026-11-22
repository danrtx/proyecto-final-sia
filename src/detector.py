"""Pipeline de visión: detección (OWLv2) + clasificación de material (ViT afinado).

Etapa 1 – OWLv2 (Vision Transformer + encoder de texto, pre-entrenado por Google):
          localiza objetos a partir de consultas en lenguaje natural ("a plastic bottle"...).
Etapa 2 – ViT afinado por nosotros: recorta cada caja y decide el material
          (plástico, papel, cartón, vidrio, metal u orgánico).
Etapa 3 – Reglas: orgánico / inorgánico y caneca según la Res. 2184 de 2019.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import (Owlv2ForObjectDetection, Owlv2Processor,
                          ViTForImageClassification, ViTImageProcessor)

from config import (CONSULTAS_DETECTOR, DETECTOR, INFO_CLASES, PISTA_CONSULTA,
                    UMBRAL_CONFIANZA_VIT, UMBRAL_DETECCION, UMBRAL_NMS, VIT_FINETUNED)


@dataclass
class Deteccion:
    caja: tuple[int, int, int, int]
    score_det: float
    consulta: str
    clase: str
    conf_clase: float
    fuente: str  # "ViT" o "detector"
    track_id: int | None = None

    @property
    def info(self):
        return INFO_CLASES[self.clase]


@dataclass
class _Track:
    id: int
    caja: tuple
    votos: Counter = field(default_factory=Counter)
    perdido: int = 0


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Seguimiento simple por IoU para no contar el mismo objeto en cada cuadro."""

    def __init__(self, iou_min=0.2, max_perdido=5):
        self.iou_min, self.max_perdido = iou_min, max_perdido
        self.tracks: list[_Track] = []
        self.finalizados: list[_Track] = []
        self._sig = 1

    def actualizar(self, dets: list[Deteccion]):
        libres = set(range(len(self.tracks)))
        for d in sorted(dets, key=lambda d: -d.score_det):
            mejor, mejor_iou = None, self.iou_min
            for i in libres:
                v = _iou(d.caja, self.tracks[i].caja)
                if v > mejor_iou:
                    mejor, mejor_iou = i, v
            if mejor is None:
                t = _Track(self._sig, d.caja); self._sig += 1
                self.tracks.append(t)
            else:
                t = self.tracks[mejor]; libres.discard(mejor)
                t.caja, t.perdido = d.caja, 0
            t.votos[d.clase] += 1
            d.track_id = t.id
        for i in libres:
            self.tracks[i].perdido += 1
        vivos = []
        for t in self.tracks:
            (self.finalizados if t.perdido > self.max_perdido else vivos).append(t)
        self.tracks = vivos

    def resumen(self, min_apariciones=3) -> Counter:
        todos = self.finalizados + self.tracks
        return Counter(t.votos.most_common(1)[0][0] for t in todos
                       if sum(t.votos.values()) >= min_apariciones)


class SistemaResiduos:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.dtype = dtype

        self.det_proc = Owlv2Processor.from_pretrained(DETECTOR)
        self.det = Owlv2ForObjectDetection.from_pretrained(DETECTOR, dtype=dtype).to(self.device).eval()

        if not VIT_FINETUNED.exists():
            raise FileNotFoundError(
                f"No existe {VIT_FINETUNED}. Ejecuta primero notebooks/01_finetuning_vit.ipynb")
        self.cls_proc = ViTImageProcessor.from_pretrained(VIT_FINETUNED)
        self.cls = ViTForImageClassification.from_pretrained(VIT_FINETUNED, dtype=dtype).to(self.device).eval()
        self.id2label = self.cls.config.id2label

    # ------------------------------------------------------------------ imagen
    @torch.no_grad()
    def detectar(self, imagen: Image.Image, umbral: float = UMBRAL_DETECCION) -> list[Deteccion]:
        imagen = imagen.convert("RGB")
        W, H = imagen.size
        inp = self.det_proc(text=[CONSULTAS_DETECTOR], images=imagen, return_tensors="pt").to(self.device)
        inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
        out = self.det(**inp)
        lado = max(W, H)  # OWLv2 rellena la imagen hasta hacerla cuadrada
        res = self.det_proc.post_process_grounded_object_detection(
            out, threshold=umbral, target_sizes=torch.tensor([[lado, lado]], device=self.device))[0]
        cajas, scores, etiquetas = res["boxes"].float(), res["scores"].float(), res["labels"]
        if len(cajas) == 0:
            return []

        cajas[:, [0, 2]] = cajas[:, [0, 2]].clamp(0, W)
        cajas[:, [1, 3]] = cajas[:, [1, 3]].clamp(0, H)
        area = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        ok = (area > 0.002 * W * H) & (area < 0.85 * W * H)
        cajas, scores, etiquetas = cajas[ok], scores[ok], etiquetas[ok]
        keep = nms(cajas, scores, UMBRAL_NMS)[:25]
        cajas, scores, etiquetas = cajas[keep], scores[keep], etiquetas[keep]
        # Supresión por contención: descarta cajas que están casi completas dentro de
        # otra (partes de un objeto, p. ej. la boca de una botella o el borde de una caja).
        areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
        lt = torch.max(cajas[:, None, :2], cajas[None, :, :2])
        rb = torch.min(cajas[:, None, 2:], cajas[None, :, 2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        contenida = inter / areas[:, None]          # fracción de la caja i dentro de j
        contenida.fill_diagonal_(0)
        mas_grande = areas[None, :] > areas[:, None]
        ok = ~((contenida > 0.75) & mas_grande).any(1)
        cajas, scores, etiquetas = cajas[ok], scores[ok], etiquetas[ok]
        if len(cajas) == 0:
            return []

        recortes, cajas_int = [], []
        for b in cajas.round().int().tolist():
            x1, y1, x2, y2 = b
            pad_x, pad_y = int(0.05 * (x2 - x1)), int(0.05 * (y2 - y1))
            recortes.append(imagen.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                                         min(W, x2 + pad_x), min(H, y2 + pad_y))))
            cajas_int.append((x1, y1, x2, y2))
        px = self.cls_proc(images=recortes, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        probs = self.cls(pixel_values=px).logits.float().softmax(-1).cpu()

        dets = []
        for caja, s, e, p in zip(cajas_int, scores.tolist(), etiquetas.tolist(), probs):
            consulta = CONSULTAS_DETECTOR[e]
            c = int(p.argmax())
            clase, conf, fuente = self.id2label[c], float(p[c]), "ViT"
            if conf < UMBRAL_CONFIANZA_VIT:  # el ViT duda: usamos la pista del detector
                clase, fuente = PISTA_CONSULTA[consulta], "detector"
                conf = float(p[self.cls.config.label2id[clase]])
            dets.append(Deteccion(caja, float(s), consulta, clase, conf, fuente))
        return dets

    # ----------------------------------------------------------------- dibujo
    @staticmethod
    def dibujar(frame_bgr: np.ndarray, dets: list[Deteccion]) -> np.ndarray:
        img = frame_bgr.copy()
        grosor = max(2, img.shape[1] // 400)
        escala = max(0.5, img.shape[1] / 1600)
        for d in dets:
            x1, y1, x2, y2 = d.caja
            color = d.info["color_bgr"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
            texto = f"{d.info['nombre']} ({d.info['tipo'][:3]}.) {d.conf_clase:.2f}"
            if d.track_id is not None:
                texto = f"#{d.track_id} " + texto
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            y0 = max(th + 6, y1)
            cv2.rectangle(img, (x1, y0 - th - 6), (x1 + tw + 6, y0), (0, 230, 255), -1)
            cv2.putText(img, texto, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 2)
        return img

    def procesar_imagen(self, imagen: Image.Image):
        dets = self.detectar(imagen)
        bgr = cv2.cvtColor(np.array(imagen.convert("RGB")), cv2.COLOR_RGB2BGR)
        anotada = cv2.cvtColor(self.dibujar(bgr, dets), cv2.COLOR_BGR2RGB)
        return anotada, dets

    # ------------------------------------------------------------------ video
    def procesar_video(self, ruta_in: str, ruta_out: str, cada_n: int = 3,
                       max_segundos: float = 60, ancho_max: int = 960, progreso=None):
        import imageio.v2 as imageio

        cap = cv2.VideoCapture(ruta_in)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1e9, fps * max_segundos))
        writer = imageio.get_writer(ruta_out, fps=fps, codec="libx264", quality=7,
                                    macro_block_size=1)
        tracker, dets, i, t0 = Tracker(), [], 0, time.time()
        while i < total:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > ancho_max:
                f = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, None, fx=f, fy=f)
            if i % cada_n == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                dets = self.detectar(pil)
                tracker.actualizar(dets)
            writer.append_data(cv2.cvtColor(self.dibujar(frame, dets), cv2.COLOR_BGR2RGB))
            i += 1
            if progreso and i % 10 == 0:
                progreso(i / max(total, 1), desc=f"Procesando cuadro {i}/{total}")
        cap.release(); writer.close()
        resumen = tracker.resumen()
        return ruta_out, resumen, {"cuadros": i, "segundos": time.time() - t0}


def resumen_texto(conteo: Counter) -> str:
    """Convierte un conteo de clases en texto para la app y para el asistente."""
    if not conteo:
        return "No se detectaron residuos."
    lineas = []
    for clase, n in conteo.most_common():
        inf = INFO_CLASES[clase]
        lineas.append(f"- {inf['nombre']}: {n} ({inf['tipo']}, caneca {inf['caneca']})")
    org = sum(n for c, n in conteo.items() if INFO_CLASES[c]["tipo"] == "Orgánico")
    inorg = sum(conteo.values()) - org
    lineas.append(f"Total: {sum(conteo.values())} objetos -> {org} orgánicos, {inorg} inorgánicos.")
    return "\n".join(lineas)
