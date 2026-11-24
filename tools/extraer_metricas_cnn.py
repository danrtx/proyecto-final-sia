"""Extrae las métricas del notebook 01 ya ejecutado y las guarda en tools/metricas_cnn.json."""
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
nb = json.loads((RAIZ / "notebooks" / "01_cnn_desde_cero.ipynb").read_text(encoding="utf-8"))
texto = "\n".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))

epocas = {}
for m in re.finditer(r"Época\s+(\d+) \| train loss ([\d.]+) acc ([\d.]+) \| val loss ([\d.]+) acc ([\d.]+) F1 ([\d.]+)", texto):
    epocas[int(m[1])] = [float(x) for x in m.groups()[1:]]
reporte = {}
for nombre in ["Cartón", "Metal", "Orgánico", "Papel", "Plástico", "Vidrio", "macro avg"]:
    m = re.search(rf"{nombre}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", texto)
    reporte[nombre] = [float(m[1]), float(m[2]), float(m[3]), int(m[4])]

datos = {
    "epocas_total": max(epocas),
    "epocas": epocas,
    "mejor_epoca": int(re.search(r"Mejor época: (\d+)", texto)[1]),
    "minutos": float(re.search(r"Entrenamiento: ([\d.]+) min", texto)[1]),
    "acc_test": float(re.search(r"Accuracy test: ([\d.]+)", texto)[1]),
    "f1_test": float(re.search(r"F1 macro: ([\d.]+)", texto)[1]),
    "acc_tipo": float(re.search(r"orgánico vs inorgánico: ([\d.]+)", texto)[1]),
    "reporte": reporte,
}
(RAIZ / "tools" / "metricas_cnn.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")
print({k: v for k, v in datos.items() if k != "epocas"})
