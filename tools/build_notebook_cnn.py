"""Genera notebooks/01_cnn_desde_cero.ipynb"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# EcoNet: CNN construida y entrenada desde cero para clasificar residuos

**Proyecto 4 – Detección de residuos y clasificación de basura** · Sistemas de Inteligencia Artificial

En este notebook se **diseña y entrena desde cero** una red neuronal convolucional (CNN): no se usan pesos pre-entrenados ni *fine-tuning*. Todos los pesos parten de valores aleatorios y se aprenden únicamente con nuestro dataset.

La red clasifica cada residuo en 6 clases, y a partir de la clase se determina si es orgánico o inorgánico y en qué caneca va (Resolución 2184 de 2019):

| Clase | Tipo | Caneca |
|---|---|---|
| Plástico, Papel, Cartón, Vidrio, Metal | Inorgánico | Blanca |
| Orgánico | Orgánico | Verde |

### ¿Cómo funciona una CNN?
- **Convolución:** filtros de 3×3 que se deslizan sobre la imagen y detectan patrones locales: bordes y colores en las primeras capas; texturas y formas (la ondulación del cartón, el brillo del metal, la transparencia del vidrio) en las capas profundas.
- **BatchNorm + ReLU:** normalizan y agregan no linealidad para que la red aprenda patrones complejos y entrene de forma estable.
- **MaxPooling:** reduce la resolución a la mitad, lo que da invariancia a pequeños desplazamientos y amplía el campo receptivo.
- **Global Average Pooling + capa lineal:** resume cada mapa de características en un número y decide la clase."""))

C.append(code("""import sys, os, time
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import DATA_PROC, CLASES, INFO_CLASES, CNN_PESOS, OUTPUTS, TAM_IMAGEN
from src.cnn import EcoNet, tf_entrenamiento, tf_evaluacion

torch.manual_seed(42); np.random.seed(42)
OUTPUTS.mkdir(exist_ok=True); CNN_PESOS.parent.mkdir(exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Dispositivo:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')"""))

C.append(md("## 1. Dataset\nDataset público *Garbage Classification (12 clases)*. `prepare_data.py` agrupa las clases en 6 (se unen los 3 tipos de vidrio y *biological* pasa a ser *orgánico*) y divide los datos en 70 % entrenamiento, 15 % validación y 15 % prueba. Las imágenes se cargan en memoria una sola vez para acelerar el entrenamiento."))

C.append(code("""from src.cnn import Residuos   # dataset definido en src/cnn.py (necesario para cargar en paralelo)
t0 = time.time()
ds = {'train': Residuos(DATA_PROC, 'train', tf_entrenamiento), 'val': Residuos(DATA_PROC, 'val', tf_evaluacion), 'test': Residuos(DATA_PROC, 'test', tf_evaluacion)}
for k, d in ds.items():
    conteo = np.bincount([y for _, y in d.items], minlength=len(CLASES))
    print(f'{k:5s} {len(d):5d} imágenes', dict(zip(CLASES, conteo.tolist())))
print(f'Cargado en {time.time()-t0:.1f} s')"""))

C.append(code("""fig, axes = plt.subplots(2, 6, figsize=(16, 6))
rng = np.random.default_rng(0)
for ax, i in zip(axes.flat, rng.choice(len(ds['train']), 12, replace=False)):
    ax.imshow(ds['train'].imgs[i]); ax.set_title(INFO_CLASES[CLASES[ds['train'].items[i][1]]]['nombre']); ax.axis('off')
plt.suptitle('Ejemplos del conjunto de entrenamiento'); plt.tight_layout()
plt.savefig(OUTPUTS / 'ejemplos_dataset.png', dpi=120); plt.show()"""))

C.append(md("### Aumento de datos (*data augmentation*)\nComo la red parte de cero y el dataset es pequeño, en cada época se generan variaciones aleatorias de cada imagen (recortes, giros, rotaciones y cambios de color). Así la red aprende la forma y la textura del material y no memoriza imágenes concretas."))
C.append(code("""fig, axes = plt.subplots(1, 6, figsize=(16, 3))
for ax in axes:
    x, y = ds['train'][5]
    ax.imshow((x.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)); ax.axis('off')
plt.suptitle('Una misma imagen con 6 aumentos aleatorios'); plt.tight_layout()
plt.savefig(OUTPUTS / 'aumento_datos.png', dpi=120); plt.show()"""))

C.append(md("## 2. Arquitectura de EcoNet (pesos aleatorios)"))
C.append(code("""modelo = EcoNet(len(CLASES)).to(device)
print(modelo)
n = sum(p.numel() for p in modelo.parameters())
print(f'\\nParámetros entrenables: {n:,} (todos inicializados aleatoriamente)')
x = torch.zeros(1, 3, TAM_IMAGEN, TAM_IMAGEN, device=device)
for i, b in enumerate(modelo.features):
    x = b(x); print(f'Salida bloque {i+1}: {tuple(x.shape[1:])}')"""))

C.append(md("## 3. Entrenamiento\n- **Función de pérdida:** entropía cruzada con *label smoothing* (0.1) y **pesos por clase**, porque vidrio tiene el doble de imágenes que las demás clases.\n- **Optimizador:** AdamW con política *One-Cycle* (la tasa de aprendizaje sube y luego baja suavemente).\n- **Precisión mixta (FP16)** en GPU para entrenar más rápido.\n- Se entrena durante 100 épocas, cargando y aumentando los datos en 6 procesos en paralelo para que la GPU no espere.\n- Se guarda el modelo con mejor F1 macro en validación."))
C.append(code("""EPOCAS, LOTE = 100, 64
dl_train = DataLoader(ds['train'], batch_size=LOTE, shuffle=True, num_workers=6, persistent_workers=True, drop_last=True)  # 6 núcleos para el aumento de datos
dl_val = DataLoader(ds['val'], batch_size=128, num_workers=0)
dl_test = DataLoader(ds['test'], batch_size=128, num_workers=0)

conteo = np.bincount([y for _, y in ds['train'].items], minlength=len(CLASES))
pesos = torch.tensor(conteo.sum() / (len(CLASES) * conteo), dtype=torch.float32, device=device)
criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.1)
optim = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=2e-3, epochs=EPOCAS, steps_per_epoch=len(dl_train))
scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))

@torch.no_grad()
def evaluar(dl):
    modelo.eval(); ys, ps, perdida = [], [], 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, enabled=(device == 'cuda')):
            out = modelo(x); perdida += criterio(out, y).item() * len(y)
        ys += y.tolist(); ps += out.argmax(-1).tolist()
    return perdida / len(ys), accuracy_score(ys, ps), f1_score(ys, ps, average='macro'), ys, ps

hist, mejor_f1, t0 = [], 0, time.time()
for ep in range(1, EPOCAS + 1):
    modelo.train(); perd, aciertos, total = 0, 0, 0
    for x, y in dl_train:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device, enabled=(device == 'cuda')):
            out = modelo(x); loss = criterio(out, y)
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update(); sched.step()
        perd += loss.item() * len(y); aciertos += (out.argmax(-1) == y).sum().item(); total += len(y)
    vl, va, vf, _, _ = evaluar(dl_val)
    hist.append(dict(epoca=ep, train_loss=perd/total, train_acc=aciertos/total, val_loss=vl, val_acc=va, val_f1=vf))
    marca = ''
    if vf > mejor_f1:
        mejor_f1 = vf; marca = '  <- mejor'
        torch.save({'modelo': modelo.state_dict(), 'clases': CLASES, 'epoca': ep, 'val_f1': vf}, CNN_PESOS)
    print(f'Época {ep:2d} | train loss {perd/total:.3f} acc {aciertos/total:.3f} | val loss {vl:.3f} acc {va:.3f} F1 {vf:.3f}{marca}')
print(f'\\nEntrenamiento: {(time.time()-t0)/60:.1f} min | mejor F1 validación: {mejor_f1:.4f}')"""))

C.append(md("## 4. Curvas de entrenamiento"))
C.append(code("""ep = [h['epoca'] for h in hist]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(ep, [h['train_loss'] for h in hist], label='entrenamiento'); a1.plot(ep, [h['val_loss'] for h in hist], label='validación')
a1.set_xlabel('época'); a1.set_ylabel('pérdida'); a1.set_title('Pérdida'); a1.legend()
a2.plot(ep, [h['train_acc'] for h in hist], label='entrenamiento'); a2.plot(ep, [h['val_acc'] for h in hist], label='validación')
a2.set_xlabel('época'); a2.set_ylabel('accuracy'); a2.set_title('Accuracy'); a2.legend()
plt.tight_layout(); plt.savefig(OUTPUTS / 'curvas_cnn.png', dpi=120); plt.show()"""))

C.append(md("## 5. Evaluación en el conjunto de prueba (imágenes nunca vistas)"))
C.append(code("""ckpt = torch.load(CNN_PESOS, map_location=device); modelo.load_state_dict(ckpt['modelo'])
print('Mejor época:', ckpt['epoca'])
tl, ta, tf1, ys, ps = evaluar(dl_test)
nombres = [INFO_CLASES[c]['nombre'] for c in CLASES]
print(f'Accuracy test: {ta:.4f} | F1 macro: {tf1:.4f}\\n')
print(classification_report(ys, ps, target_names=nombres, digits=3))
tipo = lambda c: INFO_CLASES[CLASES[c]]['tipo']
acc_tipo = np.mean([tipo(a) == tipo(b) for a, b in zip(ys, ps)])
print(f'Accuracy orgánico vs inorgánico: {acc_tipo:.4f}')
cm = confusion_matrix(ys, ps)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=nombres, yticklabels=nombres)
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de confusión – EcoNet (test)')
plt.tight_layout(); plt.savefig(OUTPUTS / 'matriz_confusion_cnn.png', dpi=120); plt.show()"""))

C.append(md("## 6. ¿Qué aprendió la CNN? Filtros y mapas de activación\nLos 32 filtros de la primera capa se aprendieron desde cero. Abajo se ven los mapas de activación de un bloque intermedio: cada mapa muestra dónde detectó la red un patrón distinto."))
C.append(code("""w = modelo.features[0][0].weight.detach().cpu()
w = (w - w.min()) / (w.max() - w.min())
fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for ax, f in zip(axes.flat, w): ax.imshow(f.permute(1, 2, 0)); ax.axis('off')
plt.suptitle('32 filtros 3×3 aprendidos en la primera capa'); plt.tight_layout()
plt.savefig(OUTPUTS / 'filtros_cnn.png', dpi=120); plt.show()

img = ds['test'].imgs[10]
x = tf_evaluacion(img).unsqueeze(0).to(device)
with torch.no_grad():
    act = modelo.features[:2](x)[0].float().cpu()
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
axes[0, 0].imshow(img); axes[0, 0].set_title('Entrada'); axes[0, 0].axis('off')
for k, ax in enumerate(list(axes.flat)[1:]): ax.imshow(act[k], cmap='viridis'); ax.axis('off')
plt.suptitle('Mapas de activación del bloque 2'); plt.tight_layout()
plt.savefig(OUTPUTS / 'activaciones_cnn.png', dpi=120); plt.show()"""))

C.append(md("## 7. Predicciones de ejemplo"))
C.append(code("""fig, axes = plt.subplots(2, 6, figsize=(16, 6))
idx = np.random.default_rng(7).choice(len(ds['test']), 12, replace=False)
modelo.eval()
for ax, i in zip(axes.flat, idx):
    x, y = ds['test'][int(i)]
    with torch.no_grad(): p = modelo(x.unsqueeze(0).to(device)).softmax(-1)[0]
    c = int(p.argmax()); ok = c == y
    ax.imshow(ds['test'].imgs[int(i)]); ax.axis('off')
    ax.set_title(f"{INFO_CLASES[CLASES[c]]['nombre']} ({p[c]:.2f})\\nreal: {INFO_CLASES[CLASES[y]]['nombre']}", color='green' if ok else 'red', fontsize=10)
plt.tight_layout(); plt.savefig(OUTPUTS / 'predicciones_cnn.png', dpi=120); plt.show()"""))

nb = nbf.v4.new_notebook(); nb['cells'] = C
nb.metadata['kernelspec'] = {'name': 'residuos', 'display_name': 'Python (residuos)', 'language': 'python'}
out = Path(__file__).resolve().parents[1] / 'notebooks' / '01_cnn_desde_cero.ipynb'
nbf.write(nb, out); print('Notebook creado:', out)
