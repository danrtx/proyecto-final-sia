# ♻️ EcoVisión – Detección y clasificación de residuos con una CNN y un Transformer desde cero

Proyecto final · Sistemas de Inteligencia Artificial · Proyecto 4

Sistema que **detecta residuos en video**, los **clasifica por material** (plástico, papel, cartón,
vidrio, metal y orgánico), indica si son **orgánicos o inorgánicos**, **muestra el resultado** y da
**retroalimentación** sobre cómo separarlos.

**Los dos modelos de IA se diseñaron, programaron y entrenaron desde cero**: no se usan modelos
pre-entrenados, pesos descargados, fine-tuning ni APIs externas.

## Arquitectura

```
 Video / imagen / webcam
          │
          ▼
 ┌──────────────────────────────┐
 │ 1. Localización (OpenCV)     │  Sustracción de fondo MOG2 + contornos
 │    algoritmo clásico         │  (no es un modelo entrenado)
 └──────────────┬───────────────┘
                │ cajas
                ▼
 ┌──────────────────────────────┐
 │ 2. EcoNet (CNN propia)       │  4 bloques convolucionales, 1,17 M parámetros
 │    entrenada desde cero      │  → cartón, metal, orgánico, papel, plástico, vidrio
 └──────────────┬───────────────┘
                │ "plástico ×2, vidrio ×1"
                ▼
 ┌──────────────────────────────┐
 │ 3. Reglas + tracker          │  orgánico/inorgánico, caneca (Res. 2184/2019),
 │                              │  conteo de objetos únicos
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 4. EcoTransformer (propio)   │  Transformer codificador-decodificador programado a mano
 │    entrenado desde cero      │  (atención softmax(QKᵀ/√dk)·V); toma la salida de la CNN
 │                              │  + pregunta y genera la retroalimentación en español
 └──────────────────────────────┘
```

## Resultados

| Modelo | Métrica | Resultado |
|---|---|---|
| EcoNet (CNN) | Exactitud en prueba (6 clases) | **88,7 %** (F1 macro 0,885) |
| EcoNet (CNN) | Exactitud orgánico vs. inorgánico | **97,8 %** |
| EcoTransformer | Respuesta exacta: frases vistas / frases nunca vistas | **100 %** / **74,5 %** |
| EcoTransformer | Caneca correcta con frases nunca vistas | **88,7 %** |

## Estructura

```
FINAL/
├── app.py                               # Aplicación Gradio (video, imagen, webcam, EcoAsistente)
├── config.py                            # Clases, rutas, umbrales
├── prepare_data.py                      # Prepara el dataset (12 → 6 clases, split 70/15/15)
├── notebooks/01_cnn_desde_cero.ipynb    # Entrenamiento y evaluación de EcoNet
├── notebooks/02_transformer_desde_cero.ipynb  # Entrenamiento y evaluación de EcoTransformer
├── src/cnn.py                           # Arquitectura de EcoNet
├── src/transformer.py                   # Transformer programado a mano + tokenizador
├── src/retro_datos.py                   # Dataset propio de retroalimentación
├── src/detector.py                      # Localización OpenCV + CNN + tracker
├── tools/                               # Generadores de notebooks, video demo e informe
├── outputs/                             # Gráficas y videos de demostración
└── informe/                             # Informe final (Word y PDF)
```

## Instalación y uso (Windows, Python 3.12, GPU NVIDIA recomendada)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```

1. Descargar el dataset en `data\raw\`:
   ```powershell
   New-Item -ItemType Directory -Force data\raw; curl.exe -L -o data\raw\garbage.zip "https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip"
   Expand-Archive data\raw\garbage.zip -DestinationPath data\raw
   ```
2. `.venv\Scripts\python prepare_data.py`
3. Registrar el kernel: `.venv\Scripts\python -m ipykernel install --user --name residuos --display-name "Python (residuos)"`
4. Ejecutar `notebooks/01_cnn_desde_cero.ipynb` (≈20 min con GPU, 100 épocas) → genera `models/econet.pt`
5. Ejecutar `notebooks/02_transformer_desde_cero.ipynb` (≈8 min con GPU) → genera `models/ecotransformer.pt`
6. `.venv\Scripts\python app.py` → http://127.0.0.1:7860

Para la demostración, sube `outputs/video_demo_banda.mp4` en la pestaña **Video**.

> **Nota:** `data/` y `models/` no están en el repositorio por su tamaño; se regeneran con los pasos 1–5.

## Cómo colaborar (revisión y correcciones)

1. Haz **fork** de este repositorio en GitHub y clónalo:
   ```powershell
   git clone https://github.com/<tu-usuario>/proyecto-final-sia.git
   cd proyecto-final-sia
   ```
2. Crea una rama para tus cambios: `git checkout -b revision/<tema>`
3. Haz tus correcciones, luego `git add`, `git commit -m "descripción"` y `git push origin revision/<tema>`.
4. En GitHub abre un **Pull Request** hacia `danrtx/proyecto-final-sia` (rama `main`) explicando qué cambiaste.

Para comentar sin cambiar código, abre un **Issue** en el repositorio.

## Dataset

*Garbage Classification* (Mostafa Mohamed, Kaggle; copia en Hugging Face): 15 515 imágenes en 12 clases.
Se usan 8 clases reagrupadas en 6: `cardboard→cartón`, `metal→metal`, `biological→orgánico`,
`paper→papel`, `plastic→plástico`, `brown/green/white-glass→vidrio` (6 576 imágenes en total).
El dataset de retroalimentación del Transformer es **propio** (`src/retro_datos.py`).
