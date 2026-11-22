# ♻️ EcoVisión – Detección y clasificación de residuos con Transformers

Proyecto final · Sistemas de Inteligencia Artificial · Proyecto 4

Sistema inteligente que **detecta residuos en video**, los **clasifica por material**
(plástico, papel, cartón, vidrio, metal), los separa en **orgánicos / inorgánicos**,
**muestra el resultado** y ofrece un **asistente** que da recomendaciones de separación.

Todos los modelos son **Transformers pre-entrenados que corren localmente**: no se usa ninguna API externa.

## Arquitectura

```
 Video / imagen / webcam
          │  (cuadro a cuadro, OpenCV)
          ▼
 ┌──────────────────────────────┐
 │ 1. OWLv2 (Google)            │  Vision Transformer + encoder de texto (tipo CLIP)
 │    Detección de objetos      │  pre-entrenado; detecta a partir de consultas
 │    "open-vocabulary"         │  como "a plastic bottle", "a banana peel"...
 └──────────────┬───────────────┘
                │ cajas (bounding boxes)
                ▼
 ┌──────────────────────────────┐
 │ 2. ViT afinado (nuestro)     │  google/vit-base-patch16-224-in21k + fine-tuning
 │    Clasificación de material │  → cartón, metal, orgánico, papel, plástico, vidrio
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 3. Reglas + tracker IoU      │  orgánico/inorgánico, caneca (Res. 2184/2019),
 │                              │  conteo de objetos únicos en el video
 └──────────────┬───────────────┘
                ▼
 ┌──────────────────────────────┐
 │ 4. EcoAsistente              │  Qwen2.5-1.5B-Instruct (Transformer decoder, tipo GPT)
 │    LLM local + base de       │  + recuperación de conocimiento (RAG simple)
 │    conocimiento              │  + resultado de la detección
 └──────────────────────────────┘
```

| Tipo de Transformer (clase 7) | Modelo usado | Rol |
|---|---|---|
| Solo encoder (como BERT) | ViT afinado | Clasificar material |
| Encoder visión + texto | OWLv2 | Detectar objetos en el video |
| Solo decoder (como GPT) | Qwen2.5-1.5B-Instruct | Asistente de recomendaciones |

## Estructura

```
FINAL/
├── app.py                         # Aplicación Gradio (video, imagen, webcam, asistente)
├── config.py                      # Clases, modelos, consultas, umbrales
├── prepare_data.py                # Prepara el dataset (12 → 6 clases, split 70/15/15)
├── notebooks/01_finetuning_vit.ipynb  # Entrenamiento, métricas, matriz de confusión, atención
├── src/detector.py                # OWLv2 + ViT + tracker
├── src/asistente.py               # LLM local + base de conocimiento
├── outputs/                       # Gráficas generadas por el notebook
└── models/vit-residuos/           # Modelo afinado (se genera al entrenar)
```

## Instalación y uso (Windows, Python 3.12, GPU NVIDIA)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -m pip install -r requirements.txt
```

1. Descargar el dataset *Garbage Classification (12 clases)* en `data/raw/garbage_classification/`
   (Hugging Face: `UdaraChamidu/Garbage-Classification-with-12-classes`).
2. `.venv\Scripts\python prepare_data.py`
3. Ejecutar `notebooks/01_finetuning_vit.ipynb` (≈10–15 min en una RTX 4050).
4. `.venv\Scripts\python app.py` → se abre http://127.0.0.1:7860

La primera vez se descargan los pesos pre-entrenados de OWLv2 y Qwen desde Hugging Face.
Después, todo funciona sin internet.

> **Nota:** las carpetas `data/` y `models/` no están en el repositorio por su tamaño (~6 GB).
> Se regeneran con los pasos 1–3 (el entrenamiento tarda ≈5 min con GPU).

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

*Garbage Classification* (Mostafa Mohamed, Kaggle; copia en Hugging Face), con 15 515 imágenes en 12 clases.
Se usan 8 de esas clases, reagrupadas en 6:
`cardboard→cartón`, `metal→metal`, `biological→orgánico`, `paper→papel`, `plastic→plástico`,
`brown/green/white-glass→vidrio`, con un máximo de 1000 imágenes por clase (≈5 500 en total).
