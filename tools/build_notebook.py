"""Genera notebooks/01_finetuning_vit.ipynb (se ejecuta luego con nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
C = []

C.append(md("""# Fine-tuning de un Vision Transformer (ViT) para clasificar residuos

**Proyecto 4 – Detección de residuos y clasificación de basura** · Sistemas de Inteligencia Artificial

Este notebook toma un **Transformer pre-entrenado** (`google/vit-base-patch16-224-in21k`, entrenado por Google en ImageNet-21k con 14 millones de imágenes) y lo **afina (fine-tuning)** para clasificar residuos en 6 clases:

| Clase | Tipo | Caneca (Res. 2184/2019) |
|---|---|---|
| Plástico, Papel, Cartón, Vidrio, Metal | Inorgánico (aprovechable) | Blanca |
| Orgánico | Orgánico | Verde |

### ¿Por qué un Transformer para imágenes?
El ViT aplica la misma idea del artículo *"Attention is All You Need"* (2017) que vimos en NLP:
la imagen de 224×224 se parte en **196 parches de 16×16**, y cada parche se trata como si fuera un **token**
(como una palabra en una frase). Se le suma un *embedding* de posición y se agrega un token especial `[CLS]`
(igual que en **BERT**). Luego, el **encoder** calcula *self-attention*:

$$\\text{Attention}(Q,K,V)=\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

Así, cada parche "mira" a todos los demás parches, y el vector final del token `[CLS]` se usa para clasificar.
Es una arquitectura **solo-encoder**, igual que BERT, pero aplicada a imágenes."""))

C.append(code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import DATA_PROC, VIT_BASE, VIT_FINETUNED, CLASES, INFO_CLASES, RAIZ

OUT = RAIZ / 'outputs'; OUT.mkdir(exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Dispositivo:', device, torch.cuda.get_device_name(0) if device == 'cuda' else '')"""))

C.append(md("## 1. Dataset\nDataset público *Garbage Classification (12 clases)*. Las 12 clases originales se agrupan en 6 con `prepare_data.py` (se unen los 3 tipos de vidrio y *biological* pasa a ser *orgánico*), se balancean (máximo 1000 imágenes por clase) y se dividen en 70% entrenamiento, 15% validación y 15% prueba."))

C.append(code("""ds = load_dataset('imagefolder', data_dir=str(DATA_PROC))
labels = ds['train'].features['label'].names
assert labels == CLASES, labels
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in id2label.items()}
print(ds)
for split in ds:
    conteo = np.bincount(ds[split]['label'], minlength=len(labels))
    print(split, dict(zip(labels, conteo.tolist())))"""))

C.append(code("""fig, axes = plt.subplots(2, 6, figsize=(16, 6))
rng = np.random.default_rng(0)
for ax, i in zip(axes.flat, rng.choice(len(ds['train']), 12, replace=False)):
    ej = ds['train'][int(i)]
    ax.imshow(ej['image']); ax.set_title(INFO_CLASES[labels[ej['label']]]['nombre']); ax.axis('off')
plt.suptitle('Ejemplos del conjunto de entrenamiento'); plt.tight_layout()
plt.savefig(OUT / 'ejemplos_dataset.png', dpi=120); plt.show()"""))

C.append(md("## 2. Pre-procesamiento y *data augmentation*\nSe usa el `ViTImageProcessor` del modelo pre-entrenado (que define el tamaño y la normalización). En entrenamiento se aplican recortes, giros y cambios de color aleatorios para que el modelo generalice mejor."))

C.append(code("""processor = ViTImageProcessor.from_pretrained(VIT_BASE)
size = processor.size['height']
norm = T.Normalize(mean=processor.image_mean, std=processor.image_std)
tf_train = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
tf_eval = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), norm])

def aplicar(tf):
    def f(batch):
        batch['pixel_values'] = [tf(im.convert('RGB')) for im in batch['image']]
        return batch
    return f

ds['train'].set_transform(aplicar(tf_train))
ds['validation'].set_transform(aplicar(tf_eval))
ds['test'].set_transform(aplicar(tf_eval))

def collate(batch):
    return {'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
            'labels': torch.tensor([b['label'] for b in batch])}"""))

C.append(md("## 3. Modelo pre-entrenado + nueva cabeza de clasificación\nSe cargan los pesos pre-entrenados del encoder (12 capas, 12 cabezas de atención, 86M de parámetros) y se agrega una capa lineal nueva de 6 salidas."))

C.append(code("""model = ViTForImageClassification.from_pretrained(
    VIT_BASE, num_labels=len(labels), id2label=id2label, label2id=label2id, dtype=torch.float32)
total = sum(p.numel() for p in model.parameters())
print(f'Parámetros: {total/1e6:.1f} M | capas: {model.config.num_hidden_layers} | cabezas de atención: {model.config.num_attention_heads}')"""))

C.append(code("""def metricas(p):
    pred = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, pred),
            'f1_macro': f1_score(p.label_ids, pred, average='macro')}

args = TrainingArguments(
    output_dir=str(RAIZ / 'models' / 'checkpoints'),
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    num_train_epochs=5, learning_rate=5e-5, weight_decay=0.01, warmup_steps=0.1,
    lr_scheduler_type='cosine', fp16=(device == 'cuda'),
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model='f1_macro',
    logging_steps=20, remove_unused_columns=False, dataloader_num_workers=0,
    report_to='none', seed=42)

trainer = Trainer(model=model, args=args, train_dataset=ds['train'], eval_dataset=ds['validation'],
                  data_collator=collate, compute_metrics=metricas)
resultado = trainer.train()
print(resultado)"""))

C.append(md("## 4. Curvas de entrenamiento"))
C.append(code("""h = trainer.state.log_history
tr = [(x['epoch'], x['loss']) for x in h if 'loss' in x]
ev = [(x['epoch'], x['eval_loss'], x['eval_accuracy']) for x in h if 'eval_loss' in x]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(*zip(*tr), label='train'); a1.plot([e[0] for e in ev], [e[1] for e in ev], 'o-', label='validación')
a1.set_xlabel('época'); a1.set_ylabel('pérdida'); a1.legend(); a1.set_title('Pérdida (cross-entropy)')
a2.plot([e[0] for e in ev], [e[2] for e in ev], 'o-', color='green')
a2.set_xlabel('época'); a2.set_ylabel('accuracy'); a2.set_title('Accuracy en validación'); a2.set_ylim(0.5, 1)
plt.tight_layout(); plt.savefig(OUT / 'curvas_entrenamiento.png', dpi=120); plt.show()"""))

C.append(md("## 5. Evaluación en el conjunto de prueba (imágenes nunca vistas)"))
C.append(code("""pred = trainer.predict(ds['test'])
y_true, y_pred = pred.label_ids, pred.predictions.argmax(-1)
nombres = [INFO_CLASES[l]['nombre'] for l in labels]
print(f"Accuracy test: {accuracy_score(y_true, y_pred):.4f} | F1 macro: {f1_score(y_true, y_pred, average='macro'):.4f}\\n")
print(classification_report(y_true, y_pred, target_names=nombres, digits=3))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=nombres, yticklabels=nombres)
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de confusión – test')
plt.tight_layout(); plt.savefig(OUT / 'matriz_confusion.png', dpi=120); plt.show()"""))

C.append(md("## 6. Guardar el modelo afinado\nEl modelo queda guardado localmente y la aplicación lo usa sin ninguna API externa."))
C.append(code("""trainer.save_model(str(VIT_FINETUNED))
processor.save_pretrained(str(VIT_FINETUNED))
print('Guardado en', VIT_FINETUNED)"""))

C.append(md("## 7. ¿Dónde \"mira\" el Transformer? – Mapa de atención\nSe visualiza la atención del token `[CLS]` hacia los 196 parches, promediando todas las cabezas y combinando las 12 capas (*attention rollout*). Las zonas claras son las que más usó el modelo para decidir."))
C.append(code("""vis = ViTForImageClassification.from_pretrained(str(VIT_FINETUNED), attn_implementation='eager').to(device).eval()
test_raw = load_dataset('imagefolder', data_dir=str(DATA_PROC), split='test')

def rollout(atts):
    res = torch.eye(atts[0].size(-1), device=atts[0].device)
    for a in atts:
        a = a.mean(1)[0]
        a = a + torch.eye(a.size(-1), device=a.device); a = a / a.sum(-1, keepdim=True)
        res = a @ res
    m = res[0, 1:].reshape(14, 14).cpu().numpy()
    return (m - m.min()) / (m.max() - m.min())

rng = np.random.default_rng(3)
idx = rng.choice(len(test_raw), 6, replace=False)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for k, i in enumerate(idx):
    im = test_raw[int(i)]['image'].convert('RGB')
    x = tf_eval(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out = vis(pixel_values=x, output_attentions=True)
    p = out.logits.softmax(-1)[0]; c = int(p.argmax())
    m = rollout(out.attentions)
    im224 = T.CenterCrop(224)(T.Resize(224)(im))
    axes[0, k].imshow(im224); axes[0, k].axis('off')
    axes[0, k].set_title(f"Real: {INFO_CLASES[labels[test_raw[int(i)]['label']]]['nombre']}")
    axes[1, k].imshow(im224); axes[1, k].imshow(np.kron(m, np.ones((16, 16))), alpha=0.55, cmap='jet'); axes[1, k].axis('off')
    axes[1, k].set_title(f"Pred: {INFO_CLASES[labels[c]]['nombre']} ({p[c]:.2f})")
plt.tight_layout(); plt.savefig(OUT / 'mapas_atencion.png', dpi=120); plt.show()"""))

nb['cells'] = C
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = Path(__file__).resolve().parents[1] / 'notebooks' / '01_finetuning_vit.ipynb'
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print('Notebook creado:', out)
