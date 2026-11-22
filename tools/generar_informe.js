// Genera informe/Informe_Proyecto_Final_EcoVision.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
  Footer, PageNumber, PageBreak, TableOfContents,
} = require("docx");

const RAIZ = path.resolve(__dirname, "..");
const OUT = path.join(RAIZ, "outputs");
const FONT = "Arial";
const VERDE = "1E7B4A";
const ANCHO = 9360; // ancho útil carta con márgenes de 1"

// ---------------------------------------------------------------- helpers
const p = (texto, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 300 }, alignment: opts.align ?? AlignmentType.JUSTIFIED,
  ...opts.par,
  children: (Array.isArray(texto) ? texto : [texto]).map(t =>
    typeof t === "string" ? new TextRun({ text: t, ...opts.run }) : t),
});
const b = t => new TextRun({ text: t, bold: true });
const i = t => new TextRun({ text: t, italics: true });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const vineta = (texto, nivel = 0) => new Paragraph({
  numbering: { reference: "vinetas", level: nivel }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(texto) ? texto : [texto]).map(t => typeof t === "string" ? new TextRun(t) : t),
});
const numerada = (texto, ref = "numeros") => new Paragraph({
  numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 288 },
  children: (Array.isArray(texto) ? texto : [texto]).map(t => typeof t === "string" ? new TextRun(t) : t),
});

let nFig = 0, nTab = 0;
function figura(archivo, anchoPx, altoPx, pie, anchoMax = 6.3) {
  nFig++;
  const ancho = Math.min(anchoMax, 6.5) * 96;
  const alto = ancho * altoPx / anchoPx;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: archivo.endsWith(".png") ? "png" : "jpg",
        data: fs.readFileSync(path.join(OUT, archivo)),
        transformation: { width: Math.round(ancho), height: Math.round(alto) },
        altText: { title: pie, description: pie, name: archivo } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figura ${nFig}. `, bold: true, size: 20 }), new TextRun({ text: pie, italics: true, size: 20 })] }),
  ];
}

const borde = { style: BorderStyle.SINGLE, size: 4, color: "B7C9BE" };
const bordes = { top: borde, bottom: borde, left: borde, right: borde };
function tabla(titulo, encabezados, filas, anchos) {
  nTab++;
  const total = anchos.reduce((a, c) => a + c, 0);
  const celda = (txt, j, enc, k) => new TableCell({
    width: { size: anchos[j], type: WidthType.DXA }, borders: bordes,
    shading: { type: ShadingType.CLEAR, color: "auto", fill: enc ? VERDE : (k % 2 ? "F1F7F3" : "FFFFFF") },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(txt), bold: enc, color: enc ? "FFFFFF" : "222222", size: 19 })] })],
  });
  return [
    new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Tabla ${nTab}. `, bold: true, size: 20 }), new TextRun({ text: titulo, italics: true, size: 20 })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: anchos,
      rows: [new TableRow({ tableHeader: true, children: encabezados.map((e, j) => celda(e, j, true, 0)) }),
        ...filas.map((f, k) => new TableRow({ children: f.map((c, j) => celda(c, j, false, k)) }))],
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ---------------------------------------------------------------- portada
const portada = [
  new Paragraph({ spacing: { before: 1800 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "[NOMBRE DE LA UNIVERSIDAD]", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1000 },
    children: [new TextRun({ text: "Sistemas de Inteligencia Artificial", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "EcoVisión", bold: true, size: 56, color: VERDE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "Sistema inteligente de detección y clasificación de residuos basado en Transformers pre-entrenados", size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1400 },
    children: [new TextRun({ text: "Proyecto final – Proyecto 4: Detección de residuos y clasificación de basura", italics: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Presentado por:", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "Danilo Andrés Montezuma Ibarra", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "Docente:", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({ text: "PhD Cristian Camilo Ordoñez", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Noviembre de 2026", size: 22 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------------------------------------------------------- contenido
const C = [];

C.push(new Paragraph({ spacing: { before: 120, after: 200 }, children: [new TextRun({ text: "Tabla de contenido", bold: true, size: 30, color: VERDE })] }),
  new TableOfContents("Tabla de contenido", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

C.push(h1("Resumen"),
  p(["Este proyecto presenta ", b("EcoVisión"), ", un sistema de inteligencia artificial que detecta residuos en video, los clasifica por material (plástico, papel, cartón, vidrio y metal), los separa en orgánicos e inorgánicos, muestra el resultado en pantalla y ofrece un asistente conversacional con recomendaciones de separación. Todo el sistema se basa en ", b("modelos Transformer pre-entrenados"), " que se ejecutan localmente, sin depender de APIs externas."]),
  p(["La solución combina tres tipos de Transformer: (1) ", b("OWLv2"), ", un detector de objetos de vocabulario abierto basado en Vision Transformer, que localiza los objetos en cada cuadro del video; (2) un ", b("Vision Transformer (ViT)"), " pre-entrenado en ImageNet-21k al que se le aplicó ", i("fine-tuning"), " con 5 510 imágenes de residuos y que alcanzó una ", b("exactitud del 97,0 % y un F1 macro de 0,969"), " en el conjunto de prueba; y (3) ", b("Qwen2.5-1.5B-Instruct"), ", un modelo de lenguaje tipo GPT (solo decodificador) que, apoyado en una base de conocimiento propia sobre la Resolución 2184 de 2019, responde en español las preguntas del usuario."]),
  p([b("Palabras clave: "), "Transformer, mecanismo de atención, Vision Transformer, detección de objetos, fine-tuning, LLM, reciclaje, clasificación de residuos."]));

C.push(h1("1. Introducción"),
  p("La separación de residuos en la fuente es uno de los pasos más importantes para que los materiales aprovechables lleguen a las cadenas de reciclaje. En Colombia, la Resolución 2184 de 2019 estableció un código de colores unificado (caneca blanca para aprovechables, verde para orgánicos y negra para no aprovechables). En la práctica, muchas personas no saben en qué caneca va cada residuo, y en las plantas de aprovechamiento la clasificación sigue haciéndose a mano sobre bandas transportadoras."),
  p("La arquitectura Transformer, presentada en el artículo “Attention is All You Need” (Vaswani et al., 2017), revolucionó primero el procesamiento de lenguaje natural (BERT, GPT, T5) y luego la visión por computador (Vision Transformer). Este proyecto aplica esa misma arquitectura al problema de la clasificación de residuos y aprovecha el aprendizaje por transferencia: se parte de modelos que ya aprendieron representaciones generales con millones de datos y se adaptan a la tarea específica."));

C.push(h2("1.1 Objetivo general"),
  p("Construir un sistema inteligente basado en modelos Transformer pre-entrenados que identifique y clasifique residuos para facilitar los procesos de reciclaje."),
  h2("1.2 Objetivos específicos"),
  numerada("Detectar los objetos presentes en un video."),
  numerada("Clasificar los residuos en plástico, papel, cartón, vidrio y metal."),
  numerada("Identificar si cada residuo es orgánico o inorgánico."),
  numerada("Mostrar el resultado de la clasificación de forma visual e interpretable."),
  numerada("Crear un asistente que brinde recomendaciones sobre la separación de residuos."));

C.push(h1("2. Marco teórico"),
  h2("2.1 Arquitectura Transformer y mecanismo de atención"),
  p("El Transformer reemplaza las redes recurrentes (RNN) y convolucionales (CNN) por un mecanismo de autoatención (self-attention), que permite que cada elemento de una secuencia se relacione directamente con todos los demás, sin importar la distancia entre ellos, y que el cálculo se haga en paralelo. Cada token se proyecta en tres vectores: consulta (Query, Q), clave (Key, K) y valor (Value, V). La atención se calcula como:"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
    children: [new TextRun({ text: "Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V", italics: true, size: 26 })] }),
  p("donde Q·Kᵀ mide la similitud entre consultas y claves, √dₖ normaliza según la dimensión de las claves y softmax convierte los puntajes en una distribución de probabilidad. La atención multi-cabeza repite este cálculo en paralelo con distintas proyecciones para capturar diferentes tipos de relaciones."),
  p("Según cómo se combinen sus bloques, hay tres familias de modelos: solo codificador (encoder, como BERT), especializados en comprender y clasificar; solo decodificador (decoder, como GPT), especializados en generar texto de forma autorregresiva; y codificador-decodificador (como T5 o BART). Este proyecto usa modelos de las dos primeras familias, además de un modelo multimodal de visión y texto."));

C.push(h2("2.2 Vision Transformer (ViT)"),
  p("El Vision Transformer (Dosovitskiy et al., 2021) aplica el Transformer directamente a imágenes. Una imagen de 224×224 píxeles se divide en 196 parches de 16×16, y cada parche se aplana y se proyecta como un “token”, igual que una palabra en una frase. Se suman embeddings de posición y se antepone un token especial [CLS], como en BERT. Tras 12 capas de autoatención, el vector final del token [CLS] alimenta la capa de clasificación. Es, por tanto, una arquitectura solo-codificador aplicada a visión."));

C.push(h2("2.3 Detección de objetos con OWLv2"),
  p("OWLv2 (Minderer et al., 2023), de Google, es un detector de vocabulario abierto (open-vocabulary). Combina un codificador de imagen ViT con un codificador de texto tipo CLIP, de modo que puede localizar objetos descritos con frases en lenguaje natural (por ejemplo, “a plastic bottle” o “a banana peel”) sin reentrenarse para cada clase. Por cada parche de la imagen predice una caja delimitadora y su similitud con cada consulta de texto."));

C.push(h2("2.4 Modelos de lenguaje grandes (LLM) y recuperación de información"),
  p("Un LLM es un Transformer solo-decodificador entrenado con grandes cantidades de texto para predecir el siguiente token. Qwen2.5-1.5B-Instruct, de Alibaba, tiene 1 540 millones de parámetros y fue ajustado para seguir instrucciones en varios idiomas, incluido el español. Para reducir las “alucinaciones” se usa una técnica de generación aumentada por recuperación (RAG): antes de responder, se buscan en una base de conocimiento los fragmentos relevantes para la pregunta y se incluyen en el contexto del modelo."));

C.push(h2("2.5 Aprendizaje por transferencia y fine-tuning"),
  p("Entrenar un Transformer desde cero exige millones de ejemplos y mucho cómputo. Con el aprendizaje por transferencia se parte de pesos pre-entrenados y se ajustan (fine-tuning) con un conjunto de datos pequeño y específico. Así, el modelo reutiliza lo que ya sabe de bordes, texturas y formas, y solo aprende a distinguir las nuevas clases."));

// --------------------------------------------------------------- metodología
C.push(h1("3. Metodología"),
  h2("3.1 Arquitectura general del sistema"),
  p("El sistema procesa el video cuadro por cuadro en cuatro etapas:"),
  ...tabla("Etapas del sistema EcoVisión.",
    ["Etapa", "Modelo / técnica", "Tipo de Transformer", "Función"],
    [["1. Detección", "OWLv2 (google/owlv2-base-patch16-ensemble)", "ViT + codificador de texto", "Localiza los objetos en cada cuadro del video"],
     ["2. Clasificación", "ViT afinado (vit-base-patch16-224-in21k)", "Solo codificador", "Asigna el material a cada objeto recortado"],
     ["3. Reglas y seguimiento", "Mapeo de clases + tracker por IoU", "—", "Orgánico/inorgánico, caneca y conteo de objetos únicos"],
     ["4. Asistente", "Qwen2.5-1.5B-Instruct + RAG", "Solo decodificador", "Recomendaciones de separación en español"]],
    [1600, 2900, 2000, 2860]),
  p(["Todos los modelos se descargan una sola vez desde Hugging Face y se ejecutan en el equipo local (GPU NVIDIA RTX 4050 de 6 GB) con las bibliotecas ", i("PyTorch"), " y ", i("transformers"), ". La interfaz se construyó con ", i("Gradio"), " y el video se procesa con ", i("OpenCV"), "."]));

C.push(h2("3.2 Conjunto de datos"),
  p(["Se usó el conjunto público ", i("Garbage Classification"), " (12 clases, 15 515 imágenes). Las 8 clases pertinentes se reagruparon en las 6 requeridas: se unieron los tres tipos de vidrio (café, verde y blanco) y la clase ", i("biological"), " se convirtió en ", i("orgánico"), ". Para evitar desbalance, se tomaron como máximo 1 000 imágenes por clase y se dividieron de forma estratificada en 70 % entrenamiento, 15 % validación y 15 % prueba."]),
  ...tabla("Distribución del conjunto de datos por clase.",
    ["Clase", "Clase(s) original(es)", "Entrenamiento", "Validación", "Prueba", "Total"],
    [["Cartón", "cardboard", 623, 134, 134, 891], ["Metal", "metal", 538, 115, 116, 769],
     ["Orgánico", "biological", 689, 148, 148, 985], ["Papel", "paper", 700, 150, 150, 1000],
     ["Plástico", "plastic", 605, 130, 130, 865], ["Vidrio", "brown/green/white-glass", 700, 150, 150, 1000],
     ["Total", "", 3855, 827, 828, 5510]],
    [1300, 2560, 1500, 1300, 1300, 1400]),
  ...figura("ejemplos_dataset.png", 1920, 720, "Ejemplos del conjunto de entrenamiento."));

C.push(h2("3.3 Fine-tuning del Vision Transformer"),
  p(["Se partió de ", i("google/vit-base-patch16-224-in21k"), " (85,8 millones de parámetros, 12 capas y 12 cabezas de atención), pre-entrenado con 14 millones de imágenes de ImageNet-21k. Se reemplazó la cabeza de clasificación por una capa lineal de 6 salidas y se entrenó toda la red. Durante el entrenamiento se aplicó aumento de datos (recortes aleatorios, giros horizontales, rotaciones de ±15° y variaciones de brillo, contraste y saturación) para mejorar la generalización."]),
  ...tabla("Hiperparámetros del entrenamiento.",
    ["Parámetro", "Valor"],
    [["Épocas", "5"], ["Tamaño de lote (batch)", "32"], ["Tasa de aprendizaje", "5×10⁻⁵ con calentamiento (10 %) y decaimiento coseno"],
     ["Optimizador", "AdamW, weight decay 0,01"], ["Precisión", "Mixta (FP16)"], ["Función de pérdida", "Entropía cruzada"],
     ["Selección del modelo", "Mejor F1 macro en validación"], ["Tiempo de entrenamiento", "5 min 20 s (RTX 4050)"]],
    [3600, 5760]));

C.push(h2("3.4 Detección y clasificación en video"),
  p("Para cada cuadro analizado se ejecuta el siguiente procedimiento:"),
  numerada(["OWLv2 recibe la imagen y 20 consultas de texto (por ejemplo, “a plastic bottle”, “an aluminum can”, “a cardboard box”, “food waste”) y devuelve cajas con un puntaje de confianza; se conservan las que superan 0,20."], "pasos"),
  numerada(["Se filtran las cajas: supresión de no máximos (NMS, IoU 0,4), eliminación de cajas muy pequeñas o que cubren casi toda la imagen, y una ", b("supresión por contención"), " que descarta cajas que están más del 75 % dentro de otra (por ejemplo, la boca de una botella)."], "pasos"),
  numerada(["Cada caja se recorta y el ViT afinado predice su material. Si la confianza del ViT es menor que 0,55, se usa como respaldo el material sugerido por la consulta de texto del detector."], "pasos"),
  numerada(["Con una tabla de reglas se asigna el tipo (orgánico o inorgánico) y la caneca según la Resolución 2184 de 2019, y se dibuja la caja con su etiqueta."], "pasos"),
  numerada(["Un tracker por IoU asocia las detecciones entre cuadros y asigna un identificador a cada objeto, de modo que cada residuo se cuente una sola vez. Un objeto se cuenta cuando aparece en al menos 3 análisis."], "pasos"),
  p("Para acelerar el proceso se analiza 1 de cada N cuadros (N configurable; por defecto 3) y las cajas se mantienen en los cuadros intermedios."));

C.push(h2("3.5 Asistente de recomendaciones (EcoAsistente)"),
  p("El asistente usa Qwen2.5-1.5B-Instruct cargado localmente en FP16. Su contexto se arma con tres fuentes:"),
  vineta([b("Instrucciones del sistema: "), "rol de experto en reciclaje en Colombia, respuestas breves en español y la regla de no inventar procedimientos."]),
  vineta([b("Recuperación (RAG): "), "una base de conocimiento propia con 15 fragmentos (botellas, vidrio roto, cartón, Tetra Pak, residuos peligrosos, compostaje, etc.). Por palabras clave se seleccionan los 3 fragmentos más relevantes para la pregunta."]),
  vineta([b("Resultado de la visión: "), "el resumen de lo detectado en la última imagen o video, para que el asistente pueda responder sobre esos residuos."]),
  p("Se usa decodificación determinista (greedy) para que las respuestas sean estables. El botón “Recomendaciones para la última detección” genera además una lista por material, construida directamente desde la tabla de reglas, y el LLM añade un consejo breve al final; así las indicaciones principales siempre son correctas."));

C.push(h2("3.6 Interfaz de usuario"),
  p("La aplicación web local (Gradio) tiene cuatro pestañas:"),
  vineta([b("Video: "), "se carga un video y se obtiene el video anotado, una tabla de residuos únicos por material, el tipo, la caneca y una recomendación."]),
  vineta([b("Imagen: "), "el mismo análisis sobre una foto, con el detalle de cada objeto."]),
  vineta([b("Webcam en vivo: "), "detección en tiempo real desde la cámara."]),
  vineta([b("EcoAsistente: "), "chat con el asistente y botón de recomendaciones automáticas."]));

// --------------------------------------------------------------- resultados
C.push(h1("4. Resultados"),
  h2("4.1 Entrenamiento del clasificador"),
  p("La pérdida de validación bajó de forma sostenida hasta la tercera época, que obtuvo el mejor F1 macro en validación (0,9694); ese fue el modelo que se guardó. En las épocas 4 y 5 la pérdida de entrenamiento siguió bajando mientras la de validación se estabilizó, señal de que más épocas no aportaban mejoras."),
  ...tabla("Métricas por época (conjunto de validación).",
    ["Época", "Pérdida entrenamiento", "Pérdida validación", "Exactitud", "F1 macro"],
    [["1", "0,510", "0,449", "0,944", "0,943"], ["2", "0,216", "0,215", "0,967", "0,967"],
     ["3 (mejor)", "0,132", "0,166", "0,970", "0,969"], ["4", "0,083", "0,149", "0,967", "0,967"],
     ["5", "0,068", "0,152", "0,969", "0,968"]],
    [1560, 2100, 2100, 1800, 1800]),
  ...figura("curvas_entrenamiento.png", 1560, 480, "Curvas de pérdida y exactitud durante el entrenamiento."));

C.push(h2("4.2 Evaluación en el conjunto de prueba"),
  p(["Sobre las 828 imágenes de prueba, que el modelo nunca vio, el ViT afinado obtuvo una ", b("exactitud de 96,98 %"), " y un ", b("F1 macro de 0,969"), ". La clase orgánico, decisiva para separar orgánicos de inorgánicos, alcanzó un F1 de 0,993. La clase con menor desempeño fue plástico (F1 de 0,930): algunos envases plásticos transparentes se confunden con vidrio y metal por su brillo y transparencia."]),
  ...tabla("Reporte de clasificación en el conjunto de prueba.",
    ["Clase", "Precisión", "Exhaustividad (recall)", "F1", "Imágenes"],
    [["Cartón", "1,000", "0,963", "0,981", "134"], ["Metal", "0,928", "1,000", "0,963", "116"],
     ["Orgánico", "0,993", "0,993", "0,993", "148"], ["Papel", "0,986", "0,960", "0,973", "150"],
     ["Plástico", "0,944", "0,915", "0,930", "130"], ["Vidrio", "0,961", "0,987", "0,974", "150"],
     ["Promedio macro", "0,969", "0,970", "0,969", "828"]],
    [2160, 1800, 2200, 1500, 1700]),
  ...figura("matriz_confusion.png", 840, 720, "Matriz de confusión en el conjunto de prueba.", 4.2));

C.push(h2("4.3 Interpretación: mapas de atención"),
  p("Para verificar que el Transformer toma decisiones con sentido, se calcularon mapas de atención con la técnica de attention rollout: se combina la atención del token [CLS] hacia los 196 parches a lo largo de las 12 capas. En la Figura 4 se ve que el modelo se fija en el objeto (la etiqueta de la botella, el cuerpo de la lata, la textura del cartón) y no en el fondo, lo que muestra directamente el mecanismo de autoatención."),
  ...figura("mapas_atencion.png", 2160, 720, "Mapas de atención del ViT afinado (arriba: imagen y clase real; abajo: atención y predicción)."));

C.push(h2("4.4 Detección en video"),
  p("Para probar el sistema completo se generó un video de 20 segundos que simula una banda transportadora, con imágenes del conjunto de prueba que se desplazan por la escena. El sistema procesó los 500 cuadros en unos 27 segundos analizando 1 de cada 3 cuadros (18 segundos analizando 1 de cada 5) y dibujó para cada objeto su identificador de seguimiento, el material, si es orgánico o inorgánico y la confianza."),
  ...figura("prueba_video_frame.jpg", 960, 540, "Cuadro del video procesado: cada residuo con su identificador, material, tipo y confianza."),
  p("Durante las pruebas se identificaron y corrigieron dos problemas. El detector marcaba partes de un mismo objeto como objetos distintos (cajas anidadas), y el tracker contaba varias veces un objeto cuya detección aparecía y desaparecía. Con la supresión por contención y el umbral mínimo de apariciones, el conteo en el video de prueba (analizando 1 de cada 3 cuadros) bajó de 179 a 57 objetos, una cifra coherente con el contenido del video (varias imágenes contienen más de un objeto, como grupos de botellas o de frutas)."));

C.push(h2("4.5 Asistente"),
  p("Se evaluó el asistente con preguntas típicas. Algunas respuestas obtenidas:"),
  ...tabla("Ejemplos de respuestas del EcoAsistente.",
    ["Pregunta", "Respuesta del asistente"],
    [["¿Dónde boto una caja de pizza con grasa?", "Bota la caja de pizza con grasa en la caneca NEGRA."],
     ["¿Qué hago con el vidrio roto?", "Recoge los trozos con escoba y recogedor, envuélvelos en papel periódico o en una caja de cartón, cierra bien, marca el paquete como 'VIDRIO ROTO – PELIGRO' y deposítalo en la caneca blanca."],
     ["¿Dónde boto las pilas usadas?", "Lleva las pilas usadas a un punto de reciclaje autorizado. En Colombia, puedes encontrar estos servicios en locales de reciclaje, puntos de venta de electrodomésticos o en programas específicos como Pilas con el Ambiente o Punto Azul."],
     ["¿Dónde boto una servilleta usada?", "Bota la servilleta usada en la caneca negra."]],
    [3200, 6160]),
  p("En una primera versión, el modelo de 1 500 millones de parámetros generaba a veces indicaciones incorrectas (por ejemplo, sugería aplastar el vidrio roto). Se corrigió priorizando los fragmentos recuperados según la pregunta, ampliando la base de conocimiento, endureciendo las instrucciones del sistema y usando decodificación determinista. Esto muestra la importancia del RAG para usar LLM pequeños en un dominio específico."));

C.push(h1("5. Cumplimiento de los requisitos"),
  ...tabla("Relación entre los requisitos del proyecto y la solución.",
    ["Requisito", "Solución implementada", "Evidencia"],
    [["Detectar objetos en video", "OWLv2 + tracker por IoU", "Video anotado con identificadores (sección 4.4)"],
     ["Clasificar en plástico, papel, cartón, vidrio y metal", "ViT pre-entrenado con fine-tuning", "Exactitud 97,0 %, F1 0,969 (sección 4.2)"],
     ["Identificar orgánicos e inorgánicos", "Clase orgánico + reglas de mapeo", "F1 0,993 en orgánico; etiqueta Org./Ino. en cada caja"],
     ["Mostrar el resultado", "Aplicación Gradio con video, imagen y webcam", "Cajas, etiquetas, tablas y resumen"],
     ["Asistente de recomendaciones", "Qwen2.5 local + RAG + resultado de la detección", "Respuestas en español (sección 4.5)"],
     ["Modelo Transformer pre-entrenado, sin APIs", "ViT, OWLv2 y Qwen ejecutados localmente", "Funciona sin conexión tras la primera descarga"]],
    [2800, 3360, 3200]));

C.push(h1("6. Limitaciones y trabajo futuro"),
  vineta("El clasificador se entrenó con imágenes de un solo objeto sobre fondos sencillos. En escenas reales muy desordenadas (como la planta del conjunto WaRP) el desempeño puede bajar. Como mejora, se podría hacer fine-tuning de un detector Transformer de extremo a extremo (DETR o RT-DETR) con conjuntos de residuos anotados con cajas, como TACO o WaRP."),
  vineta("OWLv2 depende de las consultas de texto: los objetos que no se parecen a ninguna consulta pueden pasar desapercibidos."),
  vineta("El conteo en video usa un tracker sencillo por IoU; un tracker con predicción de movimiento (por ejemplo, ByteTrack) reduciría los conteos duplicados."),
  vineta("El LLM de 1 500 millones de parámetros puede responder de forma imprecisa ante preguntas fuera de su base de conocimiento. Un modelo más grande o una base de conocimiento más amplia con búsqueda semántica por embeddings mejoraría las respuestas."),
  vineta("Las clases “residuos no aprovechables” (caneca negra) y “residuos peligrosos” (pilas, electrónicos) no están en el clasificador; las atiende solo el asistente. Sería útil incorporarlas como clases visuales."));

C.push(h1("7. Conclusiones"),
  vineta("La arquitectura Transformer, vista en clase en el contexto de NLP, se aplicó con éxito a visión por computador: el mismo mecanismo de autoatención que relaciona palabras permite relacionar los parches de una imagen."),
  vineta("El aprendizaje por transferencia fue clave: con solo 3 855 imágenes de entrenamiento y 5 minutos de cómputo, el ViT pre-entrenado alcanzó una exactitud del 97 % en la clasificación de seis tipos de residuos."),
  vineta("Combinar un detector de vocabulario abierto (OWLv2) con un clasificador especializado (ViT afinado) permitió detectar y clasificar varios residuos por cuadro de video sin anotar cajas delimitadoras."),
  vineta("Los mapas de atención confirmaron que el modelo basa sus decisiones en el objeto y no en el fondo, lo que aporta interpretabilidad al sistema."),
  vineta("Un LLM pequeño ejecutado localmente puede servir como asistente útil si se le da información confiable (RAG) y se restringe su generación; sin estas medidas tiende a inventar indicaciones."),
  vineta("El sistema cumple los cinco requisitos del proyecto y funciona completamente en un computador personal, sin APIs externas."));

C.push(h1("Referencias"),
  ...[
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., … Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR).",
    "Minderer, M., Gritsenko, A. y Houlsby, N. (2023). Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36.",
    "Devlin, J., Chang, M.-W., Lee, K. y Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT.",
    "Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33.",
    "Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.",
    "Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33.",
    "Abnar, S. y Zuidema, W. (2020). Quantifying attention flow in transformers. Proceedings of ACL.",
    "Tunstall, L., von Werra, L. y Wolf, T. (2022). Natural Language Processing with Transformers. O’Reilly Media.",
    "Mohamed, M. (2021). Garbage Classification (12 classes) [Conjunto de datos]. Kaggle.",
    "Ministerio de Ambiente y Desarrollo Sostenible (2019). Resolución 2184 de 2019, por la cual se modifica la Resolución 668 de 2016 y se adoptan otras disposiciones. Bogotá, Colombia.",
  ].map(r => new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 720, hanging: 720 },
    children: [new TextRun({ text: r, size: 21 })] })));

// --------------------------------------------------------------- documento
const doc = new Document({
  creator: "Danilo Andrés Montezuma Ibarra",
  title: "EcoVisión – Informe del proyecto final",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: VERDE },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 25, bold: true, color: "2B2B2B" },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "vinetas", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "pasos", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "numeros", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: portada },
    { properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
        pageNumbers: { start: 1 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "EcoVisión – Proyecto final de Sistemas de Inteligencia Artificial · ", size: 18, color: "777777" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "777777" })] })] }) },
      children: C },
  ],
});

const destino = path.join(RAIZ, "informe", "Informe_Proyecto_Final_EcoVision.docx");
fs.mkdirSync(path.dirname(destino), { recursive: true });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(destino, buf); console.log("Informe creado:", destino); });
