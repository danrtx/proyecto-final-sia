// Genera informe/Informe_Proyecto_Final_EcoVision.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
  Footer, PageNumber, PageBreak, TableOfContents,
} = require("docx");

// ---------------------------------------------------------------- métricas del Transformer
const TR = JSON.parse(fs.readFileSync(path.join(__dirname, "metricas_transformer.json"), "utf8"));

const RAIZ = path.resolve(__dirname, "..");
const OUT = path.join(RAIZ, "outputs");
const FONT = "Arial";
const VERDE = "1E7B4A";

// ---------------------------------------------------------------- helpers
const runs = (texto, extra = {}) => (Array.isArray(texto) ? texto : [texto]).map(t => typeof t === "string" ? new TextRun({ text: t, ...extra }) : t);
const p = (texto, opts = {}) => new Paragraph({ spacing: { after: 120, line: 300 }, alignment: opts.align ?? AlignmentType.JUSTIFIED, children: runs(texto) });
const b = t => new TextRun({ text: t, bold: true });
const i = t => new TextRun({ text: t, italics: true });
const h1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const vineta = t => new Paragraph({ numbering: { reference: "vinetas", level: 0 }, spacing: { after: 60, line: 288 }, children: runs(t) });
const numerada = (t, ref) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 288 }, children: runs(t) });
const codigo = lineas => lineas.map((l, k) => new Paragraph({
  spacing: { after: 0, line: 240 }, shading: { type: ShadingType.CLEAR, color: "auto", fill: "F3F5F4" },
  indent: { left: 360 }, children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })] }));

function tamano(archivo) {
  const buf = fs.readFileSync(path.join(OUT, archivo));
  if (archivo.endsWith(".png")) return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
  for (let k = 2; k < buf.length;) {  // JPEG: buscar marcador SOF
    const m = buf[k + 1], len = buf.readUInt16BE(k + 2);
    if (m >= 0xC0 && m <= 0xC3) return [buf.readUInt16BE(k + 7), buf.readUInt16BE(k + 5)];
    k += 2 + len;
  }
}
let nFig = 0, nTab = 0;
function figura(archivo, pie, anchoIn = 6.3) {
  nFig++;
  const [w, h] = tamano(archivo);
  const ancho = Math.min(anchoIn, 6.5) * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: archivo.endsWith(".png") ? "png" : "jpg", data: fs.readFileSync(path.join(OUT, archivo)),
        transformation: { width: Math.round(ancho), height: Math.round(ancho * h / w) },
        altText: { title: pie, description: pie, name: archivo } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figura ${nFig}. `, bold: true, size: 20 }), new TextRun({ text: pie, italics: true, size: 20 })] }),
  ];
}
const borde = { style: BorderStyle.SINGLE, size: 4, color: "B7C9BE" };
const bordes = { top: borde, bottom: borde, left: borde, right: borde };
function tabla(titulo, encabezados, filas, anchos) {
  nTab++;
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
    new Table({ width: { size: anchos.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: anchos,
      rows: [new TableRow({ tableHeader: true, cantSplit: true, children: encabezados.map((e, j) => celda(e, j, true, 0)) }),
        ...filas.map((f, k) => new TableRow({ cantSplit: true, children: f.map((c, j) => celda(c, j, false, k)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
const pct = x => (100 * x).toFixed(1).replace(".", ",") + " %";

// ---------------------------------------------------------------- portada
const centro = (t, size, extra = {}, after = 200) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after }, children: [new TextRun({ text: t, size, ...extra })] });
const portada = [
  new Paragraph({ spacing: { before: 1800 }, children: [] }),
  centro("[NOMBRE DE LA UNIVERSIDAD]", 26, { bold: true }, 120),
  centro("Sistemas de Inteligencia Artificial", 24, {}, 1000),
  centro("EcoVisión", 56, { bold: true, color: VERDE }),
  centro("Sistema inteligente de detección y clasificación de residuos con una CNN y un Transformer construidos y entrenados desde cero", 30),
  centro("Proyecto final – Proyecto 4: Detección de residuos y clasificación de basura", 24, { italics: true }, 1400),
  centro("Presentado por:", 22, {}, 80),
  centro("Danilo Andrés Montezuma Ibarra", 24, { bold: true }, 400),
  centro("Docente:", 22, {}, 80),
  centro("PhD Cristian Camilo Ordoñez", 24, { bold: true }, 1200),
  centro("Noviembre de 2026", 22),
  new Paragraph({ children: [new PageBreak()] }),
];

// ---------------------------------------------------------------- contenido
const C = [];
C.push(new Paragraph({ spacing: { before: 120, after: 200 }, children: [new TextRun({ text: "Tabla de contenido", bold: true, size: 30, color: VERDE })] }),
  new TableOfContents("Tabla de contenido", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

C.push(h1("Resumen"),
  p(["Este proyecto presenta ", b("EcoVisión"), ", un sistema que detecta residuos en video, los clasifica por material (plástico, papel, cartón, vidrio, metal y orgánico), indica si son orgánicos o inorgánicos, muestra el resultado en pantalla y entrega retroalimentación sobre cómo separarlos. Los dos modelos de inteligencia artificial del sistema se ", b("diseñaron, programaron y entrenaron desde cero"), ": no se usó ningún modelo pre-entrenado, ni pesos descargados, ni APIs externas."]),
  p(["El primer modelo es ", b("EcoNet"), ", una red neuronal convolucional (CNN) de 1,17 millones de parámetros entrenada con 4 601 imágenes. En el conjunto de prueba alcanzó una ", b("exactitud del 83,0 %"), " en las seis clases y del ", b("97,2 % al separar orgánicos de inorgánicos"), ". El segundo es ", b("EcoTransformer"), ", un Transformer codificador-decodificador implementado a mano a partir del artículo “Attention is All You Need”. Recibe la salida de la CNN y una pregunta opcional del usuario y genera la retroalimentación en español. Se entrenó con un dataset propio de 40 000 ejemplos y respondió correctamente el ", b(pct(TR.exacta_vistas)), " de los casos con plantillas conocidas y el ", b(pct(TR.exacta_nuevas)), " de los casos con frases que nunca vio (con la caneca correcta en el ", b(pct(TR.canecas_nuevas)), ")."]),
  p([b("Palabras clave: "), "red neuronal convolucional, Transformer, mecanismo de atención, entrenamiento desde cero, clasificación de residuos, reciclaje."]));

C.push(h1("1. Introducción"),
  p("La separación de residuos en la fuente es clave para que los materiales aprovechables lleguen a las cadenas de reciclaje. En Colombia, la Resolución 2184 de 2019 unificó el código de colores: caneca blanca para aprovechables, verde para orgánicos y negra para no aprovechables. Aun así, muchas personas no saben en qué caneca va cada residuo, y en las plantas de aprovechamiento la clasificación sigue haciéndose a mano."),
  p("Siguiendo la indicación del docente, el sistema se construyó con dos modelos propios, sin modelos pre-entrenados: una CNN que reconoce el residuo y un Transformer que toma la respuesta de la CNN y brinda retroalimentación. Así, el proyecto aplica dos de las arquitecturas centrales del aprendizaje profundo vistas en el curso: las redes convolucionales para visión y el Transformer con su mecanismo de atención."),
  h2("1.1 Objetivo general"),
  p("Construir desde cero un sistema inteligente, basado en una CNN y un Transformer propios, que identifique y clasifique residuos para facilitar los procesos de reciclaje."),
  h2("1.2 Objetivos específicos"),
  numerada("Detectar los objetos presentes en un video.", "obj"),
  numerada("Clasificar los residuos en plástico, papel, cartón, vidrio y metal.", "obj"),
  numerada("Identificar si cada residuo es orgánico o inorgánico.", "obj"),
  numerada("Mostrar el resultado de la clasificación.", "obj"),
  numerada("Crear un asistente que brinde recomendaciones sobre la separación de residuos.", "obj"));

C.push(h1("2. Marco teórico"),
  h2("2.1 Redes neuronales convolucionales (CNN)"),
  p("Una CNN procesa imágenes con capas de convolución: filtros pequeños (por ejemplo, de 3×3) que se deslizan sobre la imagen y responden a patrones locales. Las primeras capas detectan bordes y colores; las más profundas combinan esos patrones en texturas y formas. Cada convolución suele ir seguida de normalización por lotes (BatchNorm), que estabiliza el entrenamiento, y de una función de activación ReLU, que introduce no linealidad. Las capas de agrupamiento (max pooling) reducen la resolución, lo que da invariancia a pequeños desplazamientos. Al final, un agrupamiento global y una capa lineal convierten los mapas de características en probabilidades para cada clase."),
  h2("2.2 Arquitectura Transformer y mecanismo de atención"),
  p("El Transformer (Vaswani et al., 2017) reemplaza las redes recurrentes por un mecanismo de autoatención: cada token de la secuencia se relaciona directamente con todos los demás, y el cálculo se hace en paralelo. Cada token se proyecta en tres vectores, consulta (Q), clave (K) y valor (V), y la atención se calcula así:"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 }, children: [new TextRun({ text: "Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V", italics: true, size: 26 })] }),
  p("El producto Q·Kᵀ mide la similitud entre consultas y claves; √dₖ escala los puntajes y softmax los convierte en probabilidades con las que se ponderan los valores. La atención multi-cabeza repite este cálculo en paralelo con distintas proyecciones. Como la atención no conoce el orden, se suma una codificación posicional sinusoidal a los embeddings."),
  p("La arquitectura codificador-decodificador tiene dos partes. El codificador lee la secuencia de entrada y construye una representación contextual. El decodificador genera la salida palabra por palabra: con una auto-atención enmascarada que solo mira las palabras ya generadas, y con una atención cruzada que consulta la representación del codificador. Es la arquitectura que se usa en traducción automática, y en este proyecto “traduce” la salida de la CNN a una retroalimentación en lenguaje natural."),
  h2("2.3 Tokenización"),
  p("Antes de entrar al Transformer, el texto se divide en tokens (en este proyecto, palabras y signos de puntuación) y cada token se convierte en un número según un vocabulario construido a partir de los datos de entrenamiento. Las palabras que no están en el vocabulario se reemplazan por un token especial <unk>."),
  h2("2.4 Visión por computador clásica para localizar objetos"),
  p("Para localizar los objetos en movimiento se usa la sustracción de fondo MOG2 (Zivkovic, 2004), que modela cada píxel del fondo con una mezcla de gaussianas y marca como primer plano lo que se aparta de ese modelo. Luego, con operaciones morfológicas (apertura y cierre) y la búsqueda de contornos, se obtienen las cajas delimitadoras. Estas técnicas son algoritmos clásicos de OpenCV: no aprenden de datos ni usan modelos pre-entrenados."));

C.push(h1("3. Metodología"),
  h2("3.1 Arquitectura general"),
  ...tabla("Componentes del sistema EcoVisión.",
    ["Etapa", "Componente", "Construido por", "Función"],
    [["1. Localización", "Sustracción de fondo MOG2 + contornos (OpenCV)", "Algoritmo clásico, sin entrenamiento", "Encuentra los objetos en cada cuadro del video"],
     ["2. Clasificación", "EcoNet (CNN)", "Nosotros, desde cero", "Asigna el material a cada objeto"],
     ["3. Reglas y seguimiento", "Tabla de la Res. 2184 + tracker por IoU", "Nosotros", "Orgánico/inorgánico, caneca y conteo de objetos únicos"],
     ["4. Retroalimentación", "EcoTransformer (codificador-decodificador)", "Nosotros, desde cero", "Toma la salida de la CNN y genera las recomendaciones"]],
    [1700, 2700, 2200, 2760]),
  p(["Todo se implementó en Python con ", i("PyTorch"), " (solo como biblioteca de tensores y gradientes) y se ejecuta en un equipo con GPU NVIDIA RTX 4050 de 6 GB. La interfaz se construyó con ", i("Gradio"), "."]));

C.push(h2("3.2 Conjunto de datos de imágenes"),
  p(["Se usó el conjunto público ", i("Garbage Classification"), " (12 clases, 15 515 imágenes). Las 8 clases pertinentes se reagruparon en las 6 del proyecto: se unieron los tres tipos de vidrio y la clase ", i("biological"), " se convirtió en ", i("orgánico"), ". Se usaron todas las imágenes de esas clases (6 576) y se dividieron por clase en 70 % entrenamiento, 15 % validación y 15 % prueba. Como vidrio tiene el doble de imágenes que las demás clases, se usaron pesos por clase en la función de pérdida."]),
  ...tabla("Distribución del conjunto de imágenes.",
    ["Clase", "Clase(s) original(es)", "Entrenamiento", "Validación", "Prueba"],
    [["Cartón", "cardboard", 623, 134, 134], ["Metal", "metal", 538, 115, 116], ["Orgánico", "biological", 689, 148, 148],
     ["Papel", "paper", 735, 157, 158], ["Plástico", "plastic", 605, 130, 130], ["Vidrio", "brown/green/white-glass", 1411, 302, 303],
     ["Total", "", 4601, 986, 989]],
    [1500, 2860, 1700, 1650, 1650]),
  ...figura("ejemplos_dataset.png", "Ejemplos del conjunto de entrenamiento."));

C.push(h2("3.3 EcoNet: diseño de la CNN"),
  p("EcoNet recibe imágenes de 128×128 píxeles y tiene cuatro bloques convolucionales. Cada bloque aplica dos veces la secuencia convolución 3×3, BatchNorm y ReLU, y termina con un max pooling que reduce la resolución a la mitad. El número de filtros se duplica en cada bloque (32, 64, 128 y 256). Al final, un agrupamiento promedio global, un dropout del 40 % y una capa lineal producen las 6 salidas. Todos los pesos se inicializan aleatoriamente (inicialización de Kaiming)."),
  ...tabla("Arquitectura de EcoNet.",
    ["Capa", "Operación", "Salida"],
    [["Entrada", "Imagen RGB", "3 × 128 × 128"], ["Bloque 1", "2 × (Conv 3×3, 32 filtros + BN + ReLU) + MaxPool", "32 × 64 × 64"],
     ["Bloque 2", "2 × (Conv 3×3, 64 filtros + BN + ReLU) + MaxPool", "64 × 32 × 32"], ["Bloque 3", "2 × (Conv 3×3, 128 filtros + BN + ReLU) + MaxPool", "128 × 16 × 16"],
     ["Bloque 4", "2 × (Conv 3×3, 256 filtros + BN + ReLU) + MaxPool", "256 × 8 × 8"], ["Clasificador", "Global Average Pooling + Dropout 0,4 + Lineal", "6 clases"],
     ["Total", "1 174 758 parámetros entrenables", ""]],
    [1700, 5260, 2400]),
  p("Como la red parte de cero y el conjunto de datos es pequeño, se aplicó un aumento de datos intenso: en cada época, cada imagen se recorta, se voltea, se rota y se le cambian el brillo, el contraste y la saturación de forma aleatoria. Esto evita que la red memorice las imágenes."),
  ...figura("aumento_datos.png", "Una misma imagen con seis aumentos aleatorios distintos."),
  ...tabla("Hiperparámetros del entrenamiento de EcoNet.",
    ["Parámetro", "Valor"],
    [["Épocas", "45"], ["Tamaño de lote", "64"], ["Optimizador", "AdamW (weight decay 5×10⁻⁴)"], ["Tasa de aprendizaje", "Máximo 2×10⁻³ con política One-Cycle"],
     ["Función de pérdida", "Entropía cruzada con pesos por clase y label smoothing 0,1"], ["Precisión", "Mixta (FP16)"],
     ["Selección del modelo", "Mejor F1 macro en validación"], ["Tiempo de entrenamiento", "41 minutos (RTX 4050)"]],
    [3600, 5760]));

C.push(h2("3.4 Detección de objetos en video"),
  p("Por cada cuadro del video:"),
  numerada("La sustracción de fondo MOG2 genera una máscara con lo que se mueve respecto al fondo (los residuos sobre la banda).", "det"),
  numerada("Una apertura morfológica elimina el ruido y un cierre une las partes de un mismo objeto. Luego se extraen los contornos y sus cajas delimitadoras, y se descartan las demasiado pequeñas o demasiado grandes. Las cajas que quedan casi completamente dentro de otra se fusionan.", "det"),
  numerada("Cada caja se recorta y EcoNet predice su material. Las predicciones con confianza menor a 0,40 se descartan.", "det"),
  numerada("Con la tabla de reglas se asignan el tipo (orgánico o inorgánico) y la caneca, y se dibuja la caja con su etiqueta.", "det"),
  numerada("Un tracker por IoU asocia las detecciones entre cuadros y asigna un identificador a cada objeto. Al final, cada objeto se cuenta una vez con la clase que acumuló más confianza.", "det"),
  p("Para imágenes sueltas y la webcam, el fondo se estima con el color mediano de los bordes de la imagen y se combina con detección de bordes (Canny). Si no se encuentra ningún objeto, se clasifica la imagen completa."));

C.push(h2("3.5 EcoTransformer: diseño del Transformer de retroalimentación"),
  p("El Transformer recibe como entrada una secuencia de texto con la salida de la CNN y, opcionalmente, la pregunta del usuario. A partir de ella genera la retroalimentación:"),
  ...codigo(["Entrada : det plastico 2 vidrio 1 sep que hago con esto",
    "Salida  : Plástico (2): es inorgánico, va en la caneca blanca. Vacíalo, enjuágalo",
    "          y aplástalo. Vidrio (1): es inorgánico, va en la caneca blanca. Vacío,",
    "          sin tapa y con cuidado. Todo es aprovechable: llévalo limpio y seco a",
    "          la caneca blanca."]),
  p(" "),
  p("Todos sus componentes se programaron a mano: el tokenizador por palabras, los embeddings, la codificación posicional sinusoidal, la atención multi-cabeza (con las proyecciones W_Q, W_K, W_V y W_O y la fórmula softmax(QKᵀ/√dₖ)·V), las capas del codificador y del decodificador (con la máscara causal), y la decodificación voraz (greedy) palabra por palabra."),
  ...tabla("Hiperparámetros de EcoTransformer.",
    ["Parámetro", "Valor"],
    [["Capas del codificador / decodificador", "3 / 3"], ["Dimensión del modelo (d_modelo)", "128"], ["Cabezas de atención", "4 (dₖ = 32)"],
     ["Dimensión de la red feed-forward", "512"], ["Vocabulario de entrada / salida", `${TR.vocab_src} / ${TR.vocab_tgt} tokens`], ["Parámetros", TR.parametros.toLocaleString("es-CO")],
     ["Entrenamiento", `14 épocas, lote 128, AdamW + One-Cycle (máx. 10⁻³), ${TR.minutos} min`]],
    [4200, 5160]));

C.push(h2("3.6 Dataset propio de retroalimentación"),
  p("No existe un conjunto de datos público para esta tarea, así que se creó uno propio a partir de las normas de la Resolución 2184 de 2019. Cada ejemplo combina una salida simulada de la CNN (de 0 a 6 materiales, con su cantidad) con uno de tres tipos de pregunta:"),
  vineta([b("Retroalimentación general "), "(sin pregunta, o “dame recomendaciones”, “¿en qué caneca va esto?”): una línea por material con su caneca, su tipo y cómo prepararlo, más un mensaje de cierre según haya orgánicos, inorgánicos o ambos."]),
  vineta([b("Pregunta por material "), "(“¿dónde boto una lata de atún?”): el consejo para ese material, reconociendo sinónimos (lata, aluminio, frasco, cáscara, periódico…)."]),
  vineta([b("Pregunta de conocimiento "), "(vidrio roto, caja de pizza con grasa, pilas, servilletas, compost, aceite usado, medicamentos, colores de las canecas, etc.): la respuesta de la base de conocimiento."]),
  p(["Las preguntas se generan con plantillas, sinónimos, saludos y cortesías, y se obtuvieron 40 000 ejemplos de entrenamiento. Para medir la generalización, ", b("las dos últimas plantillas de cada tipo se excluyeron del entrenamiento"), " y se usaron solo en un conjunto de prueba aparte (“frases nuevas”)."]));

C.push(h2("3.7 Interfaz de usuario"),
  p("La aplicación web local (Gradio) tiene cuatro pestañas: Video (video anotado, tabla de residuos únicos y retroalimentación), Imagen (análisis de una foto con el detalle de cada objeto), Webcam en vivo (detección en tiempo real) y EcoAsistente (chat en el que el Transformer responde usando lo último que detectó la CNN)."));

// --------------------------------------------------------------- resultados
C.push(h1("4. Resultados"),
  h2("4.1 Entrenamiento de EcoNet"),
  p("Como la red parte de pesos aleatorios, la exactitud empezó cerca del 42 % y subió de forma sostenida durante las 45 épocas. La pérdida de validación siguió de cerca a la de entrenamiento, lo que indica que el aumento de datos y el dropout evitaron el sobreajuste. El mejor modelo se obtuvo en la época 44, y la curva seguía subiendo, así que más épocas probablemente mejorarían el resultado."),
  ...tabla("Evolución del entrenamiento de EcoNet (validación).",
    ["Época", "Pérdida entrenamiento", "Exactitud entrenamiento", "Pérdida validación", "Exactitud validación", "F1 macro"],
    [["1", "1,585", "0,420", "1,399", "0,567", "0,525"], ["10", "1,200", "0,626", "1,351", "0,567", "0,582"], ["20", "1,045", "0,705", "1,021", "0,740", "0,732"],
     ["30", "0,910", "0,777", "0,943", "0,801", "0,797"], ["40", "0,819", "0,824", "0,871", "0,836", "0,836"], ["44 (mejor)", "0,802", "0,835", "0,862", "0,842", "0,842"]],
    [1300, 1700, 1700, 1600, 1600, 1460]),
  ...figura("curvas_cnn.png", "Curvas de pérdida y exactitud de EcoNet."));

C.push(h2("4.2 Evaluación de EcoNet en el conjunto de prueba"),
  p(["Sobre las 989 imágenes de prueba, que la red nunca vio, EcoNet obtuvo una ", b("exactitud del 83,0 %"), " y un ", b("F1 macro de 0,832"), ". Para el objetivo de separar orgánicos de inorgánicos, la exactitud fue del ", b("97,2 %"), ". Las clases con mejor desempeño fueron cartón (F1 de 0,928) y orgánico (0,907). Las más difíciles fueron plástico (0,699) y metal (0,780): los envases plásticos brillantes o transparentes se confunden con metal y con vidrio."]),
  ...tabla("Reporte de clasificación de EcoNet en el conjunto de prueba.",
    ["Clase", "Precisión", "Exhaustividad", "F1", "Imágenes"],
    [["Cartón", "0,946", "0,910", "0,928", "134"], ["Metal", "0,696", "0,888", "0,780", "116"], ["Orgánico", "0,895", "0,919", "0,907", "148"],
     ["Papel", "0,892", "0,835", "0,863", "158"], ["Plástico", "0,630", "0,785", "0,699", "130"], ["Vidrio", "0,904", "0,746", "0,817", "303"],
     ["Promedio macro", "0,827", "0,847", "0,832", "989"]],
    [2160, 1800, 2000, 1700, 1700]),
  ...figura("matriz_confusion_cnn.png", "Matriz de confusión de EcoNet en el conjunto de prueba.", 4.2),
  ...figura("predicciones_cnn.png", "Predicciones de EcoNet sobre imágenes de prueba (verde: correcta; rojo: incorrecta)."));

C.push(h2("4.3 ¿Qué aprendió la CNN?"),
  p("La Figura 6 muestra los 32 filtros de la primera capa, aprendidos desde cero: varios responden a bordes con distintas orientaciones y otros a colores específicos, igual que en las CNN descritas en la literatura. La Figura 7 muestra los mapas de activación de un bloque intermedio: cada canal resalta una parte distinta del objeto (contornos, zonas brillantes, texturas)."),
  ...figura("filtros_cnn.png", "Filtros 3×3 aprendidos por la primera capa de EcoNet.", 4.5),
  ...figura("activaciones_cnn.png", "Mapas de activación del bloque 2 de EcoNet."));

C.push(h2("4.4 Evaluación de EcoTransformer"),
  p(["El Transformer aprendió rápido: desde la segunda época la exactitud por token en validación llegó al 99,99 %. Para evaluarlo de forma exigente, se le pidió generar respuestas completas él solo, palabra por palabra, y se compararon con las esperadas."]),
  ...tabla("Resultados de EcoTransformer en generación completa.",
    ["Conjunto de prueba", "Coincidencia exacta", "Canecas correctas"],
    [["Plantillas vistas, combinaciones nuevas (2 000)", pct(TR.exacta_vistas), pct(TR.canecas_vistas)],
     ["Frases nunca vistas en el entrenamiento (2 000)", pct(TR.exacta_nuevas), pct(TR.canecas_nuevas)]],
    [4560, 2400, 2400]),
  p(["Con plantillas conocidas, el modelo reproduce la retroalimentación esperada: copia bien los materiales y las cantidades detectadas por la CNN y elige el mensaje de cierre correcto. Con frases nunca vistas acierta la respuesta exacta en el ", pct(TR.exacta_nuevas), " de los casos y la caneca correcta en el ", pct(TR.canecas_nuevas), ". Los errores aparecen sobre todo cuando la frase nueva contiene palabras fuera del vocabulario (<unk>), y entonces el modelo elige otra respuesta de la base de conocimiento."]),
  ...figura("curvas_transformer.png", "Curvas de entrenamiento de EcoTransformer."),
  p("La Figura 9 muestra la atención cruzada del decodificador. Al escribir “Plástico ( 3 )”, el modelo atiende a los tokens “plastico” y “3” de la entrada, es decir, a la salida de la CNN. Así se ve directamente cómo el Transformer “lee” la respuesta de la CNN para construir la retroalimentación."),
  ...figura("atencion_transformer.png", "Atención cruzada codificador → decodificador de EcoTransformer.", 5.0));

C.push(h2("4.5 Sistema completo en video"),
  p("Para probar el sistema completo se generó un video de 20 segundos que simula una banda transportadora, con imágenes del conjunto de prueba. El sistema procesó los 500 cuadros en unos 14 segundos (analizando 1 de cada 2), dibujó cada objeto con su identificador, material, tipo y confianza, y el Transformer generó la retroalimentación a partir del conteo final."),
  ...figura("cuadro_video_resultado.jpg", "Cuadro del video procesado: cada residuo con su identificador, material (EcoNet), tipo y confianza."),
  p("En la figura se ven aciertos (cartón, orgánico, plástico) y también errores típicos de una red entrenada desde cero con pocos datos: una bandeja de aluminio clasificada como plástico o una revista como metal. Además, cuando dos objetos se tocan, la sustracción de fondo puede unirlos en una sola caja."));

C.push(h1("5. Cumplimiento de los requisitos"),
  ...tabla("Relación entre los requisitos y la solución.",
    ["Requisito", "Solución", "Evidencia"],
    [["Detectar objetos en video", "Sustracción de fondo + contornos + tracker", "Video anotado con identificadores (4.5)"],
     ["Clasificar en plástico, papel, cartón, vidrio y metal", "EcoNet, CNN propia desde cero", "Exactitud 83,0 %, F1 0,832 (4.2)"],
     ["Identificar orgánicos e inorgánicos", "Clase orgánico + reglas", "Exactitud 97,2 % orgánico vs. inorgánico"],
     ["Mostrar el resultado", "Aplicación Gradio (video, imagen, webcam)", "Cajas, etiquetas y tablas"],
     ["Asistente de recomendaciones", "EcoTransformer, Transformer propio desde cero", `Canecas correctas ${pct(TR.canecas_nuevas)} con frases nuevas (4.4)`],
     ["CNN y Transformer construidos desde cero", "Arquitecturas programadas en PyTorch, pesos aleatorios", "Notebooks 01 y 02"]],
    [2900, 3360, 3100]));

C.push(h1("6. Limitaciones y trabajo futuro"),
  vineta("La exactitud de EcoNet (83 %) es menor que la de los modelos pre-entrenados con millones de imágenes, lo esperable con 4 601 imágenes y entrenamiento desde cero. Más épocas, más datos (o datos propios capturados con la cámara del sistema) y arquitecturas residuales (ResNet) mejorarían el resultado."),
  vineta("La localización por sustracción de fondo necesita cámara fija y objetos en movimiento, y puede unir objetos que se tocan. Un detector neuronal entrenado desde cero (por ejemplo, de tipo YOLO o CenterNet) con un conjunto anotado con cajas resolvería esto."),
  vineta("El Transformer genera respuestas a partir de su dataset: no responde temas fuera de él y puede fallar con palabras desconocidas. Ampliar el dataset con más plantillas, sinónimos y preguntas reales de usuarios mejoraría su generalización."),
  vineta("Los residuos no aprovechables (caneca negra) y los peligrosos (pilas, medicamentos) solo los atiende el Transformer por texto; sería útil agregarlos como clases visuales de la CNN."));

C.push(h1("7. Conclusiones"),
  vineta("Se construyó un sistema completo de detección, clasificación y retroalimentación sobre residuos con dos modelos diseñados y entrenados desde cero, sin modelos pre-entrenados ni APIs."),
  vineta("EcoNet, con 1,17 millones de parámetros, alcanzó un 83 % de exactitud en seis clases y un 97 % al separar orgánicos de inorgánicos, que es la decisión más importante para el usuario."),
  vineta("Implementar el Transformer a mano permitió entender en la práctica la atención Q·K·V, las máscaras, la codificación posicional y la arquitectura codificador-decodificador. El mapa de atención cruzada muestra cómo el decodificador consulta la salida de la CNN para generar cada parte de la respuesta."),
  vineta("La combinación de visión por computador clásica (para localizar), una CNN (para reconocer) y un Transformer (para comunicar el resultado en lenguaje natural) resolvió los cinco requisitos del proyecto en un computador personal."),
  vineta("La diferencia frente a los modelos pre-entrenados muestra el valor del pre-entrenamiento con grandes volúmenes de datos, y también que con datos bien preparados, aumento de datos y una arquitectura adecuada se logran resultados útiles desde cero."));

C.push(h1("Referencias"),
  ...["Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
    "LeCun, Y., Bottou, L., Bengio, Y. y Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278–2324.",
    "Simonyan, K. y Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. International Conference on Learning Representations (ICLR).",
    "Ioffe, S. y Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. International Conference on Machine Learning (ICML).",
    "He, K., Zhang, X., Ren, S. y Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. IEEE International Conference on Computer Vision (ICCV).",
    "Smith, L. N. y Topin, N. (2019). Super-convergence: Very fast training of neural networks using large learning rates. Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, SPIE.",
    "Zivkovic, Z. (2004). Improved adaptive Gaussian mixture model for background subtraction. International Conference on Pattern Recognition (ICPR).",
    "Goodfellow, I., Bengio, Y. y Courville, A. (2016). Deep Learning. MIT Press.",
    "Tunstall, L., von Werra, L. y Wolf, T. (2022). Natural Language Processing with Transformers. O’Reilly Media.",
    "Mohamed, M. (2021). Garbage Classification (12 classes) [Conjunto de datos]. Kaggle.",
    "Ministerio de Ambiente y Desarrollo Sostenible (2019). Resolución 2184 de 2019. Bogotá, Colombia.",
  ].map(r => new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 720, hanging: 720 }, children: [new TextRun({ text: r, size: 21 })] })));

// --------------------------------------------------------------- documento
const lista = ref => ({ reference: ref, levels: [{ level: 0, format: ref === "vinetas" ? LevelFormat.BULLET : LevelFormat.DECIMAL,
  text: ref === "vinetas" ? "•" : "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] });
const pagina = { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } };
const doc = new Document({
  creator: "Danilo Andrés Montezuma Ibarra", title: "EcoVisión – Informe del proyecto final",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: VERDE }, paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 25, bold: true, color: "2B2B2B" }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [lista("vinetas"), lista("obj"), lista("det")] },
  sections: [
    { properties: { page: pagina }, children: portada },
    { properties: { page: { ...pagina, pageNumbers: { start: 1 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "EcoVisión – Proyecto final de Sistemas de Inteligencia Artificial · ", size: 18, color: "777777" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "777777" })] })] }) },
      children: C },
  ],
});
const destino = path.join(RAIZ, "informe", "Informe_Proyecto_Final_EcoVision.docx");
fs.mkdirSync(path.dirname(destino), { recursive: true });
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(destino, buf); console.log("Informe creado:", destino); });
