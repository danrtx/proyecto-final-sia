"""EcoTransformer: Transformer encoder-decoder construido y entrenado desde cero.

Implementa la arquitectura de "Attention is All You Need" (Vaswani et al., 2017):
  - Tokenizador propio a nivel de palabra.
  - Embeddings + codificación posicional sinusoidal.
  - Atención multi-cabeza programada a mano:  Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
  - Codificador (lee la salida de la CNN + la pregunta) y decodificador (genera la
    retroalimentación palabra por palabra, de forma autorregresiva).
No se usa ningún modelo ni peso pre-entrenado.

¿QUÉ HACE EN EL PROYECTO?
  La CNN dice qué residuos hay (por ejemplo: plástico x2, vidrio x1). Este Transformer
  "traduce" esa salida a una retroalimentación en español, igual que un traductor
  convierte una frase de un idioma a otro:

      entrada : "det plastico 2 vidrio 1 sep que hago con esto"
      salida  : "Plástico (2): es inorgánico, va en la caneca blanca. ..."

ORGANIZACIÓN DEL ARCHIVO
  1. Tokenizador y vocabulario ............ tokenizar(), Vocabulario
  2. Atención multi-cabeza ................ AtencionMultiCabeza   (el corazón del Transformer)
  3. Codificación posicional .............. CodificacionPosicional
  4. Red feed-forward ..................... FeedForward
  5. Capa del codificador ................. CapaCodificador
  6. Capa del decodificador ............... CapaDecodificador
  7. Modelo completo + máscaras ........... EcoTransformer
  8. Generación palabra por palabra ....... EcoTransformer.generar()
  9. Uso desde la aplicación .............. Retroalimentador
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn as nn

# Tokens especiales del vocabulario (siempre ocupan los IDs 0, 1, 2 y 3):
#   <pad> = relleno para que todas las frases de un lote tengan la misma longitud
#   <sos> = "start of sequence": con este token el decodificador empieza a escribir
#   <eos> = "end of sequence": cuando el modelo lo genera, la respuesta terminó
#   <unk> = "unknown": reemplaza las palabras que no están en el vocabulario
PAD, SOS, EOS, UNK = 0, 1, 2, 3
ESPECIALES = ["<pad>", "<sos>", "<eos>", "<unk>"]


# =================================================================== 1. TOKENIZADOR
# TOKENIZACIÓN (visto en clase): dividir el texto en unidades mínimas llamadas tokens.
# Usamos tokenización POR PALABRAS: cada palabra y cada signo de puntuación es un token.
#   "Hola, ¿qué tal?"  ->  ["Hola", ",", "¿", "qué", "tal", "?"]
def tokenizar(texto: str) -> list[str]:
    # \w+      -> una o más letras/números seguidos (una palabra)
    # [^\w\s]  -> un signo de puntuación (cualquier cosa que no sea letra ni espacio)
    return re.findall(r"\w+|[^\w\s]", texto)


def destokenizar(tokens: list[str]) -> str:
    """Operación inversa: une los tokens generados en un texto legible."""
    t = " ".join(tokens)
    t = re.sub(r"\s+([.,:;)?!])", r"\1", t)  # quita el espacio antes de . , : ; ) ? !
    t = re.sub(r"([(¿¡])\s+", r"\1", t)      # quita el espacio después de ( ¿ ¡
    return t


class Vocabulario:
    """Diccionario token <-> número. Las redes neuronales solo procesan números,
    así que cada palabra se convierte en un ID entero (y luego en un embedding).
      itos = "index to string": lista  [ID] -> palabra
      stoi = "string to index": dict   palabra -> ID
    """

    def __init__(self, tokens: list[str]):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def construir(cls, textos: list[str], min_frec: int = 1):
        # Se cuentan todas las palabras de los textos de ENTRENAMIENTO y cada palabra
        # distinta recibe un ID. Resultado: 204 tokens de entrada y 267 de salida.
        c = Counter(t for x in textos for t in tokenizar(x))
        return cls(ESPECIALES + sorted(t for t, f in c.items() if f >= min_frec))

    def codificar(self, texto: str, extremos: bool = False) -> list[int]:
        # Texto -> lista de IDs. Una palabra desconocida se convierte en <unk>.
        # Con extremos=True se agregan <sos> al inicio y <eos> al final (para la salida).
        ids = [self.stoi.get(t, UNK) for t in tokenizar(texto)]
        return [SOS] + ids + [EOS] if extremos else ids

    def decodificar(self, ids: list[int]) -> str:
        # Lista de IDs -> texto (se ignoran los tokens especiales).
        return destokenizar([self.itos[i] for i in ids if i not in (PAD, SOS, EOS)])

    def __len__(self):
        return len(self.itos)


# =================================================================== 2. ATENCIÓN
class AtencionMultiCabeza(nn.Module):
    """softmax(Q·Kᵀ/√d_k)·V calculada en paralelo para h cabezas.

    MECANISMO DE ATENCIÓN (el concepto central del Transformer):
      Cada token "mira" a todos los demás tokens y decide a cuáles prestarles atención.
      Analogía de la biblioteca:
        Q (Query, consulta) = lo que estoy buscando
        K (Key, clave)      = la etiqueta de cada libro
        V (Value, valor)    = el contenido de cada libro
      Comparo mi consulta con todas las etiquetas (Q·Kᵀ); cuanto más se parecen,
      más "leo" de ese libro (más peso recibe su V).

    MULTI-CABEZA: el cálculo se repite en paralelo h veces (h = 4 cabezas) con
    proyecciones distintas, para que cada cabeza pueda especializarse en un tipo
    de relación (por ejemplo, material <-> cantidad, o pregunta <-> tema).
    """

    def __init__(self, d_modelo: int, n_cabezas: int, dropout: float = 0.1):
        super().__init__()
        assert d_modelo % n_cabezas == 0
        # d_modelo = 128 números por token; con 4 cabezas, cada cabeza trabaja con
        # d_k = 128 / 4 = 32 números.
        self.h, self.d_k = n_cabezas, d_modelo // n_cabezas
        # Matrices APRENDIDAS que convierten cada token en su Q, su K y su V.
        self.W_q = nn.Linear(d_modelo, d_modelo)
        self.W_k = nn.Linear(d_modelo, d_modelo)
        self.W_v = nn.Linear(d_modelo, d_modelo)
        # Matriz que combina el resultado de las 4 cabezas.
        self.W_o = nn.Linear(d_modelo, d_modelo)
        self.dropout = nn.Dropout(dropout)
        self.pesos_atencion = None  # se guardan para visualizarlos (mapa de atención del notebook 02)

    def forward(self, q, k, v, mascara=None):
        # q, k, v tienen forma (B, L, d_modelo): B = frases en el lote, L = tokens por frase.
        # - En AUTO-ATENCIÓN, q, k y v son la misma secuencia.
        # - En ATENCIÓN CRUZADA, q viene del decodificador y k, v del codificador.
        B = q.size(0)
        # 1) Proyecciones lineales y separación en h cabezas: (B, h, L, d_k)
        Q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Puntajes de atención: Q·Kᵀ / √d_k
        #    Producto punto de cada consulta con cada clave = qué tan relacionados están.
        #    Se divide entre √d_k (√32 ≈ 5,66) para que los números no crezcan demasiado:
        #    si son muy grandes, el softmax se "satura" y los gradientes se vuelven casi 0.
        puntajes = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mascara is not None:
            # MÁSCARA: las posiciones prohibidas (relleno <pad> o palabras futuras) reciben -∞,
            # así después del softmax su peso es exactamente 0.
            puntajes = puntajes.masked_fill(~mascara, float("-inf"))
        # 3) softmax -> distribución de probabilidad sobre las posiciones
        #    Cada fila suma 1: son los "pesos de atención" (cuánto mira cada token a cada otro).
        A = torch.softmax(puntajes, dim=-1)
        self.pesos_atencion = A.detach()
        # 4) Suma ponderada de los valores y unión de cabezas
        #    A·V: cada token recibe información de los tokens a los que prestó atención.
        #    Luego se juntan las 4 cabezas (4 × 32 = 128) y se pasan por W_o.
        salida = (self.dropout(A) @ V).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(salida)


# =================================================================== 3. POSICIÓN
class CodificacionPosicional(nn.Module):
    """CODIFICACIÓN POSICIONAL SINUSOIDAL.
    La atención no sabe en qué orden vienen las palabras ("plastico 2" y "2 plastico"
    serían iguales para ella). Por eso a cada embedding se le suma una "firma" única
    de su posición, hecha con senos y cosenos de distintas frecuencias:
        PE(pos, 2i)   = sen(pos / 10000^(2i/d_modelo))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_modelo))
    No tiene parámetros que aprender: se calcula una sola vez.
    """

    def __init__(self, d_modelo: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)                       # posiciones 0, 1, 2, ...
        div = torch.exp(torch.arange(0, d_modelo, 2) * (-math.log(10000.0) / d_modelo))  # 1/10000^(2i/d)
        pe = torch.zeros(max_len, d_modelo)
        pe[:, 0::2] = torch.sin(pos * div)   # dimensiones pares: seno
        pe[:, 1::2] = torch.cos(pos * div)   # dimensiones impares: coseno
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # Se suma la firma de posición al embedding de cada token.
        return x + self.pe[:, : x.size(1)]


# =================================================================== 4. FEED-FORWARD
class FeedForward(nn.Module):
    """RED FEED-FORWARD: dos capas lineales con ReLU en medio (128 -> 512 -> 128),
    aplicada a cada token por separado. Después de que la atención mezcla información
    entre tokens, esta red "procesa" esa información dentro de cada token."""

    def __init__(self, d_modelo, d_ff, dropout):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(d_modelo, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_modelo))

    def forward(self, x):
        return self.red(x)


# =================================================================== 5. CODIFICADOR
class CapaCodificador(nn.Module):
    """UNA CAPA DEL CODIFICADOR (el modelo tiene 3 apiladas).
    El codificador LEE la entrada (salida de la CNN + pregunta) y construye una
    representación de cada token que tiene en cuenta todo su contexto.
      1. Auto-atención: cada token de la entrada mira a todos los demás
         (por ejemplo, el "2" se asocia con "plastico").
      2. Red feed-forward.
    Cada subcapa usa:
      - LayerNorm (antes de la subcapa, "pre-norm"): normaliza cada vector para que el
        entrenamiento sea estable.
      - Conexión residual (x + subcapa(x)): la entrada se suma a la salida; facilita que
        el gradiente fluya en redes profundas y que la capa solo aprenda "correcciones".
      - Dropout: apaga neuronas al azar en entrenamiento para evitar el sobreajuste.
    """

    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.atencion = AtencionMultiCabeza(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mascara):
        y = self.n1(x)
        x = x + self.drop(self.atencion(y, y, y, mascara))     # auto-atención (Q, K y V = la entrada) + residual
        return x + self.drop(self.ff(self.n2(x)))               # red feed-forward + residual


# =================================================================== 6. DECODIFICADOR
class CapaDecodificador(nn.Module):
    """UNA CAPA DEL DECODIFICADOR (el modelo tiene 3 apiladas).
    El decodificador ESCRIBE la respuesta palabra por palabra. Tiene TRES subcapas:
      1. Auto-atención ENMASCARADA: mira solo las palabras que ya escribió (no el futuro).
      2. Atención CRUZADA: Q sale de lo que va escribiendo; K y V salen del codificador.
         Aquí es donde el modelo CONSULTA LA SALIDA DE LA CNN mientras escribe
         (se ve en el mapa de atención del notebook 02).
      3. Red feed-forward.
    """

    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.auto_atencion = AtencionMultiCabeza(d, h, dropout)
        self.atencion_cruzada = AtencionMultiCabeza(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memoria, mascara_tgt, mascara_src):
        # memoria = salida del codificador (la representación de la entrada)
        y = self.n1(x)
        x = x + self.drop(self.auto_atencion(y, y, y, mascara_tgt))          # no mira el futuro (máscara causal)
        y = self.n2(x)
        x = x + self.drop(self.atencion_cruzada(y, memoria, memoria, mascara_src))  # mira la entrada (salida de la CNN)
        return x + self.drop(self.ff(self.n3(x)))


# =================================================================== 7. MODELO COMPLETO
class EcoTransformer(nn.Module):
    """TRANSFORMER CODIFICADOR-DECODIFICADOR completo (1 483 787 parámetros).
    Hiperparámetros usados: d_modelo=128, 4 cabezas, 3 capas de codificador,
    3 capas de decodificador, red feed-forward de 512, dropout 0,1.
    """

    def __init__(self, vocab_src: int, vocab_tgt: int, d_modelo=128, n_cabezas=4, n_capas=3,
                 d_ff=512, dropout=0.1, max_len=256):
        super().__init__()
        self.d = d_modelo
        # EMBEDDINGS: tabla aprendida que convierte cada ID de token en un vector de 128 números.
        # Con el entrenamiento, las palabras con significado parecido terminan con vectores parecidos.
        self.emb_src = nn.Embedding(vocab_src, d_modelo, padding_idx=PAD)   # entrada: 204 tokens
        self.emb_tgt = nn.Embedding(vocab_tgt, d_modelo, padding_idx=PAD)   # salida: 267 tokens
        self.pos = CodificacionPosicional(d_modelo, max_len)
        self.drop = nn.Dropout(dropout)
        self.codificador = nn.ModuleList([CapaCodificador(d_modelo, n_cabezas, d_ff, dropout) for _ in range(n_capas)])
        self.decodificador = nn.ModuleList([CapaDecodificador(d_modelo, n_cabezas, d_ff, dropout) for _ in range(n_capas)])
        self.n_enc, self.n_dec = nn.LayerNorm(d_modelo), nn.LayerNorm(d_modelo)
        # Capa de salida: convierte cada vector de 128 en 267 puntajes (uno por palabra posible).
        self.salida = nn.Linear(d_modelo, vocab_tgt)
        # ENTRENAMIENTO DESDE CERO: todos los pesos empiezan con valores aleatorios
        # (inicialización de Xavier, que elige la escala adecuada para que el entrenamiento arranque estable).
        for p in self.parameters():  # inicialización aleatoria (Xavier)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def mascara_src(src):
        # MÁSCARA DE RELLENO: True en los tokens reales, False en los <pad>.
        # Así ningún token le presta atención al relleno.
        return (src != PAD)[:, None, None, :]

    @staticmethod
    def mascara_tgt(tgt):
        # MÁSCARA CAUSAL: matriz triangular inferior. La palabra en la posición i solo puede
        # mirar las posiciones 0..i (las que ya se escribieron), nunca las futuras.
        #   Ejemplo con 4 palabras:   1 0 0 0
        #                             1 1 0 0
        #                             1 1 1 0
        #                             1 1 1 1
        L = tgt.size(1)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=tgt.device))
        return (tgt != PAD)[:, None, None, :] & causal[None, None]

    def codificar(self, src):
        """CODIFICADOR: IDs de la entrada -> embeddings (× √d_modelo para equilibrar su escala
        con la de la posición) + posición -> 3 capas de codificador."""
        m = self.mascara_src(src)
        x = self.drop(self.pos(self.emb_src(src) * math.sqrt(self.d)))
        for capa in self.codificador:
            x = capa(x, m)
        return self.n_enc(x), m

    def decodificar(self, tgt, memoria, m_src):
        """DECODIFICADOR: lo escrito hasta ahora + memoria del codificador -> puntajes (logits)
        para la siguiente palabra en cada posición."""
        x = self.drop(self.pos(self.emb_tgt(tgt) * math.sqrt(self.d)))
        m_tgt = self.mascara_tgt(tgt)
        for capa in self.decodificador:
            x = capa(x, memoria, m_tgt, m_src)
        return self.salida(self.n_dec(x))

    def forward(self, src, tgt_entrada):
        # Se usa en el ENTRENAMIENTO con TEACHER FORCING: el decodificador recibe la respuesta
        # correcta desplazada una posición ("<sos> Plástico (") y debe predecir la siguiente
        # palabra en cada posición ("Plástico ( 2"). La pérdida (entropía cruzada) compara
        # esas predicciones con la respuesta real (ver notebook 02, sección 5).
        memoria, m = self.codificar(src)
        return self.decodificar(tgt_entrada, memoria, m)

    # =============================================================== 8. GENERACIÓN
    @torch.no_grad()
    def generar(self, src, max_len: int = 200):
        """Decodificación voraz (greedy): elige en cada paso la palabra más probable.

        GENERACIÓN AUTORREGRESIVA (cuando el modelo ya está entrenado):
          1. El codificador lee la entrada una sola vez.
          2. Se empieza con <sos>.
          3. El decodificador calcula los puntajes de las 267 palabras y se elige la más
             probable (argmax).
          4. Esa palabra se agrega a lo escrito y se repite, hasta generar <eos>.
        """
        memoria, m = self.codificar(src)
        ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)    # empieza con <sos>
        terminado = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            sig = self.decodificar(ys, memoria, m)[:, -1].argmax(-1)   # palabra más probable en la última posición
            sig = torch.where(terminado, torch.full_like(sig, PAD), sig)
            ys = torch.cat([ys, sig[:, None]], 1)                     # se agrega a lo escrito
            terminado |= sig == EOS                                   # ¿ya escribió <eos>?
            if terminado.all():
                break
        return ys[:, 1:]


# =================================================================== 9. USO EN LA APP
class Retroalimentador:
    """Carga el Transformer entrenado (models/ecotransformer.pt) y lo usa en la aplicación:
    recibe el conteo de la CNN + la pregunta y devuelve la retroalimentación en texto."""

    def __init__(self, ruta, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ruta, map_location=self.device)
        self.v_src, self.v_tgt = Vocabulario(ck["vocab_src"]), Vocabulario(ck["vocab_tgt"])
        # .eval() desactiva el dropout: en uso real se quiere la red completa.
        self.modelo = EcoTransformer(len(self.v_src), len(self.v_tgt), **ck["hparams"]).to(self.device).eval()
        self.modelo.load_state_dict(ck["modelo"])

    def responder(self, conteo: dict[str, int], pregunta: str = "") -> tuple[str, str]:
        from src.retro_datos import entrada
        # 1) Salida de la CNN + pregunta -> texto de entrada: "det plastico 2 vidrio 1 sep ..."
        x = entrada(conteo, pregunta)
        # 2) Texto -> IDs (tokenización + vocabulario)
        src = torch.tensor([self.v_src.codificar(x)], device=self.device)
        # 3) El Transformer genera la respuesta palabra por palabra
        ids = self.modelo.generar(src)[0].tolist()
        if EOS in ids:
            ids = ids[: ids.index(EOS)]
        # 4) IDs -> texto
        return self.v_tgt.decodificar(ids), x
