"""EcoTransformer: Transformer encoder-decoder construido y entrenado desde cero.

Implementa la arquitectura de "Attention is All You Need" (Vaswani et al., 2017):
  - Tokenizador propio a nivel de palabra.
  - Embeddings + codificación posicional sinusoidal.
  - Atención multi-cabeza programada a mano:  Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V
  - Codificador (lee la salida de la CNN + la pregunta) y decodificador (genera la
    retroalimentación palabra por palabra, de forma autorregresiva).
No se usa ningún modelo ni peso pre-entrenado.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn as nn

PAD, SOS, EOS, UNK = 0, 1, 2, 3
ESPECIALES = ["<pad>", "<sos>", "<eos>", "<unk>"]


# =================================================================== tokenizador
def tokenizar(texto: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", texto)


def destokenizar(tokens: list[str]) -> str:
    t = " ".join(tokens)
    t = re.sub(r"\s+([.,:;)?!])", r"\1", t)
    t = re.sub(r"([(¿¡])\s+", r"\1", t)
    return t


class Vocabulario:
    def __init__(self, tokens: list[str]):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def construir(cls, textos: list[str], min_frec: int = 1):
        c = Counter(t for x in textos for t in tokenizar(x))
        return cls(ESPECIALES + sorted(t for t, f in c.items() if f >= min_frec))

    def codificar(self, texto: str, extremos: bool = False) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in tokenizar(texto)]
        return [SOS] + ids + [EOS] if extremos else ids

    def decodificar(self, ids: list[int]) -> str:
        return destokenizar([self.itos[i] for i in ids if i not in (PAD, SOS, EOS)])

    def __len__(self):
        return len(self.itos)


# =================================================================== bloques
class AtencionMultiCabeza(nn.Module):
    """softmax(Q·Kᵀ/√d_k)·V calculada en paralelo para h cabezas."""

    def __init__(self, d_modelo: int, n_cabezas: int, dropout: float = 0.1):
        super().__init__()
        assert d_modelo % n_cabezas == 0
        self.h, self.d_k = n_cabezas, d_modelo // n_cabezas
        self.W_q = nn.Linear(d_modelo, d_modelo)
        self.W_k = nn.Linear(d_modelo, d_modelo)
        self.W_v = nn.Linear(d_modelo, d_modelo)
        self.W_o = nn.Linear(d_modelo, d_modelo)
        self.dropout = nn.Dropout(dropout)
        self.pesos_atencion = None  # se guardan para visualizarlos

    def forward(self, q, k, v, mascara=None):
        B = q.size(0)
        # 1) Proyecciones lineales y separación en h cabezas: (B, h, L, d_k)
        Q = self.W_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Puntajes de atención: Q·Kᵀ / √d_k
        puntajes = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mascara is not None:
            puntajes = puntajes.masked_fill(~mascara, float("-inf"))
        # 3) softmax -> distribución de probabilidad sobre las posiciones
        A = torch.softmax(puntajes, dim=-1)
        self.pesos_atencion = A.detach()
        # 4) Suma ponderada de los valores y unión de cabezas
        salida = (self.dropout(A) @ V).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(salida)


class CodificacionPosicional(nn.Module):
    def __init__(self, d_modelo: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_modelo, 2) * (-math.log(10000.0) / d_modelo))
        pe = torch.zeros(max_len, d_modelo)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_modelo, d_ff, dropout):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(d_modelo, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_modelo))

    def forward(self, x):
        return self.red(x)


class CapaCodificador(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.atencion = AtencionMultiCabeza(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mascara):
        y = self.n1(x)
        x = x + self.drop(self.atencion(y, y, y, mascara))     # auto-atención
        return x + self.drop(self.ff(self.n2(x)))               # red feed-forward


class CapaDecodificador(nn.Module):
    def __init__(self, d, h, d_ff, dropout):
        super().__init__()
        self.auto_atencion = AtencionMultiCabeza(d, h, dropout)
        self.atencion_cruzada = AtencionMultiCabeza(d, h, dropout)
        self.ff = FeedForward(d, d_ff, dropout)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memoria, mascara_tgt, mascara_src):
        y = self.n1(x)
        x = x + self.drop(self.auto_atencion(y, y, y, mascara_tgt))          # no mira el futuro
        y = self.n2(x)
        x = x + self.drop(self.atencion_cruzada(y, memoria, memoria, mascara_src))  # mira la entrada
        return x + self.drop(self.ff(self.n3(x)))


# =================================================================== modelo completo
class EcoTransformer(nn.Module):
    def __init__(self, vocab_src: int, vocab_tgt: int, d_modelo=128, n_cabezas=4, n_capas=3,
                 d_ff=512, dropout=0.1, max_len=256):
        super().__init__()
        self.d = d_modelo
        self.emb_src = nn.Embedding(vocab_src, d_modelo, padding_idx=PAD)
        self.emb_tgt = nn.Embedding(vocab_tgt, d_modelo, padding_idx=PAD)
        self.pos = CodificacionPosicional(d_modelo, max_len)
        self.drop = nn.Dropout(dropout)
        self.codificador = nn.ModuleList([CapaCodificador(d_modelo, n_cabezas, d_ff, dropout) for _ in range(n_capas)])
        self.decodificador = nn.ModuleList([CapaDecodificador(d_modelo, n_cabezas, d_ff, dropout) for _ in range(n_capas)])
        self.n_enc, self.n_dec = nn.LayerNorm(d_modelo), nn.LayerNorm(d_modelo)
        self.salida = nn.Linear(d_modelo, vocab_tgt)
        for p in self.parameters():  # inicialización aleatoria (Xavier)
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def mascara_src(src):
        return (src != PAD)[:, None, None, :]

    @staticmethod
    def mascara_tgt(tgt):
        L = tgt.size(1)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=tgt.device))
        return (tgt != PAD)[:, None, None, :] & causal[None, None]

    def codificar(self, src):
        m = self.mascara_src(src)
        x = self.drop(self.pos(self.emb_src(src) * math.sqrt(self.d)))
        for capa in self.codificador:
            x = capa(x, m)
        return self.n_enc(x), m

    def decodificar(self, tgt, memoria, m_src):
        x = self.drop(self.pos(self.emb_tgt(tgt) * math.sqrt(self.d)))
        m_tgt = self.mascara_tgt(tgt)
        for capa in self.decodificador:
            x = capa(x, memoria, m_tgt, m_src)
        return self.salida(self.n_dec(x))

    def forward(self, src, tgt_entrada):
        memoria, m = self.codificar(src)
        return self.decodificar(tgt_entrada, memoria, m)

    @torch.no_grad()
    def generar(self, src, max_len: int = 200):
        """Decodificación voraz (greedy): elige en cada paso la palabra más probable."""
        memoria, m = self.codificar(src)
        ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
        terminado = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            sig = self.decodificar(ys, memoria, m)[:, -1].argmax(-1)
            sig = torch.where(terminado, torch.full_like(sig, PAD), sig)
            ys = torch.cat([ys, sig[:, None]], 1)
            terminado |= sig == EOS
            if terminado.all():
                break
        return ys[:, 1:]


# =================================================================== uso en la app
class Retroalimentador:
    def __init__(self, ruta, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ruta, map_location=self.device)
        self.v_src, self.v_tgt = Vocabulario(ck["vocab_src"]), Vocabulario(ck["vocab_tgt"])
        self.modelo = EcoTransformer(len(self.v_src), len(self.v_tgt), **ck["hparams"]).to(self.device).eval()
        self.modelo.load_state_dict(ck["modelo"])

    def responder(self, conteo: dict[str, int], pregunta: str = "") -> tuple[str, str]:
        from src.retro_datos import entrada
        x = entrada(conteo, pregunta)
        src = torch.tensor([self.v_src.codificar(x)], device=self.device)
        ids = self.modelo.generar(src)[0].tolist()
        if EOS in ids:
            ids = ids[: ids.index(EOS)]
        return self.v_tgt.decodificar(ids), x
