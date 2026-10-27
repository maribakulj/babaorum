"""R6 — architectures du banc de lecteurs (docs/READER_R6_PREREG.txt).

Chaque bras prend une ligne en niveaux de gris [B, 1, h_in, W] (blanc = 1,
remplissage de lot en blanc) ; l'image est inversée à l'entrée (fond = 0) pour
que le remplissage implicite des convolutions (zéros) soit du blanc : une
ligne lue seule ou en lot donne la même sortie. Le modèle
rend des log-probabilités CTC [B, T, C] avec T = W // pas. Décodeur CTC partout.
``ARMS[nom].describe()`` donne la liste exacte des couches (écrite dans le
pré-enregistrement avant la fumée). Aucune dépendance à axel.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def conv_bn(cin: int, cout: int, k=3, s=1, p=1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PosEnc1D(nn.Module):
    def __init__(self, d: int, max_len: int = 4000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]][None]


def key_padding(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None] >= lengths[:, None]


class BiLSTMHead(nn.Module):
    def __init__(self, d: int, hid: int, layers: int, n_out: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(d, hid, layers, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2 * hid, n_out)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y = nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=x.shape[1])[0]
        return self.out(y)


class ConformerBlock(nn.Module):
    """Demi-FFN, attention, convolution profondeur 15, demi-FFN (Gulati et al.)."""

    def __init__(self, d: int, heads: int = 4, k: int = 15) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(0.1), nn.Linear(4 * d, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(0.1), nn.Linear(4 * d, d))
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        x = x + self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        h = self.ln_conv(x).masked_fill(pad[..., None], 0).transpose(1, 2)
        h = nn.functional.glu(self.pw1(h), dim=1)
        h = self.pw2(nn.functional.silu(self.bn(self.dw(h))))
        x = x + h.transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class Reader(nn.Module):
    """Squelette commun : tronc 2D -> repli de la hauteur -> [mélangeur] -> tête."""

    name = "?"
    h_in = 64
    stride = 4

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{self.name} h_in={self.h_in} pas={self.stride} trames/hauteur={self.h_in // self.stride} params={n}\n{self}"

    def lengths(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, self.stride, rounding_mode="floor").clamp(min=1)


class B0(Reader):
    """CRNN de type kraken : 4 conv 3×3 (32, 32, 64, 64) avec 2 max-pool 2×2 (pas 4)
    et 2 max-pool 2×1 (hauteur), repli de la hauteur par moyenne, BiLSTM 3×200."""

    name, h_in, stride = "B0", 64, 4

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            conv_bn(32, 32), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            conv_bn(32, 64), nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(0.1),
            conv_bn(64, 64), nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(0.1),
        )
        self.proj = nn.Linear(64 * 4, 256)
        self.head = BiLSTMHead(256, 200, 3, n_out)

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.cnn(1.0 - x)  # [B, 64, 4, W/4] ; inversion : fond = 0
        f = f.flatten(1, 2).transpose(1, 2)  # [B, W/4, 256]
        L = self.lengths(widths)
        return self.head(self.proj(f), L).log_softmax(-1), L


class ResTrunk(nn.Module):
    """ResNet-18 tronqué : conv 7×7 pas 2 + max-pool pas 2 (pas 4 en largeur),
    layer1 (64), layer2 (128, pas 2×1), layer3 (256, pas 2×1), sans layer4 ;
    hauteur 64 -> 4, repliée par moyenne. Sortie [B, W/4, 256]."""

    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int, s) -> nn.Module:
            return _BasicBlock(cin, cout, s)

        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        self.layer1 = nn.Sequential(block(64, 64, 1), block(64, 64, 1))
        self.layer2 = nn.Sequential(block(64, 128, (2, 1)), block(128, 128, 1))
        self.layer3 = nn.Sequential(block(128, 256, (2, 1)), block(256, 256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.layer3(self.layer2(self.layer1(self.stem(1.0 - x))))  # inversion : fond = 0
        return f.mean(2).transpose(1, 2)


class _BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, s) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.down = None if (cin == cout and s == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, s, bias=False),
                                                                        nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + (x if self.down is None else self.down(x)))


class B1(Reader):
    """ResTrunk + BiLSTM 2×256."""

    name, h_in, stride = "B1", 64, 4

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.trunk = ResTrunk()
        self.head = BiLSTMHead(256, 256, 2, n_out)

    def forward(self, x, widths):
        L = self.lengths(widths)
        return self.head(self.trunk(x), L).log_softmax(-1), L


class B2(Reader):
    """ResTrunk + 2 couches transformer (d 256, 4 têtes, FFN 1 024, pré-norme,
    positions sinusoïdales, masque de remplissage) + BiLSTM 2×256."""

    name, h_in, stride = "B2", 64, 4

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.trunk = ResTrunk()
        self.pos = PosEnc1D(256)
        layer = nn.TransformerEncoderLayer(256, 4, 1024, 0.1, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = BiLSTMHead(256, 256, 2, n_out)

    def mixed(self, x, widths):
        L = self.lengths(widths)
        f = self.trunk(x)
        return self.mix(self.pos(f), src_key_padding_mask=key_padding(L, f.shape[1])), L

    def forward(self, x, widths):
        f, L = self.mixed(x, widths)
        return self.head(f, L).log_softmax(-1), L


class B3(Reader):
    """SVTR léger : tronc conv 3×3 (32 pas 2, 64 pas 2, 128 pas 2) -> grille
    h 12 × W/8, 2 blocs MHSA globaux (d 128, 4 têtes) sur les jetons 2D avec
    positions sinusoïdales séparables, repli de la hauteur par moyenne, tête
    linéaire CTC (pas de BiLSTM)."""

    name, h_in, stride = "B3", 96, 8

    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, 32, s=2), conv_bn(32, 64, s=2), conv_bn(64, 128, s=2))
        layer = nn.TransformerEncoderLayer(128, 4, 512, 0.1, batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pos_w = PosEnc1D(128)
        self.pos_h = nn.Parameter(torch.zeros(12, 128))
        self.out = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, n_out))

    def forward(self, x, widths):
        f = self.stem(1.0 - x)  # [B, 128, 12, W/8] ; inversion : fond = 0
        B, C, H, W = f.shape
        L = self.lengths(widths)
        t = f.permute(0, 2, 3, 1)  # [B, H, W, C]
        t = t + self.pos_w.pe[:W][None, None] + self.pos_h[:H][None, :, None]
        pad = key_padding(L, W)[:, None].expand(B, H, W).reshape(B, H * W)
        t = self.mix(t.reshape(B, H * W, C), src_key_padding_mask=pad).reshape(B, H, W, C)
        return self.out(t.mean(1)).log_softmax(-1), L


class B4(B2):
    """B2 avec 2 blocs Conformer (d 256, 4 têtes, convolution 15) à la place du
    BiLSTM ; tête linéaire CTC."""

    name = "B4"

    def __init__(self, n_out: int) -> None:
        super().__init__(n_out)
        self.head = None
        self.conf = nn.ModuleList([ConformerBlock(256), ConformerBlock(256)])
        self.out = nn.Linear(256, n_out)

    def forward(self, x, widths):
        f, L = self.mixed(x, widths)
        pad = key_padding(L, f.shape[1])
        for blk in self.conf:
            f = blk(f, pad)
        return self.out(f).log_softmax(-1), L


class B5(B2):
    """B2 + tête CTC intermédiaire (linéaire) en sortie des couches transformer ;
    perte = 0,7·CTC final + 0,3·CTC intermédiaire ; sans réinjection. À
    l'inférence seule la sortie finale sert (score image seul)."""

    name = "B5"
    aux_weight = 0.3

    def __init__(self, n_out: int) -> None:
        super().__init__(n_out)
        self.aux = nn.Linear(256, n_out)

    def forward(self, x, widths, with_aux: bool = False):
        f, L = self.mixed(x, widths)
        main = self.head(f, L).log_softmax(-1)
        if with_aux:
            return main, L, self.aux(f).log_softmax(-1)
        return main, L


ARMS = {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5}
