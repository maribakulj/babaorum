"""LM neuronal caractère (R2b) : LSTM, et sa notation incrémentale pour le faisceau.

Le texte est en NFC ; un caractère hors vocabulaire est noté comme <unk>.
:class:`LMScorer` garde en cache, pour chaque préfixe, l'état du LSTM et la
distribution du caractère suivant : un pas de LSTM par préfixe nouveau, quel
que soit le nombre de candidats essayés derrière lui. Interface identique à
celle qu'attend ``ancre.multi_ctc_beam`` : ``lm(texte, car)`` = log P(car |
texte), ``car = None`` pour la fin de ligne.
"""

from __future__ import annotations

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class CharLSTM(nn.Module):
    def __init__(self, vocab: list[str], emb: int = 64, hid: int = 512, layers: int = 2) -> None:
        super().__init__()
        self.vocab = list(vocab)
        self.index = {c: i + 4 for i, c in enumerate(self.vocab)}
        n = len(self.vocab) + 4
        self.emb = nn.Embedding(n, emb)
        self.rnn = nn.LSTM(emb, hid, layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hid, n)

    def ids(self, text: str) -> list[int]:
        return [self.index.get(c, UNK) for c in text]

    def forward(self, x: torch.Tensor, state=None):
        y, state = self.rnn(self.emb(x), state)
        return self.out(y).log_softmax(-1), state


class LMScorer:
    def __init__(self, model: CharLSTM, device: str = "cpu") -> None:
        self.m = model.to(device).eval()
        self.dev = device
        self.cache: dict[str, tuple[object, torch.Tensor]] = {}

    @torch.no_grad()
    def _dist(self, text: str) -> torch.Tensor:
        hit = self.cache.get(text)
        if hit is not None:
            return hit[1]
        if text == "":
            lp, st = self.m(torch.tensor([[BOS]], device=self.dev))
        else:
            prev_state = self.cache[text[:-1]][0] if text[:-1] in self.cache else None
            if prev_state is None:
                self._dist(text[:-1])
                prev_state = self.cache[text[:-1]][0]
            lp, st = self.m(torch.tensor([[self.m.index.get(text[-1], UNK)]], device=self.dev), prev_state)
        d = lp[0, -1].cpu()
        if len(self.cache) > 200_000:
            self.cache.clear()
        self.cache[text] = (st, d)
        return d

    def __call__(self, text: str, c: str | None) -> float:
        d = self._dist(text)
        if c is None:
            return float(d[EOS])
        return float(d[self.m.index.get(c, UNK)])
