"""Une ligne appariée OCR / vérité terrain, son CER initial et sa strate.

Le CER initial (``cer0``) décide de la strate. Il est calculé après la
normalisation du scorer HIPE (:func:`normalize_hipe`, définie dans
``axel.reader.metric`` et réexportée ici) pour
que les strates de ce dépôt et celles des vues ``cinoc`` au profil ``hipe``
coïncident (D-11). Le CER sur texte brut est gardé à côté (``cer0_raw``).
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rapidfuzz.distance import Levenshtein

from babaorum.legacy.reader import metric as _metric

Stratum = Literal["clean", "moderate", "heavy", "out"]
Split = Literal["train", "dev", "test"]

#: Bornes des strates, en CER initial (D-6).
CLEAN_MAX = 0.02
MODERATE_MAX = 0.15
HEAVY_MAX = 0.30

#: Lignes écartées : la raison est le nom du compteur.
DISCARD_EMPTY_TRUTH = "truth_empty"
DISCARD_EMPTY_SOURCE = "source_empty"
DISCARD_LENGTH_RATIO = "length_ratio"
DISCARD_CER_ABOVE_60 = "cer_above_60"

#: Normalisation HIPE : une seule définition, dans ``axel.reader.metric`` (C3).
_HIPE_TABLE = _metric.HIPE_TABLE
normalize_hipe = _metric.normalize_hipe


def cer(source: str, truth: str) -> float:
    """Distance de Levenshtein sur la longueur de la vérité (``inf`` si vide)."""
    if not truth:
        return float("inf")
    return Levenshtein.distance(source, truth) / len(truth)


def stratum_of(cer0: float) -> Stratum:
    """La strate d'une ligne d'après son CER initial (bornes D-6)."""
    if cer0 < CLEAN_MAX:
        return "clean"
    if cer0 <= MODERATE_MAX:
        return "moderate"
    if cer0 <= HEAVY_MAX:
        return "heavy"
    return "out"


@dataclass(frozen=True)
class Line:
    """Une unité (ligne ou phrase) OCR appariée à sa vérité terrain."""

    doc_id: str
    unit_id: str
    source: str
    truth: str
    cer0: float
    cer0_raw: float
    stratum: Stratum
    origin: str  # icdar2017, impresso, ocr17, bnl
    unit: str  # line | sentence | chunk
    split: Split
    #: Type SegmOnto du bloc (Gallicorpora : MainZone, MarginTextZone,
    #: RunningTitleZone, NumberingZone…) ; ``None`` quand la source n'en a pas.
    zone: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Line:
        return cls(**json.loads(text))


class Discarded(Counter[str]):
    """Compte les lignes écartées, par raison."""


def make_line(
    *,
    doc_id: str,
    unit_id: str,
    source: str,
    truth: str,
    origin: str,
    unit: str,
    split: Split,
    discarded: Discarded,
    zone: str | None = None,
) -> Line | None:
    """Construit la ligne, ou ``None`` en comptant la raison du rejet.

    Rejets : vérité vide, source vide, ratio de longueur hors [0,5 ; 2],
    CER normalisé au-dessus de 60 %.
    """
    # NFC, comme les parseurs et les ``ReplaceSpan`` de saknussemm (VR-14) :
    # les ALTO eScriptorium de Gallicorpora sont en NFD (``õ`` = o + U+0303)
    source = unicodedata.normalize("NFC", source.strip())
    truth = unicodedata.normalize("NFC", truth.strip())
    if not truth:
        discarded[DISCARD_EMPTY_TRUTH] += 1
        return None
    if not source:
        discarded[DISCARD_EMPTY_SOURCE] += 1
        return None
    ratio = len(source) / len(truth)
    if not 0.5 <= ratio <= 2.0:
        discarded[DISCARD_LENGTH_RATIO] += 1
        return None
    norm_source, norm_truth = normalize_hipe(source), normalize_hipe(truth)
    cer0 = cer(norm_source, norm_truth) if norm_truth else cer(source, truth)
    if cer0 > 0.60:
        discarded[DISCARD_CER_ABOVE_60] += 1
        return None
    return Line(
        doc_id=doc_id,
        unit_id=unit_id,
        source=source,
        truth=truth,
        cer0=cer0,
        cer0_raw=cer(source, truth),
        stratum=stratum_of(cer0),
        origin=origin,
        unit=unit,
        split=split,
        zone=zone,
    )


def write_lines(lines: list[Line], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(ln.to_json() + "\n" for ln in lines), encoding="utf-8")


def read_lines(path: Path) -> Iterator[Line]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield Line.from_json(raw)
