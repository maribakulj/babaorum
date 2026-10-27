"""Texte de page : CER ordonné, insensible à l'ordre, ventilation par cause (D-212).

Une chaîne rend une suite d'**unités** (lignes, ou blocs pour un VLM) dans
son ordre de lecture ; la vérité aussi (lignes dans l'ordre de lecture
humain). Normalisation (D-150) : HIPE avec ``ſ`` replié, après avoir
recollé les coupures en fin de ligne (``mot-⏎suite`` → ``motsuite``, des
deux côtés, pour ne pas punir un lecteur qui recolle).

Mesures :

- **CER ordonné** : Levenshtein entre les deux textes de page / longueur de
  la vérité ;
- **CER réordonné** (insensible à l'ordre) : chaque unité trouvée est
  ancrée dans la vérité (trigrammes de mots communs, puis meilleur
  alignement local), les unités sont remises dans l'ordre de leurs ancres,
  et on recalcule ; les unités sans ancre sont gardées à la fin ;
- **ventilation**, en points de CER : *manqué* (caractères vrais couverts
  par aucune unité), *doublon* (unité dont l'ancre recouvre à ≥ 50 % une
  zone déjà couverte), *sans ancre* (texte qui ne se retrouve pas dans la
  page : bouillie ou invention), *lecture* (distance entre chaque unité
  ancrée et sa zone vraie), *ordre* (CER ordonné − CER réordonné) ;
- **sac de mots** : erreurs (manquants + en trop) / mots vrais, et F1.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from babaorum.legacy.data.records import normalize_hipe

_HYPH = re.compile(r"[-¬­‐‑]\s*\n\s*")


def norm(text: str) -> str:
    """D-150 : HIPE avec ``ſ`` replié."""
    return normalize_hipe(text.replace("ſ", "s"))


def page_text(units: Sequence[str]) -> str:
    """Les unités jointes, coupures de fin de ligne recollées, normalisées."""
    return norm(_HYPH.sub("", "\n".join(u.strip() for u in units)))


def cer(hyp: str, ref: str) -> float:
    return Levenshtein.distance(hyp, ref) / len(ref) if ref else 0.0


@dataclass(frozen=True)
class Anchor:
    unit: int
    start: int  # dans le texte vrai normalisé
    end: int


def _words(text: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for m in re.finditer(r"\S+", text):
        out.append((m.group(0), m.start()))
    return out


def anchor_units(units: Sequence[str], ref: str, n: int = 3) -> list[Anchor | None]:
    """Ancre chaque unité (déjà normalisée) dans ``ref`` : vote sur le
    décalage des n-grammes de mots communs (n = 3, puis 2 et 1 pour les
    unités courtes, seulement si le n-gramme est unique dans la page),
    puis meilleur alignement local dans une fenêtre autour du décalage."""
    rw = _words(ref)
    index: dict[int, dict[tuple[str, ...], list[int]]] = {}
    for k in (3, 2, 1):
        d: dict[tuple[str, ...], list[int]] = {}
        for i in range(len(rw) - k + 1):
            d.setdefault(tuple(w for w, _ in rw[i : i + k]), []).append(rw[i][1])
        index[k] = d
    out: list[Anchor | None] = []
    for u, unit in enumerate(units):
        uw = _words(unit)
        if not uw:
            out.append(None)
            continue
        votes: Counter[int] = Counter()
        for k in (n, 2, 1):
            if len(uw) < k:
                continue
            for i in range(len(uw) - k + 1):
                key = tuple(w for w, _ in uw[i : i + k])
                pos = index[k].get(key)
                if pos is None:
                    continue
                if k == 3 and len(pos) <= 3:
                    for p in pos:
                        votes[(p - uw[i][1]) // 16] += 3
                elif k < 3 and len(pos) == 1:
                    votes[(pos[0] - uw[i][1]) // 16] += k
            if votes and (k == 3 or len(uw) <= 3):
                break
        if not votes:
            out.append(None)
            continue
        off = votes.most_common(1)[0][0] * 16
        lo = max(0, off - len(unit) // 2 - 16)
        hi = min(len(ref), off + len(unit) + len(unit) // 2 + 32)
        window = ref[lo:hi]
        if not window:
            out.append(None)
            continue
        al = fuzz.partial_ratio_alignment(unit, window)
        if al is None or al.score < 50:
            out.append(None)
            continue
        if len(unit) <= len(window):
            s, e = lo + al.dest_start, lo + al.dest_end
        else:  # rapidfuzz a échangé aiguille et botte de foin
            s, e = lo, hi
        out.append(Anchor(u, s, e))
    return out


#: Versions des deux CER réordonnés que rend :func:`score_page`, la seule
#: fonction de score de page (C6, N01, D-490).
METRIC = {
    "cer_reordered": "D-217 : suites, puis min avec l'ordonné (15136ad)",
    "cer_reordered_d212": "D-212 : tri unité par unité (15136ad^, E13)",
}

#: D-217 : saut toléré après la fin de l'ancre précédente (lignes manquées)
JUMP = 100
#: D-217 : une unité plus courte ne coupe ni ne commence de suite
RELIABLE = 10


def reorder_runs(
    units: Sequence[str],
    anchors: Sequence[Anchor | None],
    jump: int = JUMP,
    reliable: int = RELIABLE,
) -> str:
    """D-217 : remontage par **suites**. Une suite continue tant que l'ancre
    de l'unité suivante tombe juste après la fin de la précédente (de −20 à
    ``jump`` caractères) ; sinon une nouvelle suite commence. Seules les
    unités d'au moins ``reliable`` caractères ancrées comptent : une ligne
    courte (souvent mal ancrée) reste dans la suite où la chaîne l'a mise,
    comme les unités sans ancre. Dans une suite, l'ordre de la chaîne est
    gardé ; les suites sont remises dans l'ordre de leur première ancre
    (suites sans ancre à la fin)."""
    runs: list[list[int]] = []
    starts: list[int | None] = []
    last_end: int | None = None
    for i, a in enumerate(anchors):
        ok = a is not None and len(units[i]) >= reliable
        if ok and a is not None:
            if last_end is None or not (last_end - 20 <= a.start <= last_end + jump):
                if runs and starts[-1] is None:
                    starts[-1] = a.start  # suite ouverte sans ancre : elle la prend
                else:
                    runs.append([])
                    starts.append(a.start)
            last_end = a.end
        if not runs:
            runs.append([])
            starts.append(None)
        runs[-1].append(i)
    big = 1 << 60
    order = sorted(
        range(len(runs)), key=lambda k: (starts[k] if starts[k] is not None else big, k)
    )
    return " ".join(units[i] for k in order for i in runs[k])


@dataclass
class PageScore:
    chars: int
    cer: float  # ordonné, D-212
    cer_units: float  # ordonné, unités normalisées une à une (base de la ventilation)
    cer_reordered: float
    missed: float
    duplicate: float
    unanchored: float
    reading: float
    bow_error: float
    bow_f1: float
    #: CER réordonné **historique** (D-212, avant D-217, 15136ad^) : unités
    #: triées une à une par leur ancre, sans ancre à la fin, sans minimum.
    #: Métrique du « 4,05 % » de Gallica publié par E13, d'où vient le seuil
    #: « < 4 % » ; rapporté à côté du chiffre en vigueur, jamais à sa place
    #: (R-2, C6, D-490).
    cer_reordered_d212: float = 0.0

    @property
    def order(self) -> float:
        return self.cer_units - self.cer_reordered


def bag_of_words(hyp: str, ref: str) -> tuple[float, float]:
    h, r = Counter(hyp.split()), Counter(ref.split())
    inter = sum((h & r).values())
    nr, nh = sum(r.values()), sum(h.values())
    err = (nr - inter + nh - inter) / nr if nr else 0.0
    p = inter / nh if nh else 0.0
    rc = inter / nr if nr else 0.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return err, f


def score_page(hyp_units: Sequence[str], ref_units: Sequence[str]) -> PageScore:
    ref = page_text(ref_units)
    hyp = page_text(hyp_units)
    n = len(ref)
    units = [norm(u) for u in hyp_units]
    units = [u for u in units if u]
    ref_u = " ".join(norm(u) for u in ref_units if norm(u))
    joined = " ".join(units)
    anchors = anchor_units(units, ref_u)
    placed = sorted(
        (a for a in anchors if a is not None), key=lambda a: (a.start, a.end)
    )
    loose = [units[i] for i, a in enumerate(anchors) if a is None]
    reordered = reorder_runs(units, anchors)
    reordered_d212 = " ".join([units[a.unit] for a in placed] + loose)
    nu = len(ref_u)
    covered = bytearray(nu)
    dup = read = 0
    for a in placed:
        span = max(1, a.end - a.start)
        already = sum(covered[a.start : a.end])
        if already >= 0.5 * span:
            dup += len(units[a.unit])
            continue
        for k in range(a.start, a.end):
            covered[k] = 1
        read += Levenshtein.distance(units[a.unit], ref_u[a.start : a.end])
    # une espace non couverte ne compte que dans une suite non couverte
    # (les espaces entre deux unités ancrées ne sont couvertes par aucune)
    missed = sum(
        1
        for k in range(nu)
        if not covered[k]
        and (
            ref_u[k] != " "
            or (0 < k < nu - 1 and not covered[k - 1] and not covered[k + 1])
        )
    )
    bow_e, bow_f = bag_of_words(hyp, ref)
    d = nu or 1
    return PageScore(
        chars=n,
        cer=cer(hyp, ref),
        cer_units=cer(joined, ref_u),
        # D-217 : on ne réordonne que si cela rapproche de la vérité
        cer_reordered=min(cer(reordered, ref_u), cer(joined, ref_u)),
        missed=missed / d,
        duplicate=dup / d,
        unanchored=sum(len(u) for u in loose) / d,
        reading=read / d,
        bow_error=bow_e,
        bow_f1=bow_f,
        cer_reordered_d212=cer(reordered_d212, ref_u),
    )
