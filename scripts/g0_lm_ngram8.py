"""Contrôle 8-grammes contre les textes LM (pleias_french_pd_322 / reader_lm_corpus), contrat G0-04 v4.1 point 4 (pB, 24/10).

Normalisation : `norm` d'E6, IMPORTÉE SANS CHANGEMENT (axel/scripts/e6_contamination.py:78) : « ſ→s, NFKD, marques combinantes
retirées, minuscules, ponctuation remplacée par des espaces » :
    text = unicodedata.normalize("NFKD", text.replace("ſ", "s"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\\w]", " ", text).replace("_", " ").split())

Texte d'une page (contrat point 4) : lignes dans l'ordre ; un trait d'union en fin de ligne (-, ¬, ⸗) est retiré et le mot est
recollé à la ligne suivante ; les autres lignes sont jointes par une espace ; puis norm.
Mots : split(). Grammes : ensemble des 8-grammes DISTINCTS. Dénominateur : nombre de 8-grammes distincts de la page.
Page de moins de 50 mots après norm : verdict "incomplete" (refus), jamais PASSE ni chevauchement.
Verdict : part >= 0,5 % OU >= 3 grammes partagés -> SUIVI_TEXT_OVERLAP:<source> (refus prudent).

Sans réseau. Entrées : --texts (JSONL : file_id, text) ; --pages (JSONL : id, lines) ; sortie : un JSON par page, dans un dossier NEUF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

N = 8
MIN_WORDS = 50
SHARE_MIN = 0.005
FLOOR = 3
HYPHENS = ("-", "¬", "⸗")


def norm(text: str) -> str:
    """norm d'E6 (axel/scripts/e6_contamination.py:78), copié sans changement."""
    text = unicodedata.normalize("NFKD", text.replace("ſ", "s"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^\w]", " ", text).replace("_", " ").split())


def page_text(lines: list[str]) -> str:
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        raise ValueError("lines doit être une liste de chaînes (une par TextLine)")
    if not lines:
        raise ValueError("page sans ligne : refusée (pas de non-recouvrement sans texte)")
    out = ""
    for i, line in enumerate(lines):
        if line.endswith(HYPHENS) and i + 1 < len(lines):
            body = line[:-1]
            if body and not body[-1].isspace():
                out += body              # coupure de mot : on recolle au mot suivant, sans espace
            else:
                out += body.rstrip() + " "   # tiret isolé après une espace : ce n'est pas une coupure
        else:
            out += line + " "
    return norm(out)


def grams(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i + N]) for i in range(len(words) - N + 1)}


SPLITS = ("train", "dev", "eval")


def read_jsonl(path: Path) -> list[dict]:
    """Découpe sur LF seul : str.splitlines() couperait aussi sur des séparateurs Unicode (U+2028, U+0085...)."""
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def build_index(texts: list[dict]) -> dict[str, list[dict]]:
    """Index : gramme -> liste des textes qui le contiennent (provenance : file_id, source, split).
    Chaque texte porte OBLIGATOIREMENT sa source et son split (contrat : train/dev refusent, eval étiquette)."""
    if not texts:
        raise ValueError("aucun texte d'index : refusé (un index vide ferait passer toute page)")
    for t in texts:
        if t.get("split") not in SPLITS or not t.get("source") or not t.get("file_id"):
            raise ValueError(f"texte sans file_id/source/split valide : {t.get('file_id')!r}")
        if not isinstance(t.get("text"), str) or not t["text"].strip():
            raise ValueError(f"texte vide ou non textuel : {t.get('file_id')!r}")
    idx: dict[str, list[dict]] = {}
    for t in texts:
        for g in grams(norm(t["text"])):
            idx.setdefault(g, []).append({"file_id": t["file_id"], "source": t["source"], "split": t["split"]})
    return idx


def _hit(n_shared: int, n_grams: int, n_words: int) -> tuple[bool, float | None]:
    """Seuils du contrat, appliqués à UN groupe de splits (jamais agrégés)."""
    if n_words < MIN_WORDS:
        return n_shared >= FLOOR, None
    share = (n_shared / n_grams) if n_grams else 0.0
    return (share >= SHARE_MIN or n_shared >= FLOOR), share


def check_page(page_id: str, lines: list[str], idx: dict[str, list[dict]]) -> dict:
    """Seuils calculés PAR SOURCE et PAR CATÉGORIE (refus train/dev, étiquette eval) : aucun gramme n'est cumulé entre sources.
    Une source ne refuse que si SES propres grammes partagés atteignent le seuil du contrat."""
    text = page_text(lines)
    nwords = len(norm(text).split())      # même règle que CONTENU : compte APRÈS normalisation
    g = grams(text)
    if nwords < MIN_WORDS:               # moins de 50 mots : pas de verdict PASSE, un refus (incomplete)
        return {"page": page_id, "mots": nwords, "grammes_distincts": len(g), "par_source": {},
                "verdict": "incomplete", "verdicts": ["NGRAM8_INCOMPLETE"]}
    per: dict[tuple[str, str], set[str]] = {}          # (source, catégorie) -> grammes partagés par CETTE source
    files: dict[tuple[str, str], set[str]] = {}
    for x in g:
        for h in idx.get(x, ()):
            cat = "refus" if h["split"] in ("train", "dev") else "eval"
            key = (h["source"], cat)
            per.setdefault(key, set()).add(x)
            files.setdefault(key, set()).add(h["file_id"])
    verdicts, detail = [], {}
    for (src, cat), shared in sorted(per.items()):
        hit, share = _hit(len(shared), len(g), nwords)
        detail[f"{src}|{cat}"] = {"grammes_partages": len(shared), "part": share, "textes": sorted(files[(src, cat)])}
        if hit:
            verdicts.append(f"SUIVI_TEXT_OVERLAP:{src}" if cat == "refus" else f"TEXT_OVERLAP_EVAL:{src}")
    return {"page": page_id, "mots": nwords, "grammes_distincts": len(g), "par_source": detail,
            "verdict": (sorted(verdicts)[0] if verdicts else "PASSE"), "verdicts": sorted(verdicts)}

def load_lm_texts(corpus_dir: Path, lm_json: Path) -> list[dict]:
    """Construit les 322 entrées annotées (file_id, source, split, text) depuis le corpus réel, EN VÉRIFIANT lm_texts.json :
    SHA du fichier texte, du manifeste, de chaque texte ; compte exact ; file_id uniques ; tous les kept_file_ids présents.
    Toute écart lève ValueError (aucun index partiel n'est produit)."""
    import gzip
    data = json.loads(lm_json.read_text(encoding="utf-8"))
    lm = data["lm_texts"]
    txt = corpus_dir / lm["root"]["path"]
    man = corpus_dir / lm["manifest"]["path"]
    if hashlib.sha256(txt.read_bytes()).hexdigest() != lm["sha256_fichier"]:
        raise ValueError("SHA du fichier texte différent de lm_texts.json")
    if hashlib.sha256(man.read_bytes()).hexdigest() != lm["manifest"]["sha256"]:
        raise ValueError("SHA du manifeste différent de lm_texts.json")
    kept = json.loads(man.read_text(encoding="utf-8"))["kept_file_ids"]
    if len(kept) != lm["count_attendu"] or len(set(kept)) != len(kept):
        raise ValueError("compte ou unicité des kept_file_ids différents de lm_texts.json")
    by_id = {}
    with gzip.open(txt, "rt", encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["file_id"] in by_id:
                raise ValueError(f"file_id en double dans le fichier texte : {r['file_id']}")
            by_id[r["file_id"]] = r
    missing = [i for i in kept if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} kept_file_ids absents du fichier texte")
    expected = {x["file_id"]: x["sha256_texte"] for x in data["per_text"]}   # per_text est à la RACINE de lm_texts.json
    out = []
    for fid in kept:
        t = by_id[fid]["text"]
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"texte vide : {fid}")
        if hashlib.sha256(t.encode("utf-8")).hexdigest() != expected.get(fid):
            raise ValueError(f"SHA du texte différent de lm_texts.json : {fid}")
        out.append({"file_id": fid, "source": "pleias_french_pd_322", "split": "train", "text": t})
    return out


def lm_files_sha(corpus_dir: Path, lm_json: Path) -> list[dict]:
    """Fichiers effectivement comparés, avec SHA : texte et manifeste de lm_texts (chemins relatifs à corpus_root)."""
    data = json.loads(lm_json.read_text(encoding="utf-8"))
    lm = data["lm_texts"]
    return [{"path": lm["root"]["path"], "sha256": lm["sha256_fichier"]},
            {"path": lm["manifest"]["path"], "sha256": lm["manifest"]["sha256"]}]


def load_reader_entries(text_gz: Path, filter_json: Path, press_jsonl: Path) -> list[dict]:
    """Textes du LM du lecteur (reader_lm_corpus), EXACTEMENT :
    (a) les textes de text.jsonl.gz dont le file_id est dans la liste des 109 documents appris (R2_LM_documents_gallica.json, champ inclus) ;
    (b) les pages de press/train.jsonl, regroupées par champ `page` (texte = lignes `truth` dans l'ordre du fichier).
    Vérifie : chaque identifiant du filtre est présent, aucun texte vide. Lève ValueError sinon (aucun index partiel)."""
    import gzip
    wanted = json.loads(filter_json.read_text(encoding="utf-8"))["inclus"]
    if len(wanted) != 109 or len(set(wanted)) != 109:
        raise ValueError("la liste de filtrage doit contenir 109 identifiants distincts")
    got: dict[str, str] = {}
    with gzip.open(text_gz, "rt", encoding="utf-8") as f:
        for line in f.read().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["file_id"] in set(wanted):
                if r["file_id"] in got:
                    raise ValueError(f"file_id en double : {r['file_id']}")
                got[r["file_id"]] = r["text"]
    missing = [i for i in wanted if i not in got]
    if missing:
        raise ValueError(f"{len(missing)} identifiants du filtre absents de text.jsonl.gz")
    out = [{"file_id": i, "source": "reader_lm_corpus", "split": "train", "text": got[i]} for i in wanted]
    pages: dict[str, list[str]] = {}
    for line in press_jsonl.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        pages.setdefault(r["page"], []).append(r["truth"])
    for page, lines in pages.items():
        txt = " ".join(t for t in lines if isinstance(t, str) and t.strip())
        if txt.strip():
            out.append({"file_id": "press:" + page, "source": "reader_lm_corpus", "split": "train", "text": txt})
    if not out:
        raise ValueError("aucun texte du lecteur")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", type=Path, help="JSONL : file_id, source, split (train|dev|eval), text (fichier LM vérifié : préférer --lm-corpus)")
    ap.add_argument("--lm-corpus", type=Path, help="dossier corpus_root (avec french-pd-sample/)")
    ap.add_argument("--lm-json", type=Path, help="lm_texts.json (vérifié à la construction)")
    ap.add_argument("--reader-text", type=Path, help="text.jsonl.gz (pour le lecteur : filtré par --reader-filter)")
    ap.add_argument("--reader-filter", type=Path, help="R2_LM_documents_gallica.json (109 identifiants)")
    ap.add_argument("--reader-press", type=Path, help="press/train.jsonl (pages regroupées)")
    ap.add_argument("--pages", type=Path, help="JSONL : id, lines (pages candidates) ; absent avec --index-only")
    ap.add_argument("--out", required=True, type=Path, help="dossier NEUF")
    ap.add_argument("--index-only", action="store_true", help="construit SEULEMENT l'index des textes LM (aucune page, aucun verdict)")
    a = ap.parse_args(argv)
    if a.index_only == bool(a.pages):
        print("refusé : donner --pages, ou --index-only, mais pas les deux ni aucun", file=sys.stderr)
        return 1
    if a.out.exists():
        print(f"refusé : {a.out} existe", file=sys.stderr)
        return 1
    try:
        if bool(a.lm_corpus) != bool(a.lm_json) or bool(a.texts) == bool(a.lm_corpus):
            raise ValueError("donner soit --texts, soit --lm-corpus et --lm-json")
        texts = load_lm_texts(a.lm_corpus, a.lm_json) if a.lm_corpus else read_jsonl(a.texts)
        n_pleias = len({t["file_id"] for t in texts})
        if a.reader_text or a.reader_filter or a.reader_press:
            if not (a.reader_text and a.reader_filter and a.reader_press and a.lm_corpus):
                raise ValueError("lecteur : donner --reader-text, --reader-filter, --reader-press et --lm-corpus ensemble")
            texts = texts + load_reader_entries(a.reader_text, a.reader_filter, a.reader_press)
        files_compared = (lm_files_sha(a.lm_corpus, a.lm_json) if a.lm_corpus
                          else [{"path": str(a.texts), "sha256": hashlib.sha256(a.texts.read_bytes()).hexdigest()}])
        if a.reader_filter and a.reader_press:
            files_compared = files_compared + [
                {"path": "g0_b1_lecteur/R2_LM_documents_gallica.json", "sha256": hashlib.sha256(a.reader_filter.read_bytes()).hexdigest()},
                {"path": "data/prepared/press/train.jsonl", "sha256": hashlib.sha256(a.reader_press.read_bytes()).hexdigest()}]
        idx = build_index(texts)
        pages = [] if a.index_only else read_jsonl(a.pages)
        for r in pages:
            if not isinstance(r.get("id"), str) or not r["id"].strip():
                raise ValueError("page sans id")
            page_text(r.get("lines"))          # validation des lignes avant toute écriture
    except (ValueError, KeyError, TypeError) as exc:
        print(f"refusé : {exc}", file=sys.stderr)
        return 2
    names = {}
    for r in pages:
        safe = re.sub(r"[^\w.-]", "_", r["id"])
        if safe in names:
            print(f"refusé : ids {names[safe]!r} et {r['id']!r} donnent le même nom de fichier", file=sys.stderr)
            return 1
        names[safe] = r["id"]
    a.out.mkdir(parents=True)
    if a.index_only:
        import gzip
        with gzip.open(a.out / "index.jsonl.gz", "wt", encoding="utf-8") as f:
            for g in sorted(idx):
                f.write(json.dumps({"gram": g, "textes": idx[g]}, ensure_ascii=False) + "\n")
        summary = {"mode": "index-only", "textes": len(texts), "index_grammes": len(idx),
                   "par_source": {src: sum(1 for t in texts if t["source"] == src) for src in sorted({t["source"] for t in texts})},
                   "par_split": {sp: sum(1 for t in texts if t["split"] == sp) for sp in sorted({t["split"] for t in texts})},
                   "comparés": files_compared, "index_textes": n_pleias,
                   "index_sha256": hashlib.sha256((a.out / "index.jsonl.gz").read_bytes()).hexdigest()}
        (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"textes": len(texts), "index_grammes": len(idx), "sortie": str(a.out)}))
        return 0
    for r in pages:
        res = check_page(r["id"], r["lines"], idx)
        res["index_textes"] = n_pleias   # textes pleias (322) : le convertisseur n'admet que les pages calculées contre les 322 textes
        res["comparés"] = files_compared                             # empreintes des fichiers RÉELLEMENT comparés (vérifiées par le convertisseur)
        (a.out / (re.sub(r"[^\w.-]", "_", r["id"]) + ".json")).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"textes": len(texts), "index_grammes": len(idx), "sortie": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
