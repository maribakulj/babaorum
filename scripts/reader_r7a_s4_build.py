"""R7(a) — S4 : lecture dots.mocr (dots libre et dispositif guidé gelé) sur Kaggle T4 ×2 (docs/READER_R7A_PREREG.txt ;
docs/READER_R7A_BUILD_DESIGN.txt S4 ; CLASSE B ; accord pA du 22/10). Écrit SANS DONNÉES ; aucune vraie page avant
G0 « suivi », relecture Sonnet et pT.

    ~/outils-seg/dots-hf-env/bin/python scripts/reader_r7a_s4_build.py --s3-inputs <jeu S3, tel que téléversé> \\
        --s3-inputs-manifest-sha256 <SHA> --s3-output-manifest-sha256 <SHA du MANIFEST.json de la sortie Kaggle S3> \\
        --inputs <dossier vide : jeu de code S4> [--notebook <chemin .ipynb>]

Le carnet P3 est RÉUTILISÉ (scripts/reader_dots_p3_build.py, audité) : cellules CHRONO (plafond dur 6 h), MODEL (P1,
telle quelle : fp32 sur T4 ×2, correctifs D-182, patch_beam_expansion) et RUN (guide CTCGuide, masque de vocabulaire,
top-K 10, PEN = FLOOR, max_new = min(512, T + 16), reprise ligne à ligne, fsync), avec ces SEULS remplacements,
vérifiés (motif présent une fois) :
  - treillis : lus dans a1.jsonl.gz de la SORTIE S3 (logp float16, MANIFEST de sortie épinglé), plus de a1.jsonl ;
  - panneau : les items retenus (lignes presse/livre et zones vides), couverture exacte des treillis ;
  - configurations : dots libre (glouton) et guide_l0.3 avec λ LU dans le gel (= 0,3 exigé), sur TOUS les items ;
    aucune autre (guide_l0.0 retiré : PREREG, « aucune autre configuration ») ;
  - noms de sortie r7a_s4 ; journal de passage complété des versions (mineur 2 pT).
SETUP et IMAGES sont propres à R7(a) : trois entrées épinglées (jeu S3, sortie S3, jeu de code S4), chaîne exclusion
→ constructeur → sources → images revérifiée comme en S3, MANIFEST d'exclusion = EXCL_SHA AVANT toute lecture, gel
(FREEZE_SHA, 9 empreintes) ; τ, λ* et NORM LUS dans le gel ; le code du guide (ctc_prefix, dots_guided, dots_beam)
est pris dans les fichiers DU GEL eux-mêmes ; recadrages recalculés (MD5, σ, abstention = σ < τ), toile P1 (échelle 1).
Aucune reprise (B1 pT) : une sortie S4 préexistante est refusée. Toute fin autre que eos ou plafond (coupure par le
temps) marque le passage interrompu et est listée (B2 pT) : aucun verdict (PREREG). T4 ×2 et fp32 imposés (M1 pT).
L'abstention (σ < τ) n'est pas appliquée ici : elle l'est en S5 sur la sortie guidée (dispositif gelé), comme en P3.
Le carnet ne contient que des SHA et du code (< 250 Ko, contrôlé) ; vlm.py et lattice.py d'axel, épinglés, sont dans
le petit jeu de code S4. Lancement : attacher le jeu S3, la sortie du noyau S3 et le jeu S4, rien d'autre ; T4 ×2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
NB = HERE / "scripts/reader_notebooks/kaggle_r7a_s4_dots.ipynb"
DATASET = "babaorum-r7a-s4-code"
LIMITE_CARNET = 250_000
FREEZE_SHA = "53e17a2bcb78133315ac4d309d03e39ae71e37d7c4fc8a53ab410bad182e42e7"
CODE = {"vlm.py": (AXEL / "src/axel/image/vlm.py", "1af22eca8a213d0976113fbe3aa6d12660943b4caa0707921fcf923a181dbde5"),
        "lattice.py": (AXEL / "src/axel/image/lattice.py", "313d6e7d62a6a9da57c4dc41827ff81369fdd45b68875f0afb1cff798f0e37d9")}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gel_ok() -> None:
    """Mineur 1 Sonnet : les constructeurs P1/P3 réutilisés (et les 7 autres fichiers du gel) sont ceux du gel."""
    raw = (HERE / "docs/READER_DOTS_FREEZE_R7A.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FREEZE_SHA, "document du gel modifié"
    ent = {m.group(2): m.group(1) for m in (re.match(r"\s+([0-9a-f]{64})\s+(\S+)", x) for x in raw.decode().splitlines()) if m}
    assert len(ent) == 9 and all(hashlib.sha256((HERE / p).read_bytes()).hexdigest() == h for p, h in ent.items()), "gel rompu"


_gel_ok()  # avant tout import des constructeurs P1/P3 ; assert ici : ex.need n'est pas encore chargé
p3 = _load("p3", "scripts/reader_dots_p3_build.py")
p1, sub = p3.p1, p3.sub
ex = _load("exclure", "scripts/reader_r7a_exclure.py")
need, sha = ex.need, ex.sha

CHRONO = sub(p3.CHRONO, "/kaggle/working/dots_p3_arret_dur.json", "/kaggle/working/r7a_s4_arret_dur.json")

INSTALL = """%%bash
set -eo pipefail
pip install -q 'transformers==4.51.3' 'accelerate>=0.34' 2>&1 | tail -1"""

SETUP = r'''
import sys, gzip, glob, hashlib, shutil, subprocess, zipfile, math, types, importlib.machinery, ast, re, unicodedata
S3_INPUTS_SHA = "{s3_inputs}"
S3_OUT_SHA = "{s3_out}"
S4_CODE_SHA = "{s4_code}"
EXCL_SHA = "{excl}"
FREEZE_SHA = "{freeze}"
PHASE = "EVAL"
D, TMP, OUT = "/kaggle/input", "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
class Arret(Exception):
    pass
def need(c, m):
    if not c:
        raise Arret(m)
# B1 pT (23/10) : AUCUNE reprise ; une sortie S4 déjà présente (d'un autre lancement) est refusée : session neuve exigée
need(not any(n.startswith("r7a_s4") for n in os.listdir(OUT)), "sortie S4 préexistante : session neuve exigée (aucune reprise)")
def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
MANS = {{}}
for p in glob.glob(f"{{D}}/**/MANIFEST.json", recursive=True):
    MANS.setdefault(h(p), []).append(os.path.dirname(p))
need(all(len(MANS.get(s, [])) == 1 for s in (S3_INPUTS_SHA, S3_OUT_SHA, S4_CODE_SHA)),
     "les trois entrées épinglées (jeu S3, sortie S3, code S4) doivent être attachées, une fois chacune")
need(len(MANS) == 3, "aucune autre entrée ne doit être attachée")
IN, S3O, C4 = (MANS[s][0] for s in (S3_INPUTS_SHA, S3_OUT_SHA, S4_CODE_SHA))
for d_ in (IN, C4):  # jeux à plat : MANIFEST {{nom: sha}}
    m_ = json.load(open(f"{{d_}}/MANIFEST.json"))
    need(set(m_) == set(os.listdir(d_)) - {{"MANIFEST.json", "dataset-metadata.json"}}, f"fichiers en plus ou en moins : {{d_}}")
    for n, s in m_.items():
        need(h(f"{{d_}}/{{n}}") == s, f"fichier modifié : {{d_}}/{{n}}")
MAN = json.load(open(f"{{IN}}/MANIFEST.json"))
S3M = json.load(open(f"{{S3O}}/MANIFEST.json"))  # sortie S3 : {{entrees, sorties}}
need(S3M["entrees"] == {{"INPUTS_SHA": S3_INPUTS_SHA, "EXCL_SHA": EXCL_SHA, "FREEZE_SHA": FREEZE_SHA,
                         "REV": "e539fbb52280393adc081b289ec597430a0f9031"}}, "la sortie S3 ne vient pas de ce jeu S3")
for n, s in S3M["sorties"].items():
    need(h(f"{{S3O}}/{{n}}") == s, f"sortie S3 modifiée : {{n}}")
# chaîne exclusion -> constructeur -> sources -> images, AVANT toute lecture (comme S3)
need(h(f"{{IN}}/excl_MANIFEST.json") == EXCL_SHA, "MANIFEST d'exclusion différent du SHA épinglé")
EXM = json.load(open(f"{{IN}}/excl_MANIFEST.json"))
need(h(f"{{IN}}/items_retenus.jsonl") == EXM["sorties"]["items_retenus.jsonl"], "items_retenus.jsonl n'est pas celui de l'exclusion")
need(h(f"{{IN}}/build_MANIFEST.json") == EXM["entrees"]["build_MANIFEST.json"], "MANIFEST du constructeur non cité par l'exclusion")
BM = json.load(open(f"{{IN}}/build_MANIFEST.json"))
need(h(f"{{IN}}/build_sources.json") == BM["sorties"]["sources.json"], "sources.json n'est pas celui du constructeur")
PAGE_FILE = {{}}
for pk, s in json.load(open(f"{{IN}}/build_sources.json")).items():
    if s.get("refusee"):
        continue
    f = [n for n in MAN if n.startswith(f"page_{{s['image_sha256']}}.")]
    need(len(f) == 1 and MAN[f[0]] == s["image_sha256"], f"image absente ou différente : {{pk}}")
    PAGE_FILE[s["image"]] = f"{{IN}}/{{f[0]}}"
# gel : document épinglé, 9 empreintes, valeurs LUES et vérifiées ; code du guide pris dans le gel
GEL = f"{{IN}}/gel__docs__READER_DOTS_FREEZE_R7A.txt"
need(h(GEL) == FREEZE_SHA, "document du gel différent de FREEZE_SHA")
ent = {{m.group(2): m.group(1) for m in (re.match(r"\s+([0-9a-f]{{64}})\s+(\S+)", l) for l in open(GEL, encoding="utf-8")) if m}}
need(len(ent) == 9, "9 empreintes de gel attendues")
for p, s in ent.items():
    need(h(f"{{IN}}/gel__{{p.replace('/', '__')}}") == s, f"gel rompu : {{p}}")
TAU = json.load(open(f"{{IN}}/gel__reports__reader_dots_abst_v2.json"))["tau"]
LAMBDA = json.load(open(f"{{IN}}/gel__reports__reader_dots_p2_tune.json"))["choix"]["lambda_star"]
need(TAU == 20.827896720535783 and LAMBDA == 0.3, f"valeurs gelées inattendues : tau {{TAU}}, lambda {{LAMBDA}}")
src = open(f"{{IN}}/gel__scripts__reader_dots_p3_report.py", encoding="utf-8").read()
keep = [n for n in ast.parse(src).body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_NORM" for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "norm")]
need(len(keep) == 2, "NORM introuvable dans reader_dots_p3_report.py")
NS = {{"unicodedata": unicodedata}}
exec(compile(ast.Module(body=keep, type_ignores=[]), "p3_report:NORM", "exec"), NS)
NORM = NS["norm"]
need(NORM("l’a  «b» —c") == "l'a \"b\" -c", "NORM ne se comporte pas comme le gel")
CODE = f"{{TMP}}/code"; os.makedirs(CODE, exist_ok=True)
for f in ("ctc_prefix.py", "dots_guided.py", "dots_beam.py"):
    shutil.copy(f"{{IN}}/gel__src__babaorum__reader__{{f}}", f"{{CODE}}/{{f}}")
shutil.copy(f"{{IN}}/crops.py", CODE)
for f in ("vlm.py", "lattice.py"):
    shutil.copy(f"{{C4}}/{{f}}", CODE)
pkg = f"{{CODE}}/babaorum/reader"; os.makedirs(pkg, exist_ok=True)
open(f"{{CODE}}/babaorum/__init__.py", "w").close(); open(f"{{pkg}}/__init__.py", "w").close()
shutil.copy(f"{{CODE}}/ctc_prefix.py", pkg); shutil.copy(f"{{CODE}}/dots_guided.py", pkg)
sys.path.insert(0, CODE)
MANIFEST_SHA = {{"s3_inputs": S3_INPUTS_SHA, "s3_sortie": S3_OUT_SHA, "s4_code": S4_CODE_SHA, "exclusion": EXCL_SHA, "gel": FREEZE_SHA}}
ITEMS = [json.loads(l) for l in open(f"{{IN}}/items_retenus.jsonl", encoding="utf-8")]
need(len({{i["stem"] for i in ITEMS}}) == len(ITEMS), "tiges en double")
print(len(ITEMS), "items ;", len(PAGE_FILE), "pages ; tau", TAU, "; lambda*", LAMBDA, "; entrées conformes")
'''

IMAGES = r'''
import torch
# M1 pT : matériel et précision GELÉS (T4 ×2, fp32) imposés AVANT le chargement du modèle
need(torch.cuda.device_count() >= 2 and all(torch.cuda.get_device_capability(i)[0] == 7 and "T4" in torch.cuda.get_device_name(i)
                                              for i in range(torch.cuda.device_count())), "accélérateur « GPU T4 x2 » exigé")
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from crops import crop_line
from vlm import line_canvas_size
ORDRE = {"presse": 0, "vide": 1, "livre": 2}  # presse (critère primaire) d'abord, puis les zones, puis les livres
need(all(it["part"] in ORDRE for it in ITEMS), "part inconnue")
items = sorted(ITEMS, key=lambda it: (ORDRE[it["part"]], it["stem"]))
CANVAS, ecarts = {}, []
by_img = {}
for it in items:
    by_img.setdefault(it["image"], []).append(it)
for img, group in sorted(by_img.items()):
    need(img in PAGE_FILE, f"page non gardée ou inconnue : {img}")
    with Image.open(PAGE_FILE[img]) as raw_img:
        page = raw_img.convert("L")
    for it in group:
        crop = crop_line(page, tuple(it["box"]), it["polygon"], pad=4).image
        sig = float(np.asarray(crop, dtype=np.float64).std())
        if (hashlib.md5(crop.tobytes()).hexdigest() != it["crop_md5"] or abs(sig - it["sigma"]) > 1e-6
                or (sig < TAU) != it["abstention"]):
            ecarts.append(it["stem"])
        cv = line_canvas_size(crop.width, crop.height, 0, 0)
        need(cv.scale == 1.0, f"toile à l'échelle {cv.scale} : {it['stem']}")
        canvas = Image.new("RGB", (cv.width, cv.height), (255, 255, 255))
        canvas.paste(crop.convert("RGB"), (cv.left, cv.top))
        CANVAS[it["stem"]] = canvas
need(not ecarts, f"recadrages différents du constructeur : {len(ecarts)} ({ecarts[:3]})")
print(len(CANVAS), "toiles ; recadrages identiques au constructeur (MD5, sigma, abstention)")
'''

RUN = sub(p3.RUN, "from ctc_prefix import CTCPrefixScorer",
          "need(dtype == torch.float32 and device_map == {'model': 0, 'lm_head': 0, 'vision_tower': 1}, 'fp32 sur T4 x2 exigé (M1 pT)')\n"
          "from ctc_prefix import CTCPrefixScorer")
RUN = sub(RUN, """LAT = {}
for l in open(f'{IN}/a1.jsonl'):
    r = json.loads(l)
    LAT[r['stem']] = np.asarray(r['logp'], dtype=np.float64)""",
          """LAT = {}
for l in gzip.open(f'{S3O}/a1.jsonl.gz', 'rt', encoding='utf-8'):  # treillis T4 fp16 de S3 (sortie épinglée)
    r = json.loads(l)
    need(r['stem'] not in LAT, f"treillis en double : {r['stem']}")
    LAT[r['stem']] = np.asarray(r['logp'], dtype=np.float64)""")
RUN = sub(RUN, """assert len(stems) == len(set(stems)) == 3005, (len(stems), len(set(stems)))
assert sum(it['part'] == 'eval' for it in items) == 2405 and set(stems) == set(LAT), 'panneau ou treillis'""",
          """need(len(stems) == len(set(stems)) == len(ITEMS) and set(stems) == set(LAT), 'panneau ou treillis (couverture exacte)')""")
RUN = sub(RUN, """CONFIGS_EVAL = [('libre', None, 1), ('guide_l0.0', 0.0, 4), ('guide_l0.3', 0.3, 4)]
CONFIGS_NEG = [('libre', None, 1), ('guide_l0.3', 0.3, 4)]""",
          """# R7(a) : dots libre et dispositif gelé SEULEMENT (PREREG) ; λ LU dans le gel (cellule SETUP), = 0,3
need(LAMBDA == 0.3, 'lambda du gel')
CONFIGS_EVAL = [('libre', None, 1), ('guide_l0.3', LAMBDA, 4)]
CONFIGS_NEG = CONFIGS_EVAL
FINS_TEMPS = []  # B2 pT : générations coupées par max_time""")
RUN = sub(RUN, """                  ('temps' if time.time() >= GEN_END - 1 or time.time() - t0 >= 599 else 'autre'))""",
          """                  ('temps' if time.time() >= GEN_END - 1 or time.time() - t0 >= 599 else 'autre'))
            if fin not in ('eos', 'plafond'):  # B2 pT : coupure (temps) ou fin anormale = passage interrompu, aucun verdict
                INTERROMPU = True
                FINS_TEMPS.append([it['stem'], name, fin])""")
RUN = sub(RUN, """        for name, lam, nb in (CONFIGS_EVAL if it['part'] == 'eval' else CONFIGS_NEG):""",
          """        for name, lam, nb in CONFIGS_EVAL:  # mêmes configurations pour lignes et zones""")
RUN = sub(RUN, "dest = f'{OUT}/dots_p1_{PHASE.lower()}.jsonl'", "dest = f'{OUT}/r7a_s4.jsonl'")
RUN = sub(RUN, "done = set()\n", "done = set()\nneed(not os.path.exists(dest), 'sortie S4 préexistante : aucune reprise (B1 pT)')\n")
RUN = sub(RUN, "open(f'{OUT}/dots_p1_{PHASE.lower()}_run.json', 'w'), indent=1)",
          "open(f'{OUT}/r7a_s4_run.json', 'w'), indent=1)")
RUN = sub(RUN, """           'python': sys.version.split()[0], 'lignes_prevues': len(items), 'configs': CONFIGS,""",
          """           'python': sys.version.split()[0], 'numpy': np.__version__, 'PIL': Image.__version__,
           'accelerate': __import__('accelerate').__version__, 'gpu': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           'fins_anormales': FINS_TEMPS, 'lignes_prevues': len(items), 'configs': CONFIGS,""")

FIN = r'''
man = {"entrees": MANIFEST_SHA, "sorties": {n: h(f"{OUT}/{n}") for n in sorted(os.listdir(OUT))
                                           if n.startswith("r7a_s4") and n != "r7a_s4_MANIFEST.json" and os.path.isfile(f"{OUT}/{n}")}}
json.dump(man, open(f"{OUT}/r7a_s4_MANIFEST.json", "w"), indent=1)
print(json.dumps(man, indent=1))
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def notebook(s3_inputs, s3_out, s4_code, excl) -> dict:
    head = ("# babaorum — R7(a) S4 : dots libre et dispositif guidé gelé (T4 ×2, fp32)\n\nCarnet P3 réutilisé ; "
            "trois entrées épinglées (jeu S3, sortie S3, " + DATASET + ") ; « GPU T4 x2 », Internet ON.")
    setup = SETUP.format(s3_inputs=s3_inputs, s3_out=s3_out, s4_code=s4_code, excl=excl, freeze=FREEZE_SHA)
    return {"cells": [md(head), code(CHRONO), code(INSTALL), code(setup), code(IMAGES), code(p1.MODEL), code(RUN), code(FIN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def dedans(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-inputs", required=True)
    ap.add_argument("--s3-inputs-manifest-sha256", required=True)
    ap.add_argument("--s3-output-manifest-sha256", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--notebook", default=str(NB))
    a = ap.parse_args()
    s3, out, nbp = Path(a.s3_inputs), Path(a.inputs), Path(a.notebook)
    need(not dedans(out, s3) and not dedans(s3, out) and not dedans(nbp, s3) and not dedans(nbp, out),
         "entrées S4 et carnet hors du jeu S3 (et l'un hors de l'autre)")
    need(not out.exists() or not any(out.iterdir()), "dossier d'entrées non vide")
    mp = s3 / "MANIFEST.json"
    need(sha(mp) == a.s3_inputs_manifest_sha256, "MANIFEST du jeu S3 différent du SHA épinglé")
    man = json.loads(mp.read_text())
    for n, h in man.items():
        need(sha(s3 / n) == h, f"jeu S3 modifié : {n}")
    excl = sha(s3 / "excl_MANIFEST.json")
    need(sha(s3 / "gel__docs__READER_DOTS_FREEZE_R7A.txt") == FREEZE_SHA, "gel du jeu S3")
    out.mkdir(parents=True, exist_ok=True)
    for n, (p, h) in CODE.items():
        need(sha(p) == h, f"code d'axel modifié : {p}")
        shutil.copy(p, out / n)
    manifest = {p.name: sha(p) for p in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    s4_code = sha(out / "MANIFEST.json")
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"mabaku/{DATASET}",
                                                           "licenses": [{"name": "other"}]}))
    raw = json.dumps(notebook(a.s3_inputs_manifest_sha256, a.s3_output_manifest_sha256, s4_code, excl),
                     indent=1, ensure_ascii=False).encode("utf-8")
    need(len(raw) < LIMITE_CARNET, f"carnet de {len(raw)} octets : au-delà de {LIMITE_CARNET}")
    nbp.write_bytes(raw)
    fiche = {"carnet": nbp.name, "carnet_sha256": hashlib.sha256(raw).hexdigest(), "carnet_octets": len(raw),
             "s3_inputs_manifest_sha256": a.s3_inputs_manifest_sha256,
             "s3_output_manifest_sha256": a.s3_output_manifest_sha256, "s4_code_manifest_sha256": s4_code,
             "exclusion_manifest_sha256": excl, "gel_sha256": FREEZE_SHA,
             "constructeur_sha256": sha(Path(__file__))}  # mineur 2 Sonnet : fiche de lancement, à verser au dépôt
    nbp.with_suffix(".lancement.json").write_text(json.dumps(fiche, indent=1))
    print(json.dumps(fiche, indent=1))


if __name__ == "__main__":
    main()
