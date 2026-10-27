"""R7(a) — S3 : treillis A1 sur Kaggle T4 en fp16 (docs/READER_R7A_PREREG.txt Q3 ; CLASSE B ; accord pA du 22/10).
Écrit SANS DONNÉES : fumée sur pages synthétiques seulement ; aucune vraie page avant G0 « suivi », relecture et pT.

    ~/outils-seg/dots-hf-env/bin/python scripts/reader_r7a_s3_build.py --build <sortie du constructeur> \\
        --build-manifest-sha256 <SHA> --exclusion <sortie de reader_r7a_exclure.py> --exclusion-manifest-sha256 <SHA> \\
        --inputs <dossier vide, HORS du build et de l'exclusion> [--notebook <chemin .ipynb>]

Ce constructeur (Mac, léger) prépare le jeu Kaggle PRIVÉ et le carnet :
- contrôles : MANIFEST du constructeur et MANIFEST d'exclusion épinglés, chaque sortie rehachée (check_build de
  reader_r7a_exclure) ; le MANIFEST d'exclusion cite le MANIFEST du constructeur ; le dossier d'entrées n'est ni le
  build, ni l'exclusion, ni dedans (pT, mineur du 22/10) ;
- jeu (fichiers à plat, pas de .gz, pas de sous-dossier) : items_retenus.jsonl et MANIFEST d'exclusion TELS QUELS
  (excl_MANIFEST.json), MANIFEST et sources.json du constructeur TELS QUELS (build_MANIFEST.json, build_sources.json),
  les images des pages (page_<sha256>.<ext>), le document du gel et ses 9 fichiers (gel__<chemin, / → __>), le code
  de lecture d'axel (vit.py, ctc.py, crops.py), head_a1.pt et timing.json d'A1, tous ÉPINGLÉS ici (CODE) ; les 9
  fichiers du gel revérifiés sur le Mac ; MANIFEST.json du jeu ;
- carnet (< 250 Ko : au-delà de 300 Ko Kaggle refuse en silence, « Expecting value ») : il ne contient que des SHA
  et du code court ; tout le volumineux est dans le jeu privé, vérifié par SHA.
Lancement : n'attacher QUE le jeu privé (un seul MANIFEST.json sous /kaggle/input), session neuve (sortie vide).
Le carnet impose, AVANT toute lecture : MANIFEST du jeu = INPUTS_SHA ; excl_MANIFEST.json = EXCL_SHA (épinglé ici ;
  refus s'il change) ; items_retenus.jsonl = celui du MANIFEST d'exclusion ; build_MANIFEST.json = celui que cite
  l'exclusion ; build_sources.json = celui du MANIFEST du constructeur ; chaque image = son SHA de sources ; document
  du gel = FREEZE_SHA et ses 9 empreintes ; valeurs gelées LUES dans le gel et vérifiées : τ (σ < 20,83 =
  abstention), λ* = 0,3, NORM (extraite de reader_dots_p3_report.py, sans l'importer) ; recadrages recalculés
  (crop_line, pad 4) : MD5 = crop_md5, σ = sigma, abstention = σ < τ, sinon arrêt. Puis lecture A1 (encodeur dots.mocr
  coupé à 32 blocs, révision épinglée, fp16 sur T4, autocast ; lecture indépendante du lot, axel read_lines) ;
  logp non fini = arrêt (pas de repli silencieux). Sortie /kaggle/working/r7a_s3 : a1.jsonl.gz (stem, hyp, logp
  float16, finitude contrôlée en float16), journal, MANIFEST de sortie (SHA des sorties, INPUTS_SHA, EXCL_SHA, FREEZE_SHA, τ, λ*, contrôles).
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
NB = HERE / "scripts/reader_notebooks/kaggle_r7a_s3_a1.ipynb"
DATASET = "babaorum-r7a-s3-inputs"
LIMITE_CARNET = 250_000
FREEZE = "docs/READER_DOTS_FREEZE_R7A.txt"
FREEZE_SHA = "53e17a2bcb78133315ac4d309d03e39ae71e37d7c4fc8a53ab410bad182e42e7"
CODE = {  # code de lecture et tête d'A1, épinglés (mineur 3 Sonnet)
    "vit.py": (AXEL / "src/axel/reader/vit.py", "e57b43b0b32caaa957046792c84cffc391072613fd1e3396ec449ed342cffda7"),
    "ctc.py": (AXEL / "src/axel/reader/ctc.py", "1213796968a6d207de822b184849d289aa107084398adb3e891f5da05ce9a3d1"),
    "crops.py": (AXEL / "src/axel/image/crops.py", "e7b3a42144f033592f898941b03918d8a35106121b2f1f54c0a1d94fac0a491b"),
    "head_a1.pt": (AXEL / "runs/e17/kaggle/a1/head_a1.pt", "ffb953a16b6beb5f82100cea3f9eb4701d176ff84cb1f267dd888bdc703f3546"),
    "timing.json": (AXEL / "runs/e17/kaggle/a1/timing.json", "9cc562fecfe015aadf87243329233615da9c87f023751745e7fe98525521155f")}

_spec = importlib.util.spec_from_file_location("exclure", HERE / "scripts/reader_r7a_exclure.py")
ex = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex)
need, sha = ex.need, ex.sha

SETUP = r'''
import os, sys, json, gzip, glob, hashlib, ast, time, math, unicodedata
T_START = time.time()
INPUTS_SHA = "{inputs_sha}"
EXCL_SHA = "{excl_sha}"
FREEZE_SHA = "{freeze_sha}"
REPO, REV = "dots-studio/dots.mocr", "e539fbb52280393adc081b289ec597430a0f9031"
L, HEIGHT, MAXW, UP, BS = 32, 56, 2800, 2, 32
OUT, TMP = "/kaggle/working/r7a_s3", "/kaggle/tmp"
PLAFOND_S = 5 * 3600
assert not OUT.startswith("/kaggle/input"), "sortie dans les entrées"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
assert not os.listdir(OUT), "dossier de sortie non vide : relancer dans une session neuve"
class Arret(Exception):
    pass
def need(c, m):
    if not c:
        raise Arret(m)
def h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
mf = glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)
need(len(mf) == 1, f"un seul MANIFEST.json attendu dans /kaggle/input : {{mf}}")
D = os.path.dirname(mf[0])
need(h(mf[0]) == INPUTS_SHA, "MANIFEST du jeu différent de INPUTS_SHA")
MAN = json.load(open(mf[0]))
need(set(MAN) == set(os.listdir(D)) - {{"MANIFEST.json", "dataset-metadata.json"}}, "fichiers du jeu en plus ou en moins")
for n, s in MAN.items():
    need(h(f"{{D}}/{{n}}") == s, f"fichier du jeu modifié : {{n}}")
# chaîne exclusion -> constructeur -> sources, AVANT toute lecture
need(h(f"{{D}}/excl_MANIFEST.json") == EXCL_SHA, "MANIFEST d'exclusion différent du SHA épinglé")
EXM = json.load(open(f"{{D}}/excl_MANIFEST.json"))
need(h(f"{{D}}/items_retenus.jsonl") == EXM["sorties"]["items_retenus.jsonl"], "items_retenus.jsonl n'est pas celui de l'exclusion")
need(h(f"{{D}}/build_MANIFEST.json") == EXM["entrees"]["build_MANIFEST.json"], "MANIFEST du constructeur non cité par l'exclusion")
BM = json.load(open(f"{{D}}/build_MANIFEST.json"))
need(h(f"{{D}}/build_sources.json") == BM["sorties"]["sources.json"], "sources.json n'est pas celui du constructeur")
SOURCES = json.load(open(f"{{D}}/build_sources.json"))
PAGE_FILE = {{}}
for pk, s in SOURCES.items():
    if s.get("refusee"):
        continue
    f = [n for n in MAN if n.startswith(f"page_{{s['image_sha256']}}.")]
    need(len(f) == 1 and MAN[f[0]] == s["image_sha256"], f"image absente ou différente : {{pk}}")
    PAGE_FILE[s["image"]] = f"{{D}}/{{f[0]}}"
# gel : document épinglé, 9 empreintes, valeurs LUES et vérifiées
GEL = f"{{D}}/gel__docs__READER_DOTS_FREEZE_R7A.txt"
need(h(GEL) == FREEZE_SHA, "document du gel différent de FREEZE_SHA")
import re
ent = {{m.group(2): m.group(1) for m in (re.match(r"\s+([0-9a-f]{{64}})\s+(\S+)", l) for l in open(GEL, encoding="utf-8")) if m}}
need(len(ent) == 9, "9 empreintes de gel attendues")
for p, s in ent.items():
    need(h(f"{{D}}/gel__{{p.replace('/', '__')}}") == s, f"gel rompu : {{p}}")
TAU = json.load(open(f"{{D}}/gel__reports__reader_dots_abst_v2.json"))["tau"]
LAMBDA = json.load(open(f"{{D}}/gel__reports__reader_dots_p2_tune.json"))["choix"]["lambda_star"]
need(TAU == 20.827896720535783 and LAMBDA == 0.3, f"valeurs gelées inattendues : tau {{TAU}}, lambda {{LAMBDA}}")
src = open(f"{{D}}/gel__scripts__reader_dots_p3_report.py", encoding="utf-8").read()
keep = [n for n in ast.parse(src).body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_NORM" for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "norm")]
need(len(keep) == 2, "NORM introuvable dans reader_dots_p3_report.py")
NS = {{"unicodedata": unicodedata}}
exec(compile(ast.Module(body=keep, type_ignores=[]), "p3_report:NORM", "exec"), NS)
NORM = NS["norm"]
need(NORM("l’a  «b» —c") == "l'a \"b\" -c", "NORM ne se comporte pas comme le gel")
sys.path.insert(0, D)
ITEMS = [json.loads(l) for l in open(f"{{D}}/items_retenus.jsonl", encoding="utf-8")]
need(len({{i["stem"] for i in ITEMS}}) == len(ITEMS), "tiges en double")
print(len(ITEMS), "items ;", len(PAGE_FILE), "pages ; tau", TAU, "; lambda*", LAMBDA, "; contrôles d'entrée OK")
'''

INSTALL = """%%bash
set -eo pipefail
pip install -q safetensors huggingface_hub 2>&1 | tail -1
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"""

WEIGHTS = r'''
import torch, shutil
from safetensors import safe_open
from huggingface_hub import hf_hub_download
need(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7, "T4 attendu (fp16)")
DT = torch.float16
from vit import NaViT, SPECS, MEAN, STD, prep, rename_map
from ctc import Head, read_lines, white_value
from crops import crop_line
need((REPO, REV) == (SPECS["dots_mocr"]["repo"], SPECS["dots_mocr"]["rev"]), "révision de dots.mocr")
idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json", revision=REV, cache_dir=f"{TMP}/hf")))
s = SPECS["dots_mocr"]
ren = rename_map(s["prefix"], L)
sd = {}
for shard in sorted({idx["weight_map"][k] for k in ren}):
    p = hf_hub_download(REPO, shard, revision=REV, cache_dir=f"{TMP}/hf")
    with safe_open(p, framework="pt") as fh:
        for k in fh.keys():
            if k in ren:
                sd[ren[k]] = fh.get_tensor(k)
    os.remove(os.path.realpath(p))
shutil.rmtree(f"{TMP}/hf", ignore_errors=True)
enc = NaViT(s["dim"], s["heads"], s["ffn"], L, s["rope"])
missing, unexpected = enc.load_state_dict(sd, strict=False)
need(not unexpected and [k for k in missing if k != "inv_freq"] == [], "poids de l'encodeur")
enc = enc.to("cuda", DT).eval(); del sd
TIMING = json.load(open(f"{D}/timing.json"))
need(TIMING.get("L") == L, "timing.json n'est pas celui d'A1 (L = 32)")
CHARS = TIMING["chars"]
head = Head(len(CHARS) + 1, up=UP)
head.load_state_dict(torch.load(f"{D}/head_a1.pt", map_location="cpu"))
head = head.to("cuda").eval()
WHITE = white_value(MEAN, STD)
print("encodeur", L, "blocs", DT, "; tête", len(CHARS), "caractères")
'''

CROPS = r'''
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
need(time.time() - T_START < PLAFOND_S, "plafond de temps atteint avant les recadrages")  # mineur 3 pT
CROP, ecarts = {}, []
by_img = {}
for it in ITEMS:
    by_img.setdefault(it["image"], []).append(it)
for img, group in sorted(by_img.items()):
    need(time.time() - T_START < PLAFOND_S, "plafond de temps atteint pendant les recadrages")
    need(img in PAGE_FILE, f"page non gardée ou inconnue : {img}")
    with Image.open(PAGE_FILE[img]) as im:
        page = im.convert("L")
    for it in group:
        c = crop_line(page, tuple(it["box"]), it["polygon"], pad=4).image
        sig = float(np.asarray(c, dtype=np.float64).std())
        if (hashlib.md5(c.tobytes()).hexdigest() != it["crop_md5"] or abs(sig - it["sigma"]) > 1e-6
                or (sig < TAU) != it["abstention"]):
            ecarts.append(it["stem"])
        CROP[it["stem"]] = c
need(not ecarts, f"recadrages différents du constructeur : {len(ecarts)} ({ecarts[:3]})")
print(len(CROP), "recadrages identiques au constructeur (MD5, sigma, abstention)")
'''

READ = r'''
amp = lambda: torch.autocast("cuda", dtype=DT)
order = sorted(ITEMS, key=lambda it: CROP[it["stem"]].width * HEIGHT / max(1, CROP[it["stem"]].height))
# mineur 4 pT : chaque lot est écrit aussitôt (mémoire bornée à un lot) ; contrôle d'unicité et de compte à la fin
n_ecrits, vus, non_finis = 0, set(), []
g = gzip.open(f"{OUT}/a1.jsonl.gz", "wt", encoding="utf-8")
try:  # en cas d'arrêt, le fichier est fermé ; sans MANIFEST de sortie, S4 le refuse
  with torch.no_grad():
    for i in range(0, len(order), BS):
        need(time.time() - T_START < PLAFOND_S, "plafond de temps atteint")
        b = order[i:i + BS]
        for it, (hyp, lp) in zip(b, read_lines(enc, head, [prep(CROP[it["stem"]], HEIGHT, MAXW) for it in b], L, CHARS, WHITE, "cuda", amp)):
            lp16 = lp.float().cpu().half()  # mineur 1 Sonnet : finitude contrôlée sur ce qui est écrit (float16)
            if not torch.isfinite(lp16).all():
                non_finis.append(it["stem"])
            # hyp = glouton de read_lines AVANT quantification ; le témoin « A1 seul » de R7(a) est le glouton recalculé
            # sur ce logp float16 écrit (PREREG, mineur 5 pT) ; les deux sont comparés en S5
            g.write(json.dumps({"stem": it["stem"], "part": it["part"], "hyp": hyp, "logp": lp16.numpy().tolist()},
                               ensure_ascii=False) + "\n")
            n_ecrits += 1; vus.add(it["stem"])
finally:
    g.close()
need(not non_finis, f"logp non fini en fp16 : {len(non_finis)} lignes ({non_finis[:3]}) ; arrêt, retour à pA")
need(n_ecrits == len(ITEMS) == len(vus), "lignes manquantes ou en double")
import PIL, numpy, safetensors, huggingface_hub, platform
journal = {"items": n_ecrits, "par_part": {p: sum(it["part"] == p for it in ITEMS) for p in sorted({it["part"] for it in ITEMS})},
           "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
                        "PIL": PIL.__version__, "safetensors": safetensors.__version__,
                        "huggingface_hub": huggingface_hub.__version__},  # mineur 2 pT
           "octets_a1_gz": os.path.getsize(f"{OUT}/a1.jsonl.gz"),
           "abstentions_prevues": sum(it["abstention"] for it in ITEMS), "dtype": str(DT), "gpu": torch.cuda.get_device_name(0),
           "secondes": round(time.time() - T_START), "tau": TAU, "lambda_star": LAMBDA, "norm_verifiee": True}
json.dump(journal, open(f"{OUT}/journal.json", "w"), indent=1, ensure_ascii=False)
man = {"entrees": {"INPUTS_SHA": INPUTS_SHA, "EXCL_SHA": EXCL_SHA, "FREEZE_SHA": FREEZE_SHA, "REV": REV},
       "sorties": {n: h(f"{OUT}/{n}") for n in sorted(os.listdir(OUT))}}
json.dump(man, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print(json.dumps(journal, ensure_ascii=False))
'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def notebook(inputs_sha: str, excl_sha: str) -> dict:
    head = ("# babaorum — R7(a) S3 : treillis A1 (T4, fp16)\n\ndocs/READER_R7A_PREREG.txt (Q3) ; contrôles d'entrée AVANT "
            "toute lecture ; « GPU T4 x2 », Internet ON ; jeu privé " + DATASET + " attaché.")
    return {"cells": [md(head), code(SETUP.format(inputs_sha=inputs_sha, excl_sha=excl_sha, freeze_sha=FREEZE_SHA)),
                      code(INSTALL), code(WEIGHTS), code(CROPS), code(READ)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def dedans(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--build-manifest-sha256", required=True)
    ap.add_argument("--exclusion", required=True)
    ap.add_argument("--exclusion-manifest-sha256", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--notebook", default=str(NB))
    a = ap.parse_args()
    build, excl, out = Path(a.build), Path(a.exclusion), Path(a.inputs)
    need(not dedans(build, excl) and not dedans(excl, build), "build et exclusion doivent être des dossiers distincts")
    for d in (build, excl):
        need(not dedans(out, d) and not dedans(d, out), f"le dossier d'entrées ne peut être ni dans {d} ni le contenir")
        need(not dedans(Path(a.notebook), d) and not dedans(Path(a.notebook), out), f"carnet à écrire hors de {d} et des entrées")
    need(not out.exists() or not any(out.iterdir()), "dossier d'entrées non vide")
    bman = ex.check_build(build, a.build_manifest_sha256)
    eman = ex.check_build(excl, a.exclusion_manifest_sha256)
    need(eman["entrees"].get("build_MANIFEST.json") == a.build_manifest_sha256, "l'exclusion ne cite pas ce constructeur")
    need(sha(excl / "items_retenus.jsonl") == eman["sorties"]["items_retenus.jsonl"], "items_retenus.jsonl modifié")
    need(sha(HERE / FREEZE) == FREEZE_SHA, "document du gel modifié")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(excl / "items_retenus.jsonl", out / "items_retenus.jsonl")
    shutil.copy(excl / "MANIFEST.json", out / "excl_MANIFEST.json")
    shutil.copy(build / "MANIFEST.json", out / "build_MANIFEST.json")
    shutil.copy(build / "sources.json", out / "build_sources.json")
    sources = json.loads((build / "sources.json").read_text())
    for pk, s in sorted(sources.items()):
        if s.get("refusee"):
            continue
        src = Path(s["image"])
        need(sha(src) == s["image_sha256"], f"image modifiée : {pk}")
        dst = out / f"page_{s['image_sha256']}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy(src, dst)
    gel = (HERE / FREEZE).read_text()
    ent = {m.group(2): m.group(1) for m in (re.match(r"\s+([0-9a-f]{64})\s+(\S+)", x) for x in gel.splitlines()) if m}
    need(len(ent) == 9 and all(sha(HERE / p) == h for p, h in ent.items()), "gel rompu sur le Mac")  # mineur 2
    for p in [FREEZE, *ent]:
        shutil.copy(HERE / p, out / f"gel__{p.replace('/', '__')}")
    for n, (p, h) in CODE.items():
        need(sha(p) == h, f"code ou tête d'A1 modifiés : {p}")
        shutil.copy(p, out / n)
    need(not any(p.suffix == ".gz" or p.is_dir() for p in out.iterdir()), "jeu Kaggle : ni .gz ni sous-dossier")
    manifest = {p.name: sha(p) for p in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    inputs_sha = sha(out / "MANIFEST.json")
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"mabaku/{DATASET}",
                                                           "licenses": [{"name": "other"}]}))
    raw = json.dumps(notebook(inputs_sha, a.exclusion_manifest_sha256), indent=1, ensure_ascii=False).encode("utf-8")
    need(len(raw) < LIMITE_CARNET, f"carnet de {len(raw)} octets : au-delà de {LIMITE_CARNET} (Kaggle refuse dès 300 Ko)")
    Path(a.notebook).write_bytes(raw)
    fiche = {"carnet": Path(a.notebook).name, "carnet_sha256": hashlib.sha256(raw).hexdigest(), "carnet_octets": len(raw),
             "inputs_manifest_sha256": inputs_sha, "exclusion_manifest_sha256": a.exclusion_manifest_sha256,
             "build_manifest_sha256": a.build_manifest_sha256, "gel_sha256": FREEZE_SHA,
             "constructeur_sha256": sha(Path(__file__)), "fichiers": len(manifest),
             "octets_jeu": sum((out / n).stat().st_size for n in manifest)}  # mineur 2 Sonnet : fiche de lancement
    Path(a.notebook).with_suffix(".lancement.json").write_text(json.dumps(fiche, indent=1))
    print(json.dumps(fiche, indent=1))


if __name__ == "__main__":
    main()
