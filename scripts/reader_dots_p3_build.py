"""P3 de docs/READER_DOTS_CTC_PLAN.txt — construit les entrées et le carnet Kaggle de l'ÉVAL (SECOND REGARD) et des
négatifs Q2 (CLASSE B ; rien n'est lancé). Pré-enregistrement : « PRÉ-ENREGISTREMENT P3 » en fin du plan.

    ~/axel/.venv/bin/python scripts/reader_dots_p3_build.py --inputs <dossier vide du scratchpad>

Réutilise TELLES QUELLES les cellules SETUP et MODEL du constructeur P1 (scripts/reader_dots_p1_build.py, audité
par pT) ; IMAGES et RUN sont dérivées de celles de P1 par des remplacements EXPLICITES et vérifiés (assert sur
chaque motif), listés ci-dessous :
  - lignes : part ∈ {eval, neg} au lieu de tune ;
  - négatifs : le recadrage d'origine est contrôlé (crop_md5), puis perturbé par src/babaorum/reader/perturb.py
    avec la même clé que le treillis A1 (stem + « #r-gate » pour scramble et blank, + « #r-dens » pour strips),
    puis l'image perturbée est contrôlée à son tour (pert_md5, calculé ici sur le Mac) : dots.mocr voit exactement
    l'image qu'A1 a lue ;
  - configurations : éval = libre, guide_l0.0, guide_l0.3 (λ* gelé) ; négatifs = libre, guide_l0.3 ;
  - plafond dur : aucune nouvelle ligne commencée après 5 h 45 depuis le début de la cellule IMAGES ; l'arrêt est
    écrit dans le journal de passage (interrompu = vrai) ; ordre : toute l'éval, puis les négatifs ;
  - contrôles de panneau : 2 405 + 600 tiges distinctes, couverture exacte des 3 005 treillis.
Entrées (à plat, sans .gz : Kaggle décompresse) : items.jsonl (2 405 éval tirées d'axel e8_items, tiges = extrait
R6 B0 dont l'empreinte est contrôlée ; 600 négatifs = 200 lignes de reports/reader_items_photo1983_pert200.jsonl
× 3 perturbations, tige unique « <stem>#<mode> »), a1.jsonl (treillis A1 : éval depuis axel dev_pred_a1, négatifs
depuis reports/reader_a1_photo1983_<mode>.jsonl.gz, lus sur le Mac), timing.json, modules (ceux de P1 + perturb.py),
MANIFEST.json (empreinte épinglée dans le carnet).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
NB = HERE / "scripts/reader_notebooks"
DATASET = "babaorum-dots-p3-inputs"
B0_SHA = "b3e33a9b0e12fca995e5daebc9d26be4678944fd82657060da36784bc3bd8451"
MODES = {"scramble": "#r-gate", "blank": "#r-gate", "strips": "#r-dens"}

_spec = importlib.util.spec_from_file_location("p1", HERE / "scripts/reader_dots_p1_build.py")
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)


def sub(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, ("motif absent ou multiple", old[:80])
    return text.replace(old, new)


IMAGES = sub(p1.IMAGES, "items = [it for it in items if it['part'] == ('tune' if PHASE == 'TUNE' else 'smoke')]",
             "items = [it for it in items if it['part'] in ('eval', 'neg')]\n"
             "items.sort(key=lambda it: it['part'] != 'eval')  # toute l'éval d'abord, puis les négatifs\n"
             "shutil.copy(f'{IN}/perturb.py', CODE)\n"
             "from perturb import perturb")
IMAGES = sub(IMAGES, """        if hashlib.md5(crop.tobytes()).hexdigest() != it['crop_md5']:
            mism.append(it['stem'])""", """        if hashlib.md5(crop.tobytes()).hexdigest() != it['crop_md5']:
            mism.append(it['stem'])
        if it['part'] == 'neg':  # même perturbation, même clé que le treillis A1 ; image perturbée contrôlée
            crop = perturb(crop, it['mode'], it['cle_perturbation'])
            if hashlib.md5(crop.tobytes()).hexdigest() != it['pert_md5']:
                mism.append(it['stem'])""")

IMAGES = sub(IMAGES, "        subprocess.run(['wget', '-q', '-O', path, url], check=True)",
             "        subprocess.run(['wget', '-q', '--timeout=60', '--tries=3', '-O', path, url], check=True,\n"
             "                       timeout=max(60, SOFT - time.time()))  # audit pT : téléchargement borné")

CHRONO = r'''# Audit pT (P3) : PLAFOND DUR. Chrono dès la PREMIÈRE cellule. SOFT (5 h 45) : aucune ligne commencée après ;
# GEN_END (5 h 50) : chaque generate reçoit max_time = min(600 s, temps restant) et s'arrête de lui-même ;
# DEADLINE (6 h) : une sentinelle écrit dots_p3_arret_dur.json puis termine le processus (ne sert que si quelque
# chose reste bloqué ; les sorties sont écrites ligne par ligne).
import time, threading, os, json
T_START = time.time()
SOFT, GEN_END, DEADLINE = T_START + 5.75 * 3600, T_START + (5 * 60 + 50) * 60, T_START + 6 * 3600
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
def _sentinelle():
    while time.time() < DEADLINE - 30:
        time.sleep(15)
    try:
        with open('/kaggle/working/dots_p3_arret_dur.json', 'w') as f:
            json.dump({'arret_dur': True, 'secondes': round(time.time() - T_START)}, f)
            f.flush(); os.fsync(f.fileno())
        # pA (a) : tuer AUSSI les sous-processus (wget, etc.). On tue le SOUS-ARBRE du noyau (descendants), pas tout
        # le groupe de processus, qui peut contenir l'exécutant de Kaggle chargé de sauver /kaggle/working.
        try:
            import psutil
            for ch in psutil.Process().children(recursive=True):
                ch.kill()
        except Exception:
            pass
    finally:
        os._exit(3)
threading.Thread(target=_sentinelle, daemon=True).start()
print('plafond dur armé : 6 h')'''

RUN = sub(p1.RUN, """assert len(stems) == len(set(stems)) == (400 if PHASE == 'TUNE' else 3), (PHASE, len(stems), len(set(stems)))
assert set(stems) <= set(LAT) and len(LAT) == 403, 'couverture des treillis'""",
          """assert len(stems) == len(set(stems)) == 3005, (len(stems), len(set(stems)))
assert sum(it['part'] == 'eval' for it in items) == 2405 and set(stems) == set(LAT), 'panneau ou treillis'""")
RUN = sub(RUN, """CONFIGS = [('libre', None, 1), ('faisceau_libre', None, 4)] + (
    [('guide_l0.5', 0.5, 4)] if PHASE == 'SMOKE' else [(f'guide_l{l}', l, 4) for l in (0.0, 0.3, 0.5, 0.7)])""",
          """# P3 : λ* = 0,3 GELÉ (choisi sur TUNE) ; faisceau_libre retiré (identique à guide_l0.0 sur TUNE)
CONFIGS_EVAL = [('libre', None, 1), ('guide_l0.0', 0.0, 4), ('guide_l0.3', 0.3, 4)]
CONFIGS_NEG = [('libre', None, 1), ('guide_l0.3', 0.3, 4)]
CONFIGS = CONFIGS_EVAL
INTERROMPU = False""")
RUN = sub(RUN, """    for k, it in enumerate(items):
        img = CANVAS[it['stem']]""", """    for k, it in enumerate(items):
        if time.time() > SOFT:  # arrêt préventif à 5 h 45 (plafond dur : sentinelle à 6 h)
            INTERROMPU = True
            print('PLAFOND atteint : arrêt avant', it['stem'], flush=True)
            break
        img = CANVAS[it['stem']]""")
RUN = sub(RUN, """                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, **kw)""",
          """                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                         max_time=max(1.0, min(600.0, GEN_END - time.time())), **kw)""")
RUN = sub(RUN, """                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=nb,
                                         length_penalty=0.0, logits_processor=[guide])""",
          """                    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=nb,
                                         length_penalty=0.0, logits_processor=[guide],
                                         max_time=max(1.0, min(600.0, GEN_END - time.time())))""")
RUN = sub(RUN, """            fin = 'eos' if gen and gen[-1] in eos_ids else ('plafond' if len(gen) >= max_new else 'autre')""",
          """            fin = 'eos' if gen and gen[-1] in eos_ids else ('plafond' if len(gen) >= max_new else
                  ('temps' if time.time() >= GEN_END - 1 or time.time() - t0 >= 599 else 'autre'))""")
RUN = sub(RUN, """        for name, lam, nb in CONFIGS:""", """        for name, lam, nb in (CONFIGS_EVAL if it['part'] == 'eval' else CONFIGS_NEG):""")
RUN = sub(RUN, """           'python': sys.version.split()[0], 'lignes': len(items), 'configs': CONFIGS, 'jetons_permis': len(ALLOWED),""",
          """           'python': sys.version.split()[0], 'lignes_prevues': len(items), 'configs': CONFIGS,
           'jetons_permis': len(ALLOWED), 'configs_negatifs': CONFIGS_NEG, 'interrompu': INTERROMPU,
           'lignes_traitees': len({json.loads(l)['stem'] for l in open(dest, encoding='utf-8')}),""")
RUN = sub(RUN, "            fh.write(json.dumps({'stem': it['stem'], 'config': name,",
          "            fh.write(json.dumps({'stem': it['stem'], 'part': it['part'], 'mode': it.get('mode'), 'config': name,")
RUN = sub(RUN, """            fh.flush()
""", """            fh.flush()
            os.fsync(fh.fileno())  # pA (b) : chaque ligne sur disque avant la suivante
""")


def notebook(manifest_sha: str) -> dict:
    head = ("# babaorum — dots.mocr guidé par le CTC, P3 ÉVAL (SECOND REGARD) + négatifs Q2\n\n"
            "docs/READER_DOTS_CTC_PLAN.txt (PRÉ-ENREGISTREMENT P3). RÉSERVE B4 : la tête A1 a été choisie sur tout "
            "press/dev. GPU T4 ×2 (accélérateur « GPU T4 x2 » et Internet ON obligatoires) ; plafond dur 6 h. λ* = 0,3 gelé.")
    return {"cells": [p1.md(head), p1.code(CHRONO), p1.code(p1.SETUP.format(phase="EVAL", manifest_sha=manifest_sha)),
                      p1.code("%%bash\nset -e\npip install -q 'transformers==4.51.3' 'accelerate>=0.34' 2>&1 | tail -1"),
                      p1.code(IMAGES), p1.code(p1.MODEL), p1.code(RUN)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    out = Path(a.inputs)
    assert not out.exists() or not any(out.iterdir()), "dossier d'entrées non vide"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(AXEL / "src"))
    sys.path.insert(0, str(HERE / "src"))
    from PIL import Image

    from axel.image.crops import crop_line
    from babaorum.reader.perturb import perturb

    Image.MAX_IMAGE_PIXELS = None
    b0 = R / "reader_r6_B0_eval.jsonl.gz"
    assert hashlib.sha256(b0.read_bytes()).hexdigest() == B0_SHA
    ev = [json.loads(x)["stem"] for x in gzip.open(b0, "rt")]
    assert len(ev) == len(set(ev)) == 2405
    tune = {r["stem"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    assert not tune & set(ev), "tune et éval se recouvrent"
    by = {}
    for x in gzip.open(AXEL / "data/e8/kaggle/e8_items.jsonl.gz", "rt"):
        r = json.loads(x)
        by[r["stem"]] = r
    items = [{"stem": s, "part": "eval", "image": by[s]["image"], "box": by[s]["box"], "polygon": by[s]["polygon"],
              "truth": by[s]["truth"], "crop_md5": by[s]["crop_md5"]} for s in ev]
    assert all(by[s]["split"] == "dev" for s in ev)
    pert = [json.loads(x) for x in open(R / "reader_items_photo1983_pert200.jsonl")]
    assert len(pert) == 200
    for r in pert:
        stem = r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"
        with Image.open(r["image"]) as im:
            crop = crop_line(im.convert("L"), tuple(r["box"]), r.get("polygon"), pad=4).image
        md5 = hashlib.md5(crop.tobytes()).hexdigest()
        for mode, suffix in MODES.items():
            p = perturb(crop, mode, stem + suffix)
            items.append({"stem": f"{stem}#{mode}", "part": "neg", "mode": mode, "src_stem": stem,
                          "cle_perturbation": stem + suffix, "image": "newseye/" + Path(r["image"]).name,
                          "box": r["box"], "polygon": r.get("polygon"), "truth": r["truth"], "crop_md5": md5,
                          "pert_md5": hashlib.md5(p.tobytes()).hexdigest()})
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    want_ev = set(ev)
    got = 0
    with open(out / "a1.jsonl", "w") as g:
        for x in gzip.open(AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz", "rt"):
            r = json.loads(x)
            if r.get("stem") in want_ev:
                g.write(json.dumps({"stem": r["stem"], "logp": r["logp"]}) + "\n")
                got += 1
        neg_stems = {i["src_stem"] for i in items if i["part"] == "neg"}
        for mode in MODES:
            for x in gzip.open(R / f"reader_a1_photo1983_{mode}.jsonl.gz", "rt"):
                r = json.loads(x)
                if r.get("stem") in neg_stems:
                    g.write(json.dumps({"stem": f"{r['stem']}#{mode}", "logp": r["logp"]}) + "\n")
                    got += 1
    assert got == 3005, got
    shutil.copy(AXEL / "runs/e17/kaggle/a1/timing.json", out)
    for f in (HERE / "src/babaorum/reader/ctc_prefix.py", HERE / "src/babaorum/reader/dots_guided.py",
              HERE / "src/babaorum/reader/dots_beam.py", HERE / "src/babaorum/reader/perturb.py",
              AXEL / "src/axel/image/crops.py", AXEL / "src/axel/image/vlm.py", AXEL / "src/axel/image/lattice.py"):
        shutil.copy(f, out)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    manifest_sha = hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest()
    (out / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"mabaku/{DATASET}",
                                                           "licenses": [{"name": "other"}]}))
    (NB / "kaggle_dots_p3_eval.ipynb").write_text(json.dumps(notebook(manifest_sha), indent=1, ensure_ascii=False))
    print(json.dumps({"items": len(items), "a1": got, "manifest_sha256": manifest_sha}, indent=1))


if __name__ == "__main__":
    main()
