"""Generates notebooks/kaggle_p04_blla.ipynb (LOT-P04). A working file: NOT committed, NOT run here, nothing uploaded."""
import hashlib
import json
from pathlib import Path

D = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages/kaggle_dataset/babaorum-p04"
MANIFEST_SHA = hashlib.sha256((D / "MANIFEST_SHA256.txt").read_bytes()).hexdigest()
EXCL_SHA = hashlib.sha256((D / "exclusion_list.json").read_bytes()).hexdigest()
BLLA_SHA = "77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056"   # blla.mlmodel shipped with kraken 7.1.1
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})


md("""
# LOT-P04 — `blla` affiné sur la presse, SANS le dev n° 2 (Kaggle T4, une session, plafond 6 h)
Adapté de `colab_seg_train.ipynb` (cellule 13) : `RUN['blla']` seul, `BUDGET_H = 6`. Entraînement : 12 pages READ (311 zones) ; validation interne : 6 pages
(127 zones) tirées par groupes (numéros) — elle ne sert qu'à arrêter l'entraînement. **Aucune page exclue** dans le jeu ; la liste d'exclusion est vérifiée par SHA et le
carnet **échoue** si une zone exclue est présente. **Aucun `segtest` sur le dev** ; le modèle est archivé sans être ouvert par une mesure. Tout s'écrit sous `/kaggle/working`.
""")
code(f"""
import os, sys, json, glob, hashlib, shutil, subprocess, time, signal
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rapidfuzz'], check=True)   # KAGGLE_INSTALL.txt (p9): hors image, importé par axel.reader.metric
RUN = {{'blla': True}}           # le seul interrupteur de ce carnet (les autres étapes d'E21 n'y sont pas)
BUDGET_H = 6                    # plafond de la session : à la limite, arrêt propre, le meilleur point est gardé
SEED = 20261006
SMOKE = os.environ.get('P04_SMOKE') == '1'   # essai local : 1 époque sur 2 zones, CPU (annoncé, jamais utilisé sur Kaggle)
MANIFEST_SHA256 = '{MANIFEST_SHA}'
EXCLUSION_SHA256 = '{EXCL_SHA}'
BLLA_SHA256 = '{BLLA_SHA}'      # blla.mlmodel livré avec kraken 7.1.1 : le point de départ
WORK = Path('/kaggle/working')
ARCHIVE_SHA256 = 'ee2bfe407337e5849e08442d606a8da1bf1c275cd843420546260ab43be7a86c'
N_PARTS = 27
# the dataset arrives as 9 MB pieces (read-only input): reassembled, checked by SHA, then extracted into a fresh working folder (v2 rule, pA)
parts = sorted(glob.glob('/kaggle/input/**/part_??', recursive=True), key=os.path.basename)
assert len(parts) == N_PARTS, (len(parts), parts[:3])
with open('/kaggle/working/babaorum-p04.tar.gz', 'wb') as g:
    for q in parts:
        g.write(open(q, 'rb').read())
assert hashlib.sha256(open('/kaggle/working/babaorum-p04.tar.gz', 'rb').read()).hexdigest() == ARCHIVE_SHA256, 'SHA de l archive incorrect'
import tarfile
os.makedirs('/kaggle/working/dataset', exist_ok=True)
with tarfile.open('/kaggle/working/babaorum-p04.tar.gz') as t:
    t.extractall('/kaggle/working/dataset')
IN = Path('/kaggle/working/dataset/babaorum-p04')
print('jeu extrait :', IN, '| SMOKE', SMOKE, '| BUDGET_H', BUDGET_H)
""")
md("## 1. Intégrité du jeu et liste d'exclusion — échec si une zone exclue est présente")
code("""
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
assert sha256(IN / 'MANIFEST_SHA256.txt') == MANIFEST_SHA256, 'le manifeste n est pas celui annoncé'
bad = []
for line in (IN / 'MANIFEST_SHA256.txt').read_text().splitlines():
    h, rel = line.split('  ', 1)
    if sha256(IN / rel) != h:
        bad.append(rel)
assert not bad, f'fichiers modifiés : {bad[:5]}'
assert sha256(IN / 'exclusion_list.json') == EXCLUSION_SHA256, 'la liste d exclusion n est pas celle annoncée'
excl = json.loads((IN / 'exclusion_list.json').read_text())
zones = sorted(p.name for p in (IN / 'lines/read').glob('*.xml'))
present_excluded_files = [z for z in zones if 'lines/read/' + z in set(excl['zone_xml_files'])]
import re
stems = {re.match(r'read_(.+)_z\\d+_\\d+\\.xml$', z).group(1) for z in zones}
numbers = {str(int(re.search(r'(?<!\\d)(\\d{6,8})(?!\\d)', s).group(1))) for s in stems}
present_excluded_pages = sorted(stems & set(excl['pages']))
present_excluded_numbers = sorted(numbers & set(excl['numbers']))
assert not present_excluded_files and not present_excluded_pages and not present_excluded_numbers, (present_excluded_files[:3], present_excluded_pages, present_excluded_numbers)
train = [l for l in (IN / 'lists/train.txt').read_text().split()]
val = [l for l in (IN / 'lists/validation.txt').read_text().split()]
assert len(train) == 311 and len(val) == 127 and not set(train) & set(val), (len(train), len(val))
tp = {re.match(r'read/read_(.+)_z\\d+_\\d+\\.xml$', z).group(1) for z in train}; vp = {re.match(r'read/read_(.+)_z\\d+_\\d+\\.xml$', z).group(1) for z in val}
tn = {re.search(r'(?<!\\d)(\\d{6,8})(?!\\d)', s).group(1).lstrip('0') for s in tp}; vn = {re.search(r'(?<!\\d)(\\d{6,8})(?!\\d)', s).group(1).lstrip('0') for s in vp}
assert not tn & vn, 'un numéro est à la fois en entraînement et en validation'
assert tp == {r['page'] for r in json.loads((IN / 'train.json').read_text())} and vp == {r['page'] for r in json.loads((IN / 'validation.json').read_text())}
print('jeu intact :', len(open(IN / 'MANIFEST_SHA256.txt').read().splitlines()), 'fichiers | exclusion ok |', len(train), 'zones d entraînement,', len(val), 'de validation')
""")
md("## 2. GPU, kraken 7.1.1 et point de départ (SHA du `blla` livré)")
code("""
import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('GPU', GPU, '| bf16', BF16, '| torch', torch.__version__)
if not SMOKE:
    assert torch.cuda.is_available(), 'pas de GPU : le run P-04 exige le T4'
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'kraken==7.1.1'], check=True)   # also in the smoke: the fresh venv holds what the carnet installs
# v5 (failure of version 1 on Kaggle, 17/10): kraken 7.1.1 pins safetensors~=0.7.0, while the transformers of the Kaggle image needs >=0.8 and is
# imported on the way by ketos; kraken does not use transformers: it is removed from the session (no-op in the smoke venv, where it is absent)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'transformers', 'diffusers'], check=False)
import importlib.util
assert importlib.util.find_spec('transformers') is None, 'transformers is still importable'
import importlib.metadata as md_
assert md_.version('kraken') == '7.1.1', md_.version('kraken')
import kraken
BLLA = os.path.join(os.path.dirname(kraken.__file__), 'blla.mlmodel')
assert sha256(BLLA) == BLLA_SHA256, 'le blla de départ n est pas celui annoncé : ' + sha256(BLLA)
print('kraken', md_.version('kraken'), '| blla de départ SHA256', BLLA_SHA256)
""")
md("## 3. Entrées copiées sous `/kaggle/working` (seul endroit où l'on écrit)")
code("""
L = WORK / 'lines'
if L.exists():
    shutil.rmtree(L)
shutil.copytree(IN / 'lines', L)
for root, dirs, files in os.walk(L):          # /kaggle/input est en lecture seule : les copies doivent être modifiables (listes à écrire)
    for n in dirs + files:
        os.chmod(os.path.join(root, n), 0o755 if n in dirs else 0o644)
os.chmod(L, 0o755)
tr, va = train, val
if SMOKE:
    tr = train[:2]                   # essai local : 2 zones prises dans les 12 pages d entraînement (aucune page exclue, aucune page de validation)
    va = tr                          # ketos veut un jeu de validation : les mêmes 2 zones (aucune page de validation n est ouverte)
(L / 'train.txt').write_text('\\n'.join(tr) + '\\n')
(L / 'validation.txt').write_text('\\n'.join(va) + '\\n')
print(len(tr), 'zones d entraînement,', len(va), 'de validation')
""")
md("## 4. Entraînement `ketos segtrain` (arrêt précoce sur la validation interne ; plafond de temps)")
code("""
RUNS = WORK / 'runs'; OUTDIR = RUNS / 'blla_P04'
OUTDIR.mkdir(parents=True, exist_ok=True)
device = 'cpu' if SMOKE else 'cuda:0'
prec = '32-true' if SMOKE else '16-mixed'      # v5: 16-mixed as announced to pJ (the T4 reports bf16 as supported, emulated)
epochs = '1' if SMOKE else '50'
# -q fixed in the smoke: ketos ignores -N with -q early (the smoke ran 33 stages, found 2026-10-17)
cmd = ['ketos', '-d', device, '--precision', prec, '--workers', '0' if SMOKE else '4', 'segtrain', '-i', BLLA, '--resize', 'union', '-f', 'page',
       '-t', 'train.txt', '-e', 'validation.txt', '-o', str(OUTDIR), '-q', 'fixed' if SMOKE else 'early', '--lag', '8', '-N', epochs, '--augment']
print(' '.join(cmd))
if not SMOKE:
    # v5 pre-flight IN THE KAGGLE IMAGE, before the long run: 1 epoch on the first 2 training zones (no validation page opened), same device and
    # precision; its output goes to a separate folder and is never used
    pre = RUNS / 'preflight'; pre.mkdir(exist_ok=True)
    (L / 'preflight.txt').write_text('\\n'.join(train[:2]) + '\\n')
    pcmd = ['ketos', '-d', device, '--precision', prec, '--workers', '0', 'segtrain', '-i', BLLA, '--resize', 'union', '-f', 'page',
            '-t', 'preflight.txt', '-e', 'preflight.txt', '-o', str(pre), '-q', 'fixed', '-N', '1']
    pr = subprocess.run(pcmd, cwd=L, capture_output=True, text=True)
    print('pre-flight code', pr.returncode)
    assert pr.returncode == 0, pr.stdout[-3000:] + pr.stderr[-3000:]
t0 = time.time()
log = open(RUNS / 'ketos_blla_P04.log', 'w')
proc = subprocess.Popen(cmd, cwd=L, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
stopped = None
while proc.poll() is None:
    time.sleep(30)
    if time.time() - t0 > BUDGET_H * 3600:
        stopped = 'plafond de temps atteint'
        os.killpg(proc.pid, signal.SIGINT)        # arrêt propre : le meilleur point déjà écrit est gardé
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
        break
log.close()
elapsed = time.time() - t0
print('code', proc.returncode, '| durée', round(elapsed), 's |', stopped or 'terminé de lui-même')
assert proc.returncode == 0 or stopped, (proc.returncode, open(RUNS / 'ketos_blla_P04.log').read()[-3000:])
""")
md("## 5. Archive du modèle : SHA-256, rapport ; AUCUNE mesure, AUCUN `segtest` sur le dev")
code("""
models = sorted(glob.glob(str(OUTDIR / 'best_*.safetensors')))
assert models, 'aucun best_*.safetensors : ' + str(os.listdir(OUTDIR))
report = {'lot': 'P-04', 'session_smoke': SMOKE, 'seconds': round(elapsed), 'stopped': stopped, 'ketos_returncode': proc.returncode,
          'kraken': md_.version('kraken'), 'blla_start_sha256': BLLA_SHA256, 'manifest_sha256': MANIFEST_SHA256, 'exclusion_sha256': EXCLUSION_SHA256,
          'train_zones': len(tr), 'validation_zones': len(va), 'budget_h': BUDGET_H, 'gpu': GPU,
          'models': {os.path.basename(m): sha256(m) for m in models}, 'checkpoints': sorted(os.path.basename(p) for p in glob.glob(str(OUTDIR / '*.ckpt')))}
(WORK / 'p04_report.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("kaggle_p04_blla.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("notebook written:", len(cells), "cells ;", MANIFEST_SHA[:12], EXCL_SHA[:12])
