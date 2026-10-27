"""Generates notebooks/kaggle_p07b.ipynb (LOT-P07b v2). A working file: NOT committed, NOT run here, nothing uploaded."""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})


md("""
# LOT-P07b — chaîne C1+M+A sur les 20 pages restantes du dev n° 2 (Kaggle CPU)
Session **SESSION** (1 ou 2) : 10 pages chacune. Aucune vérité (aucun XML) ; aucun test ; aucun réglage ; aucune écriture ailleurs que `/kaggle/working`.
Le contrôle de portabilité (page *La Presse* 0545076, sorties du Mac de P-07) tourne **au début de la session 1** ; au-delà d'un seuil, **arrêt** avec rapport.
Le SHA du carnet et celui du jeu d'entrée (`MANIFEST_SHA256.txt`) ont été annoncés à pJ avant le lancement.
""")
md("""
## Étape 0 — le jeu d'entrée arrive en morceaux de 9 Mo (méthode de p9, lot 2d, sans jeton) : recollage, SHA de l'archive, extraction
""")
code("""
import glob, hashlib, os, tarfile
from pathlib import Path
ARCHIVE_SHA256 = 'ccfc91bba35df7a54994de0ed9c928ca98b2c973ab13d7db7da48425f7e4c2d5'
N_PARTS = 76
parts = sorted(glob.glob('/kaggle/input/**/part_??', recursive=True), key=os.path.basename)
assert len(parts) == N_PARTS, (len(parts), parts[:3])
with open('/kaggle/working/babaorum-p07b.tar.gz', 'wb') as g:
    for q in parts:
        g.write(open(q, 'rb').read())
got = hashlib.sha256(open('/kaggle/working/babaorum-p07b.tar.gz', 'rb').read()).hexdigest()
assert got == ARCHIVE_SHA256, f'SHA de l archive : {got} != {ARCHIVE_SHA256}'
os.makedirs('/kaggle/working/dataset', exist_ok=True)
with tarfile.open('/kaggle/working/babaorum-p07b.tar.gz') as t:
    t.extractall('/kaggle/working/dataset')
print('archive recollée et extraite :', got)
""")
code("""
SESSION = 1                      # 1 ou 2 ; à fixer AVANT d'exécuter
RUN_PORTABILITY_CONTROL = (SESSION == 1)
import os, sys, json, hashlib, shutil, subprocess, time, tarfile, glob
from pathlib import Path
IN = Path('/kaggle/working/dataset/babaorum-p07b')   # racine du jeu : l'archive extraite (étape 0)
W = Path('/kaggle/working')
AX = W / 'axel'
RUNS = W / 'runs'                # AXEL_RUNS : chemin absolu, exporté avant chaque processus
RUNS_CTRL = W / 'runs_control'
assert IN.exists(), 'jeu d entrée absent : /kaggle/working/dataset/babaorum-p07b'
manifest_sha = hashlib.sha256((IN / 'MANIFEST_SHA256.txt').read_bytes()).hexdigest()
print('SESSION', SESSION, '| manifeste du jeu SHA256', manifest_sha)
print((IN / 'AXEL_COMMIT.txt').read_text())
""")
md("## 0. Intégrité du jeu d'entrée et absence de vérité")
code("""
bad = []
for line in (IN / 'MANIFEST_SHA256.txt').read_text().splitlines():
    h, rel = line.split('  ', 1)
    if hashlib.sha256((IN / rel).read_bytes()).hexdigest() != h:
        bad.append(rel)
assert not bad, f'fichiers modifiés : {bad[:5]}'
assert not list(IN.rglob('*.xml')), 'aucun XML (aucune vérité) ne doit se trouver dans le jeu'
import tarfile as _tf
assert not [m for m in _tf.open(IN / 'axel.tar.gz').getnames() if m.lower().endswith('.xml')], 'l archive d axel contient un XML'
print('jeu intact :', len(open(IN / 'MANIFEST_SHA256.txt').read().splitlines()), 'fichiers')
""")
md("## 1. Environnements figés (trois, comme sur le Mac) — versions imprimées, écart avec les gels du Mac rapporté")
code("""
PINS = {
  'layout-env': dict(python='3.12', pkgs=['torch==2.14.1', 'torchvision==0.29.1', 'transformers==5.18.0', 'numpy==2.5.3', 'pillow==12.3.0', 'safetensors', 'huggingface_hub', 'opencv-python-headless==5.0.0.93', 'scipy==1.18.1', 'shapely==2.1.2', 'lxml==6.1.3', 'httpx==0.28.1', 'certifi==2026.7.22']),
  'kraken-env': dict(python='3.11', pkgs=['kraken==7.1.1', 'torch==2.14.0', 'numpy==2.4.6', 'coremltools==9.0', 'pillow', 'lxml', 'opencv-python-headless==5.0.0.93']),
  'axel-venv':  dict(python='3.12', pkgs=['numpy', 'pillow', 'lxml', 'rapidfuzz', 'scipy', 'shapely']),
}
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
ENV = {}
for name, spec in PINS.items():
    d = Path('/tmp/p07b_envs') / name  # v10 : hors de /kaggle/working (sorties légères)
    if not d.exists():
        subprocess.run(['uv', 'venv', '--python', spec['python'], str(d)], check=True)
        subprocess.run(['uv', 'pip', 'install', '--python', str(d / 'bin/python'), '--extra-index-url', 'https://download.pytorch.org/whl/cpu', '--index-strategy', 'unsafe-best-match', *spec['pkgs']], check=True)
    ENV[name] = str(d / 'bin/python')
    out = subprocess.run([ENV[name], '-c', 'import sys;print(sys.version.split()[0])'], capture_output=True, text=True).stdout.strip()
    print(name, 'python', out)
checks = {'layout-env': ['torch', 'transformers', 'numpy', 'PIL'], 'kraken-env': ['kraken', 'torch', 'numpy', 'coremltools']}
import importlib.metadata  # noqa
for name, mods in checks.items():
    code_ = 'import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in %r}))' % ([ {'PIL': 'pillow'}.get(x, x) for x in mods ],)
    print(name, subprocess.run([ENV[name], '-c', code_], capture_output=True, text=True).stdout.strip())
# écart avec les gels du Mac : calculé, publié dans le rapport de fin, jamais silencieux
FREEZE_DIFF = {}
for name in PINS:
    mac = {l.split('==')[0].lower().replace('_', '-'): l.split('==')[1] for l in (IN / f'mac_freeze_{name}.txt').read_text().splitlines() if '==' in l}
    here_txt = subprocess.run(['uv', 'pip', 'freeze', '--python', ENV[name]], capture_output=True, text=True).stdout
    here = {l.split('==')[0].lower().replace('_', '-'): l.split('==')[1] for l in here_txt.splitlines() if '==' in l}
    FREEZE_DIFF[name] = {k: (mac[k], here.get(k)) for k in mac if here.get(k) != mac[k] and k in here}
    print(name, ': paquets communs aux deux gels', len([k for k in mac if k in here]), '| versions différentes', len(FREEZE_DIFF[name]),
          '| absents ici', len([k for k in mac if k not in here]))
""")
md("## 2. Code d'axel (commit de la branche), modèles en lecture, instantané Hugging Face hors ligne")
code("""
if not AX.exists():
    AX.mkdir()
    with tarfile.open(IN / 'axel.tar.gz') as t:
        t.extractall(AX)
commit = (IN / 'AXEL_COMMIT.txt').read_text().split()[0]
for rel in ('runs/e8/models/mccatmus_press.mlmodel', 'runs/e5/models/german_print_s.mlmodel'):
    dst = AX / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = IN / 'models' / Path(rel).name
    shutil.copy2(src, dst)          # modèles lus sous <axel>/runs, hors AXEL_RUNS (comme sur le Mac)
shutil.copy2(IN / 'scripts/p07_read_rest_all.py', W / 'p07_read_rest_all.py')
# garde-fou : les modules d'axel utilisés par chaque étape s'importent dans l'environnement qui les lance
SMOKE = {'axel-venv': 'import sys;sys.path[:0]=["%s/src","%s/scripts"];import axel.layout.fill,axel.layout.chains,axel.eval.seg_errors,axel.layout.pagetext,axel.layout.gt,axel.pipeline.route;import e28_fill,e32;print("ok")',
         'kraken-env': 'import sys;sys.path[:0]=["%s/src","%s/scripts"];import kraken,axel.image.kraken_read,axel.layout.chains;print("ok")',
         'layout-env': 'import torch,transformers,PIL;print("ok")'}
for name, snippet in SMOKE.items():
    snippet = snippet.replace('%s', str(W / 'axel')) if '%s' in snippet else snippet
    r = subprocess.run([ENV[name], '-c', snippet], capture_output=True, text=True)
    print(name, 'import', r.stdout.strip() or r.stderr[-600:])
    assert r.returncode == 0, f'{name} : un import de la chaîne échoue (voir ci-dessus) : arrêt'
assert not list(AX.rglob('*.xml')), 'aucun XML ne doit se trouver non plus dans l arbre d axel extrait'
route = (AX / 'src/axel/pipeline/route.py').read_text()
pages_all = json.loads((IN / f'pages_session{SESSION}.json').read_text())
for e in pages_all:    # le jeu est extrait sous /kaggle/working/dataset : les images y sont, pas sous /kaggle/input
    e['image'] = str(IN / 'images' / Path(e['image']).name)
assert all(f'"{p["page"]}"' in route for p in pages_all), 'route._YEARS lacks a page : wrong axel commit'
HF = IN / 'hf'
BASE_ENV = dict(os.environ, HF_HOME=str(HF), HF_HUB_CACHE=str(HF / 'hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
print('axel commit', commit, '| pages de la session :', len(pages_all))
""")
md("## 3. Racines de sortie, listes de pages (`xml` pointe vers un chemin inexistant) et fonctions de la chaîne")
code("""
def set_pages(root, entries):
    (root / 'e12').mkdir(parents=True, exist_ok=True)
    (root / 'e12/pages.json').write_text(json.dumps(entries, indent=1))
    for e in entries:
        assert not Path(e['xml']).exists() and Path(e['image']).exists(), e['page']
PAGES = [e['page'] for e in pages_all]
def env_for(root):
    return dict(BASE_ENV, AXEL_RUNS=str(root))        # chemin absolu, exporté avant chaque processus
def run(label, env_name, args, root, cwd=AX):
    t0 = time.time()
    p = subprocess.run([ENV[env_name], *args], cwd=cwd, env=env_for(root), capture_output=True, text=True)
    print(f'[{label}] code {p.returncode} en {time.time() - t0:.0f} s')
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:]); raise SystemExit(f'étape {label} échouée')
    return p.stdout
def complete(root, folder, pages):
    return all((root / folder / f'{p}.json').exists() and (root / folder / f'{p}.json').stat().st_size > 0 for p in pages)
def produced(root, folder, pages):
    # contrôle « N sur N » : une page sans fichier = l'étape n'a rien produit (certaines étapes d'axel sautent en silence)
    miss = [p for p in pages if not (root / folder / f'{p}.json').exists() or (root / folder / f'{p}.json').stat().st_size == 0]
    assert not miss, f'{folder} : {len(pages) - len(miss)}/{len(pages)} fichiers ; manquent {miss[:3]}'
    print(f'  {folder} : {len(pages)}/{len(pages)}')
def archive(n, label, root, folders, tag):
    out = W / f'p07b_{tag}_{n}_{label}.tar.gz'
    with tarfile.open(out, 'w:gz') as t:
        for f in folders:
            if (root / f).exists():
                t.add(root / f, arcname=f'{root.name}/{f}')
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f'  archive {out.name} {out.stat().st_size} octets SHA256 {h}')
    return h
S = 'scripts/'
# (étiquette, environnement, arguments, dossiers contrôlés « N sur N », dossiers à vider si l'étape est refaite)
STEPS = [
  ('detect',      'layout-env', [S + 'e12_detect.py', '--det', 'ppdl3', '--split', 'dev'],                      ['e12/blocks/ppdl3'],  ['e12/blocks/ppdl3']),
  ('blocks',      'kraken-env', [S + 'e12_kraken.py', 'blocks', '--det', 'ppdl3', '--zones', '--split', 'dev'], ['e12/lines/ppdl3_z'], ['e12/lines/ppdl3_z']),
  ('e12_read_z',  'kraken-env', [S + 'e12_kraken.py', 'read', '--src', 'ppdl3_z', '--split', 'dev'],            ['e12/read/ppdl3_z'],  ['e12/read/ppdl3_z']),
  ('c1_read',     'kraken-env', [S + 'e22_read.py', '--bench', 'e12', '--src', 'ppdl3_z', '--split', 'dev'],    ['e22/read/ppdl3_z'],  ['e22/read/ppdl3_z']),
  ('e28_seg',     'kraken-env', [S + 'e28_fill.py', 'seg', '--what', 'rest'],                                    ['e28/lines/rest_z'],  ['e28/lines']),
  ('e28_select',  'axel-venv',  [S + 'e28_fill.py', 'select'],                                                   ['e28/add/a_z'],       ['e28/add']),
  ('e28_read',    'kraken-env', [S + 'e28_fill.py', 'read'],                                                     ['e28/read/mccatmus_press/a_z', 'e28/read/german_print_s/a_z'], ['e28/read']),
  ('e32_select',  'axel-venv',  [S + 'e32.py', 'select', '--bench', 'dev'],                                      ['e32/dev/add/a_m'],   ['e32/dev/add']),
  ('e32_read',    'kraken-env', [S + 'e32.py', 'read', '--bench', 'dev'],                                        [],                    ['e32/dev/read']),   # e32 extra : seules les pages avec lignes ajoutées sans lecture en cache
]
def run_chain(root, pages, tag, with_archive=True):
    # Une sortie existante n'est reprise que si l'étape est COMPLÈTE pour toute la liste ; sinon l'étape se refait, et toutes les suivantes aussi.
    times, arch, redo = {}, {}, False
    for n, (label, env_name, args, folders, clean) in enumerate(STEPS, 1):
        t0 = time.time()
        done = bool(folders) and all(complete(root, f, pages) for f in folders)
        if done and not redo:
            print(f'[{label}] étape complète pour la liste : reprise des sorties existantes')
        else:
            redo = True                                           # cascade : une étape refaite invalide les suivantes
            for c in clean:
                shutil.rmtree(root / c, ignore_errors=True)       # jamais de fichier partiel repris
            run(label, env_name, args, root)
        for f in folders:
            produced(root, f, pages)
        if with_archive:
            arch[label] = archive(n, label, root, folders + (['e32/dev/read'] if label == 'e32_read' else []), tag)
        times[label] = round(time.time() - t0)
    return times, arch
""")
md("""
## 4. Contrôle de portabilité (session 1 seulement) — seuils écrits avant le lancement (v2, étendus à c1+ma par la v4)
*La Presse* 0545076 passe **toute la chaîne** jusqu'à E32 (c1+ma) ; ses sorties sont comparées à celles du Mac de P-07.
Seuils : détections ≥ 0,3 : même nombre et mêmes étiquettes une par une ; lignes de blla : écart ≤ 1 % par zone et au total ; pour **c1 et pour c1+ma** : écart du nombre de lignes ≤ 1 %,
chaque ligne du Mac retrouvée (couverture de ligne de base ≥ 0,95) pour ≥ 99 %, CER Mac/Kaggle des lignes retrouvées ≤ 0,1 % au total. Un seul seuil dépassé = arrêt.
""")
code("""
PORT = {}
if RUN_PORTABILITY_CONTROL:
    page = 'La_Presse_0545076_T0000002'
    set_pages(RUNS_CTRL, [dict(e, image=str(IN / 'images' / Path(e['image']).name)) for e in json.loads((IN / 'pages_control.json').read_text())])
    run_chain(RUNS_CTRL, [page], 'ctrl', with_archive=False)
    ref = IN / 'reference_mac'
    sys.path[:0] = [str(AX / 'src')]
    from axel.layout.fill import baseline_cover, line_height
    tp = lambda s_: tuple((float(a), float(b)) for a, b in s_)
    def cov(a, b):
        ta = tp(a['baseline']); return baseline_cover(ta, tp(b['baseline']), 0.5 * line_height(tp(a['polygon']), ta))
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    # 1. détections ≥ 0,3
    def dets(path):
        return sorted((b.get('label'), round(b['score'], 3)) for b in json.loads(Path(path).read_text())['blocks'] if b['score'] >= 0.3)
    mac_d, kag_d = dets(ref / f'e12/blocks/ppdl3/{page}.json'), dets(RUNS_CTRL / f'e12/blocks/ppdl3/{page}.json')
    PORT['detections'] = {'mac': len(mac_d), 'kaggle': len(kag_d), 'same_labels_in_order': [a[0] for a in mac_d] == [a[0] for a in kag_d]}
    # 2. lignes de blla (par zone et au total)
    mac_l = json.loads((ref / f'e12/lines/ppdl3_z/{page}.json').read_text())['blocks']
    kag_l = json.loads((RUNS_CTRL / f'e12/lines/ppdl3_z/{page}.json').read_text())['blocks']
    nm, nk = sum(len(b['lines']) for b in mac_l), sum(len(b['lines']) for b in kag_l)
    kz = {b['id']: len(b['lines']) for b in kag_l}
    PORT['blla_lines'] = {'mac': nm, 'kaggle': nk, 'gap_total': abs(nm - nk) / max(nm, 1), 'zones_mac': len(mac_l), 'zones_kaggle': len(kag_l),
                          'gap_per_zone_max': max((abs(len(b['lines']) - kz.get(b['id'], 0)) / max(len(b['lines']), 1) for b in mac_l), default=0)}
    # 3-4. les chaînes c1 et c1+ma (lignes finales et lectures)
    out = subprocess.run([ENV['axel-venv'], str(AX / 'scripts/p07b_chain_units.py') if (AX / 'scripts/p07b_chain_units.py').exists() else str(IN / 'scripts/p07b_chain_units.py'),
                          '--axel', str(AX), '--page', page, '--out', str(W / 'kaggle_chain_units.json')], env=env_for(RUNS_CTRL), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    mac_u = json.loads((ref / 'chain_units.json').read_text())
    kag_u = json.loads((W / 'kaggle_chain_units.json').read_text())
    for chain in ('c1', 'c1+ma'):
        mu, ku = mac_u[chain], kag_u[chain]
        found, edit, chars = 0, 0, 0
        for m in mu:
            # couverture SYMÉTRIQUE (v. fumée n°3) : min(cov(m,k), cov(k,m)) ; argmax et seuil 0,95 portent sur cette valeur
            best = max(ku, key=lambda k: min(cov(m, k), cov(k, m)), default=None)
            if best is not None and min(cov(m, best), cov(best, m)) >= 0.95:
                found += 1; edit += lev(m['text'], best['text']); chars += max(len(m['text']), 1)
        PORT[chain] = {'lines_mac': len(mu), 'lines_kaggle': len(ku), 'gap_count': abs(len(mu) - len(ku)) / max(len(mu), 1),
                       'found_share': found / max(len(mu), 1), 'reading_cer_mac_vs_kaggle': edit / max(chars, 1)}
    ok = (PORT['detections']['mac'] == PORT['detections']['kaggle'] and PORT['detections']['same_labels_in_order']
          and PORT['blla_lines']['gap_total'] <= 0.01 and PORT['blla_lines']['gap_per_zone_max'] <= 0.01
          and all(PORT[c]['gap_count'] <= 0.01 and PORT[c]['found_share'] >= 0.99 and PORT[c]['reading_cer_mac_vs_kaggle'] <= 0.001 for c in ('c1', 'c1+ma')))
    PORT['passed'] = bool(ok)
    json.dump(PORT, open(W / 'portability_report.json', 'w'), indent=1)
    print(json.dumps(PORT, indent=1))
    assert ok, 'contrôle de portabilité NON passé : arrêt (rapport écrit dans portability_report.json)'
else:
    print('contrôle de portabilité : session 1 seulement')
""")
md("## 5. Chaîne de la session (mêmes étapes que P-07), contrôle « 10 sur 10 » et archive SHA après chaque étape")
code("""
set_pages(RUNS, pages_all)
TIMES, ARCH = run_chain(RUNS, PAGES, f's{SESSION}')
# lecture supplémentaire des candidats du reste non sélectionnés (v5) : à part, jamais dans read/ ni add/
t0 = time.time()
shutil.rmtree(W / 'read_rest_all', ignore_errors=True) if not all((W / 'read_rest_all' / f'{x}.json').exists() for x in PAGES) else None
if not all((W / 'read_rest_all' / f'{x}.json').exists() for x in PAGES):
    assert (W / 'axel/src/axel/image/kraken_read.py').is_file(), 'axel.image absent of the extracted tree'
    # v11 (failure of session 1 v3): the script finds axel at Path.home() / 'axel' (true on the Mac only); on Kaggle HOME=/root has no axel and
    # /kaggle/working/axel (the repository root, no __init__) was imported as a namespace package without axel.image. HOME=W for this step only.
    p = subprocess.run([ENV['kraken-env'], str(W / 'p07_read_rest_all.py')], cwd=AX, env=dict(env_for(RUNS), HOME=str(W)), capture_output=True, text=True)
    assert p.returncode == 0, (p.stdout[-2000:], p.stderr[-2000:])
miss = [x for x in PAGES if not (W / 'read_rest_all' / f'{x}.json').exists()]
assert not miss, miss
ARCH['read_rest_all'] = archive(len(STEPS) + 1, 'read_rest_all', W, ['read_rest_all'], f's{SESSION}'); TIMES['read_rest_all'] = round(time.time() - t0)
print(json.dumps(TIMES, indent=1))
""")
md("## 6. Rapport de fin de session")
code("""
extra = len(list((RUNS / 'e32/dev/read/mccatmus_press/extra').glob('*.json'))) if (RUNS / 'e32/dev/read/mccatmus_press/extra').exists() else 0
report = {'session': SESSION, 'axel_commit': commit, 'dataset_manifest_sha256': manifest_sha, 'pages': PAGES, 'seconds_by_step': TIMES, 'archives_sha256': ARCH,
          'e32_extra_files_mccatmus_press': extra, 'e32_extra_note': 'seules les pages avec des lignes ajoutées sans lecture en cache ont un fichier extra',
          'portability': PORT or 'non exécuté (session 2)', 'env_freeze_diff_vs_mac': FREEZE_DIFF, 'truth_in_chain': False}
json.dump(report, open(W / f'p07b_session{SESSION}_report.json', 'w'), indent=1)
print(json.dumps(report, indent=1)[:4000])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("kaggle_p07b.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("notebook written:", len(cells), "cells")
