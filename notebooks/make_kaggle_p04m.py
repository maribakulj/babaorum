"""Generates notebooks/kaggle_p04m.ipynb (AV-P04, texte 939a1430 et avenants v1.1 à v1.4). Working file: NOT committed, NOT run here, nothing uploaded.
ONE notebook (one SHA) for both passes: PASSE is deduced from the mounted sources (pass 1: neither the fine-tuned model nor the gate file;
pass 2: both). Derived from make_kaggle_p07b.py (same steps, same envs); no truth, no test, no tuning, nothing written outside /kaggle/working.
"""
import json
from pathlib import Path

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t, wrap=True):
    # every cell but the first runs inside try/except BaseException -> _abort() (Sonnet review, pQ B1): ANY failure (assert, SystemExit,
    # Popen error...) leaves a failure report and no open gate in the outputs; the exception is re-raised, so the run stops as before
    t = t.strip("\n")
    if wrap:
        t = "try:\n" + "\n".join("    " + x if x else x for x in t.splitlines()) + "\nexcept BaseException as _e:\n    _abort(_e)\n    raise"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)})


md("""
# AV-P04 — mesure unique du `blla` affiné : chaîne C1+M+A sur les 26 pages du dev n° 2 (Kaggle CPU)
**Passe 1** : `blla` livré, aucun modèle affiné monté ; à la fin, la **porte de pairage** (`porte_pairage.json`).
**Passe 2** : seulement si la porte est ouverte, ou si c'est la porte fermée 05c1bb52 acceptée par la note B de pA (25/10) ; `blla` affiné (`best_0.6432.safetensors`, SHA vérifié d'abord) sur `blocks` et `e28_seg`.
Une session de 12 h au plus par passe ; une passe qui la dépasse est incomplète, sans reprise. Aucune vérité (aucun XML) sur Kaggle.
""")
md("## 0. Sources montées, PASSE, consignation de la machine (première cellule ; en passe 2 : SHA du modèle et comparaison avec la passe 1 AVANT tout calcul)")
code("""
import glob, hashlib, json, os, platform, signal, subprocess, sys, time
from pathlib import Path
T0 = time.monotonic()                                                         # horloge MONOTONE du carnet (pQ B1), lue avant toute autre opération
LIMIT_H = 11.5                                                                # échéance : au-delà, passe incomplète (session Kaggle de 12 h au plus)
RESERVE_S = 900                                                               # réservé aux archives finales et au rapport : aucun sous-processus ne mord dessus
def elapsed_h():
    return (time.monotonic() - T0) / 3600
def remaining():
    return LIMIT_H * 3600 - (time.monotonic() - T0)
W = Path('/kaggle/working')
LOGS = Path('/tmp/p04m/logs')                                                 # journaux des sous-processus : hors de /kaggle/working, jamais publiés
LOGS.mkdir(parents=True, exist_ok=True)
def _report_path():
    return W / f"p04m_passe{globals().get('PASSE', 'x')}_report.json"
def _close_gate(why):
    # une porte déjà publiée ne reste jamais OUVERTE à côté d'un rapport d'échec (Sonnet, point 7) : réécrite fermée
    gp = W / 'porte_pairage.json'
    if globals().get('PASSE') == 1 and gp.exists():
        try:
            g = json.loads(gp.read_text())
        except ValueError:
            g = {}
        g = g if isinstance(g, dict) else {}
        g.update({'porte_ouverte': False, 'closed_because': f'passe incomplète : {why}'})
        g.setdefault('pages_closing', ['(porte fermée par le carnet)'])
        json.dump(g, open(gp, 'w'), indent=1)
def _write_failure(reason):
    json.dump({'passe': globals().get('PASSE'), 'incomplete': [reason], 'machine': globals().get('MACHINE'), 'elapsed_hours': round(elapsed_h(), 3)},
              open(_report_path(), 'w'), indent=1)
    _close_gate(reason)
def fail(reason):
    # rapport d'échec FINAL, toujours écrit (pQ B1) ; PASSE et MACHINE peuvent ne pas exister encore
    _write_failure(reason)
    raise SystemExit(f'PASSE INCOMPLÈTE : {reason}, sans reprise')
def _abort(e):
    # toute autre sortie d'une cellule : rapport d'échec s'il n'y en a pas déjà un ; en passe 2, seuls nos propres messages (assert, SystemExit)
    if _report_path().exists():
        _close_gate(f'{type(e).__name__}')
        return
    own = isinstance(e, (AssertionError, SystemExit))
    _write_failure(f'{type(e).__name__}: ' + (str(e)[:300] if own or globals().get('PASSE') != 2 else '(message non affiché en passe 2)'))
def guard(label):
    if remaining() <= RESERVE_S:
        fail(f'temps : {elapsed_h():.2f} h, échéance (réserve comprise) atteinte avant {label}')
def _kill_group(proc, grace_s=10.0):
    # tout le GROUPE (l'étape et ses descendants) : SIGTERM, puis SIGKILL (mécanisme de data2_common)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
        t = time.monotonic() + grace_s
        while time.monotonic() < t:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
def bounded(label, argv, cwd=None, env=None, quiet=False):
    # TOUT sous-processus du carnet passe par ici (pQ B1) : groupe propre, sorties dans des fichiers (pas de tube), attente bornée par le temps
    # restant moins la réserve ; à l'échéance le groupe entier est tué et le rapport d'échec écrit ; après une fin normale, les descendants
    # restants du groupe sont tués aussi. Le groupe est tué et récolté dans un finally : TOUTE sortie après Popen (échéance, KeyboardInterrupt,
    # autre exception) le termine, et l'exception d'origine est conservée (pQ B1, 3e audit).
    guard(label)
    o, e = LOGS / f'{label}.out', LOGS / f'{label}.err'
    timed_out = False
    with open(o, 'w') as fo, open(e, 'w') as fe:
        proc = subprocess.Popen([str(a) for a in argv], cwd=cwd, env=env, stdout=fo, stderr=fe, start_new_session=True)
        try:
            try:
                proc.wait(timeout=max(1.0, remaining() - RESERVE_S))
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if proc.poll() is None:
                _kill_group(proc)                                             # l'étape tourne encore : SIGTERM puis SIGKILL au groupe, puis récolte
            try:
                os.killpg(proc.pid, signal.SIGKILL)                           # descendants restants du groupe
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    if timed_out:
        fail(f'temps : échéance atteinte pendant {label} (groupe de processus tué)')
    return proc.returncode, ('' if quiet else o.read_text(errors='replace')), ('' if quiet else e.read_text(errors='replace'))
def porte_acceptee(gate, gate_sha):
    # note B (pA, 25/10) : une porte fermée est acceptée UNIQUEMENT si c'est ce fichier exact, publié par la passe 1, non réécrit
    NOTE_B_GATE_SHA = '05c1bb5267a31c1396f48ddd69a71eba38a3c6804d039927a8e2b6cf9d9b6e07'
    NOTE_B_CLOSING = ['L_OEuvre_4613537_T0000001', 'Le_Matin_0551841_T0000003', 'Marie-Claire_4621234_T0000008']
    if not isinstance(gate, dict):
        return False
    return gate.get('porte_ouverte') is True or (gate_sha == NOTE_B_GATE_SHA and gate.get('porte_ouverte') is False
                                                and gate.get('pages_closing') == NOTE_B_CLOSING)
""", wrap=False)
code("""
MODEL_NAME, MODEL_SHA = 'best_0.6432.safetensors', '3e278266d316f12e9dc0452862b53a37e542eab6ceda1a3213afdbe268b1d742'
EXPECTED_LEGACY_KEYS = ['hyper_params', 'bounding_regions']          # p04_config.json 39d3ed4e (AV-P04 : configuration épinglée)
models = glob.glob('/kaggle/input/**/' + MODEL_NAME, recursive=True)
gates = glob.glob('/kaggle/input/**/porte_pairage.json', recursive=True)
if not models and not gates:
    PASSE = 1
elif len(models) == 1 and len(gates) == 1:
    PASSE = 2
else:
    raise SystemExit(f'sources incohérentes : {len(models)} modèle(s), {len(gates)} porte(s) : arrêt')
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '4'      # même valeur dans les deux passes
lscpu = bounded('lscpu', ['lscpu'])[1]
cpu_model = next((l.split(':', 1)[1].strip() for l in lscpu.splitlines() if l.startswith('Model name')), 'inconnu')
MACHINE = {'cpu_model': cpu_model, 'nproc': os.cpu_count(), 'lscpu': lscpu, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'platform': platform.platform()}
print('PASSE', PASSE, '| CPU', cpu_model, '| nproc', os.cpu_count())
INCOMPLETE = []
if PASSE == 2:
    gate = json.loads(Path(gates[0]).read_text())                            # la porte d'abord : fermée = le modèle n'est pas touché
    assert porte_acceptee(gate, sha256(gates[0])), 'porte de pairage FERMÉE : la passe 2 ne démarre pas'   # note B : seule la porte 05c1bb52
    MODEL = models[0]
    assert sha256(MODEL) == MODEL_SHA, 'SHA du modèle affiné différent de l archive du 18/10 : arrêt'
    CONSUMED = {'porte_sha256': sha256(gates[0]), 'porte_note_b': gate.get('porte_ouverte') is not True}   # liens SHA porte -> passe 1 -> passe 2 (pQ B4)
    p1rep = glob.glob('/kaggle/input/**/p04m_passe1_report.json', recursive=True)
    assert len(p1rep) == 1, 'rapport de la passe 1 absent'
    P1 = json.loads(Path(p1rep[0]).read_text())
    CONSUMED['passe1_report_sha256'] = sha256(p1rep[0])
    assert (P1.get('gate') or {}).get('sha256') == CONSUMED['porte_sha256'], 'la porte montée n est pas celle du rapport de la passe 1 : arrêt'
    if P1['machine']['cpu_model'] != cpu_model or P1['machine']['nproc'] != os.cpu_count():
        INCOMPLETE.append('CPU différent de la passe 1')
    print('modèle affiné : SHA OK ; porte', 'fermée acceptée par la note B (05c1bb52)' if CONSUMED['porte_note_b'] else 'ouverte', '; CPU identique :', not INCOMPLETE)
    if INCOMPLETE:
        json.dump({'passe': 2, 'incomplete': INCOMPLETE, 'machine': MACHINE}, open(W / 'p04m_passe2_report.json', 'w'), indent=1)
        raise SystemExit(f'MESURE INCOMPLÈTE : {INCOMPLETE}')
""")
md("## 1. Jeux d'entrée : babaorum-p07b (morceaux, recollés et vérifiés comme en P-07b) et babaorum-p04m (petit, manifeste vérifié) ; aucun XML")
code("""
import tarfile
ARCHIVE_SHA256, N_PARTS = 'ccfc91bba35df7a54994de0ed9c928ca98b2c973ab13d7db7da48425f7e4c2d5', 76
parts = sorted(glob.glob('/kaggle/input/**/part_??', recursive=True), key=os.path.basename)
assert len(parts) == N_PARTS, (len(parts), parts[:3])
arch = Path('/tmp/babaorum-p07b.tar.gz')
with open(arch, 'wb') as g:
    for q in parts:
        g.write(open(q, 'rb').read())
assert sha256(arch) == ARCHIVE_SHA256, 'SHA de l archive babaorum-p07b'
DS = Path('/tmp/p04m_dataset')                                                 # hors de /kaggle/working : sorties légères
DS.mkdir(exist_ok=True)
with tarfile.open(arch) as t:
    t.extractall(DS)
arch.unlink()
IN7 = DS / 'babaorum-p07b'
IN4 = Path(next(p for p in glob.glob('/kaggle/input/**/MANIFEST_SHA256.txt', recursive=True) if 'pins_p04m.json' in os.listdir(os.path.dirname(p)))).parent
for root in (IN7, IN4):
    bad = [rel for h, rel in (l.split('  ', 1) for l in (root / 'MANIFEST_SHA256.txt').read_text().splitlines()) if sha256(root / rel) != h]
    assert not bad, f'{root} : fichiers modifiés {bad[:5]}'
    assert not list(root.rglob('*.xml')), f'{root} : un XML (vérité) ne doit pas s y trouver'
for a_ in (IN7 / 'axel.tar.gz', IN4 / 'axel_tar_gz', IN4 / 'runs_26_reference_tar_gz'):          # p04m: no archive extension (Kaggle unpacks *.tar.gz)
    assert not [m for m in tarfile.open(a_).getnames() if m.lower().endswith('.xml')], f'{a_.name} contient un XML'
print('jeux intacts ; manifeste p04m', sha256(IN4 / 'MANIFEST_SHA256.txt'))
""")
md("## 2. Environnements : versions EXACTES de P-07b sur Kaggle (pins_p04m.json), gel complet consigné ; en passe 2, comparé à la passe 1 avant la chaîne")
code("""
PINS = json.loads((IN4 / 'pins_p04m.json').read_text())
PY = {'layout-env': '3.12', 'kraken-env': '3.11', 'axel-venv': '3.12'}
def must(label, argv, **kw):
    rc, out, err = bounded(label, argv, **kw)
    if rc:
        print(err[-1500:])
        fail(f'{label} : code {rc}')
    return out
must('pip_uv', [sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
ENV, FREEZE = {}, {}
for name, pkgs in PINS.items():
    d = Path('/tmp/p07b_envs') / name                       # même chemin qu'en P-07b
    if not d.exists():
        must(f'venv_{name}', ['uv', 'venv', '--python', PY[name], str(d)])
        must(f'install_{name}', ['uv', 'pip', 'install', '--python', str(d / 'bin/python'), '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                                 '--index-strategy', 'unsafe-best-match', *pkgs])
    ENV[name] = str(d / 'bin/python')
    FREEZE[name] = must(f'freeze_{name}', ['uv', 'pip', 'freeze', '--python', ENV[name]]).splitlines()
    thr = bounded(f'threads_{name}', [ENV[name], '-c', 'import torch;print(torch.get_num_threads())'])[1].strip() or 'sans torch'
    MACHINE[f'torch_threads_{name}'] = thr
    print(name, len(FREEZE[name]), 'paquets ; threads torch', thr)
if PASSE == 2 and FREEZE != P1['freeze']:
    INCOMPLETE.append('gel des environnements différent de la passe 1')
    json.dump({'passe': 2, 'incomplete': INCOMPLETE, 'machine': MACHINE, 'freeze': FREEZE}, open(W / 'p04m_passe2_report.json', 'w'), indent=1)
    raise SystemExit('MESURE INCOMPLÈTE : gel différent de la passe 1')
""")
md("## 3. axel (c62abf9, branche p04-blla-modele), modèles des lecteurs, Hugging Face hors ligne, 26 pages dans l'ordre figé (`xml` inexistant)")
code("""
import shutil
AX = W / 'axel'
guard('extraction axel')
if not AX.exists():
    AX.mkdir()
    with tarfile.open(IN4 / 'axel_tar_gz') as t:
        t.extractall(AX)
commit = (IN4 / 'AXEL_COMMIT.txt').read_text().split()[0]
assert commit.startswith('c62abf9'), commit
for rel in ('runs/e8/models/mccatmus_press.mlmodel', 'runs/e5/models/german_print_s.mlmodel'):
    (AX / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(IN7 / 'models' / Path(rel).name, AX / rel)
SMOKE = {'axel-venv': 'import sys;sys.path[:0]=["%s/src","%s/scripts"];import axel.layout.fill,axel.layout.chains,axel.eval.seg_errors,axel.layout.pagetext,axel.layout.gt,axel.pipeline.route;import e28_fill,e32;print("ok")',
         'kraken-env': 'import sys;sys.path[:0]=["%s/src","%s/scripts"];import kraken,axel.image.kraken_read,axel.layout.chains,e12_kraken;print("ok")',
         'layout-env': 'import torch,transformers,PIL;print("ok")'}
for name, snippet in SMOKE.items():
    rc, out, err = bounded(f'import_{name}', [ENV[name], '-c', snippet.replace('%s', str(AX))])
    print(name, 'import', out.strip() or err[-600:])
    assert rc == 0, f'{name} : un import de la chaîne échoue : arrêt'
assert not list(AX.rglob('*.xml'))
pages_all = json.loads((IN4 / 'pages_26.json').read_text())
for e in pages_all:
    own = IN4 / e['image']
    e['image'] = str(own if own.exists() else IN7 / 'images' / Path(e['image']).name)
    assert Path(e['image']).exists() and not Path(e['xml']).exists(), e['page']
route = (AX / 'src/axel/pipeline/route.py').read_text()
assert all(f'"{p["page"]}"' in route for p in pages_all), 'route._YEARS : page absente, mauvais commit d axel'
PAGES = [e['page'] for e in pages_all]
assert len(PAGES) == 26 == len(set(PAGES))
HF = IN7 / 'hf'
BASE_ENV = dict(os.environ, HF_HOME=str(HF), HF_HUB_CACHE=str(HF / 'hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
print('axel', commit, '| 26 pages, ordre figé')
""")
md("## 4. Passe 2 seulement : le modèle affiné se charge par le code d'axel ; clés copiées du `blla` livré = celles de la configuration épinglée (journaux du chargeur coupés)")
code("""
if PASSE == 2:
    chk = ('import logging,sys;logging.disable(logging.WARNING);sys.path[:0]=["%s/src","%s/scripts"];from e12_kraken import blla_model;'
           'n=blla_model(sys.argv[1]);print(",".join(n.legacy_keys_from_stock))') % (AX, AX)
    rc, out, _ = bounded('model_load', [ENV['kraken-env'], '-c', chk, MODEL])         # sortie lue, jamais imprimée ; journaux hors des sorties
    assert rc == 0, 'le modèle affiné ne se charge pas par blla_model : arrêt (sortie non affichée)'
    got = out.strip().split(',') if out.strip() else []
    assert got == EXPECTED_LEGACY_KEYS, f'clés copiées {got} != {EXPECTED_LEGACY_KEYS} : mesure incomplète'
    print('blla affiné chargé ; clés copiées du blla livré :', got)
""")
md("## 5. La chaîne (étapes de P-07b), « 26 sur 26 » après chaque étape, archive SHA de chaque étape (diagnostic seulement, jamais de reprise)")
code("""
RUNS = Path('/tmp/p04m/runs')                                                  # AXEL_RUNS (doit finir par /runs), hors de /kaggle/working
(RUNS / 'e12').mkdir(parents=True, exist_ok=True)
(RUNS / 'e12/pages.json').write_text(json.dumps(pages_all, indent=1))
def env_for(root):
    return dict(BASE_ENV, AXEL_RUNS=str(root))
def run(label, env_name, args, extra_env=None):
    t0 = time.monotonic()                                                     # échéance et groupe de processus : bounded() (pQ B1)
    rc, out, err = bounded(label, [ENV[env_name], *args], cwd=AX, env=dict(env_for(RUNS), **(extra_env or {})), quiet=(PASSE == 2))
    print(f'[{label}] code {rc} en {time.monotonic() - t0:.0f} s')
    if rc:
        if PASSE == 1:                                                        # passe 2 : rien n'est lu ni imprimé (journaux du chargeur du modèle affiné)
            print(out[-3000:], err[-3000:])
        fail(f'étape {label} échouée (code {rc})')
def produced(folder):
    miss = [x for x in PAGES if not (RUNS / folder / f'{x}.json').exists() or (RUNS / folder / f'{x}.json').stat().st_size == 0]
    assert not miss, f'{folder} : {26 - len(miss)}/26 ; manquent {miss[:3]}'
def archive(n, label, folders):
    out = W / f'p04m_passe{PASSE}_{n}_{label}.tar.gz'
    with tarfile.open(out, 'w:gz') as t:
        for f in folders:
            if (RUNS / f).exists():
                t.add(RUNS / f, arcname=f'runs/{f}')
    return sha256(out)
S = 'scripts/'
BLLA = ['--blla-model', MODEL] if PASSE == 2 else []                   # passe 1 : aucune option = code de fc4d70ed
STEPS = [
  ('detect',     'layout-env', [S + 'e12_detect.py', '--det', 'ppdl3', '--split', 'dev'],                             ['e12/blocks/ppdl3']),
  ('blocks',     'kraken-env', [S + 'e12_kraken.py', 'blocks', '--det', 'ppdl3', '--zones', '--split', 'dev', *BLLA], ['e12/lines/ppdl3_z']),
  ('e12_read_z', 'kraken-env', [S + 'e12_kraken.py', 'read', '--src', 'ppdl3_z', '--split', 'dev'],                   ['e12/read/ppdl3_z']),
  ('c1_read',    'kraken-env', [S + 'e22_read.py', '--bench', 'e12', '--src', 'ppdl3_z', '--split', 'dev'],           ['e22/read/ppdl3_z']),
  ('e28_seg',    'kraken-env', [S + 'e28_fill.py', 'seg', '--what', 'rest', *BLLA],                                   ['e28/lines/rest_z']),
  ('e28_select', 'axel-venv',  [S + 'e28_fill.py', 'select'],                                                          ['e28/add/a_z']),
  ('e28_read',   'kraken-env', [S + 'e28_fill.py', 'read'],                                                            ['e28/read/mccatmus_press/a_z', 'e28/read/german_print_s/a_z']),
  ('e32_select', 'axel-venv',  [S + 'e32.py', 'select', '--bench', 'dev'],                                             ['e32/dev/add/a_m']),
  ('e32_read',   'kraken-env', [S + 'e32.py', 'read', '--bench', 'dev'],                                               []),
]
TIMES, ARCH = {}, {}
for n, (label, env_name, args, folders) in enumerate(STEPS, 1):
    t0 = time.monotonic()
    guard(f'avant {label}')
    run(label, env_name, args)
    for f in folders:
        produced(f)
    ARCH[label] = archive(n, label, folders + (['e32/dev/read'] if label == 'e32_read' else []))
    TIMES[label] = round(time.monotonic() - t0)
# read_rest_all, comme en P-07b (pour la décomposition P-08 « pour information ») : à part, jamais dans read/ ni add/ ; HOME=W (axel sous W)
t0 = time.monotonic()
shutil.copy2(IN7 / 'scripts/p07_read_rest_all.py', W / 'p07_read_rest_all.py')
run('read_rest_all', 'kraken-env', [str(W / 'p07_read_rest_all.py')], extra_env={'HOME': str(W)})
miss = [x for x in PAGES if not (RUNS.parent / 'read_rest_all' / f'{x}.json').exists()]
assert not miss, miss
TIMES['read_rest_all'] = round(time.monotonic() - t0)
print(json.dumps(TIMES))
""")
md("## 6. Passe 1 seulement : la PORTE de pairage contre la référence historique (script relu, fonctions de P-07b inchangées)")
code("""
GATE = None
if PASSE == 1:
    ref = Path('/tmp/p04m_reference')
    guard('extraction de la référence')
    with tarfile.open(IN4 / 'runs_26_reference_tar_gz') as t:
        t.extractall(ref)
    shutil.copy2(RUNS / 'e12/pages.json', ref / 'runs/e12/pages.json')      # chain_units needs the page list (no truth: xml does not exist)
    guard('porte')
    gp = W / 'porte_pairage.json'
    if gp.exists():
        gp.unlink()                                                           # jamais de porte préexistante (pQ B2)
    gtmp = Path('/tmp/p04m/porte_brute.json')                                 # le script écrit HORS des sorties ; seul le carnet publie gp, après contrôle
    if gtmp.exists():                                                         # (une porte interrompue par l'échéance ne laisse rien dans les sorties)
        gtmp.unlink()
    GATE, why = None, None
    try:
        g_rc, g_out, g_err = bounded('porte', [ENV['axel-venv'], str(IN4 / 'scripts/p04m_porte.py'), '--ref', str(ref / 'runs'), '--pass1', str(RUNS),
                                     '--axel', str(AX), '--python-axel', ENV['axel-venv'], '--out', str(gtmp)], env=dict(BASE_ENV, HOME=str(W)))
        print(g_out[-2000:], g_err[-2000:])
        got = json.loads(gtmp.read_text()) if gtmp.exists() else None
        pg = (got or {}).get('pages')
        coherent = (isinstance(got, dict) and got.get('porte_ouverte') is True and got.get('pages_closing') == [] and isinstance(pg, dict)
                    and len(pg) == 26 and set(pg) == set(PAGES) and all(v.get('page_ok') is True for v in pg.values()))
        if g_rc == 0 and coherent:
            GATE = got                                                        # OUVERTE seulement si code 0 ET fichier complet et cohérent
        else:
            why = f'code {g_rc}, fichier cohérent : {coherent}'
            GATE = dict(got) if isinstance(got, dict) else {}
    except Exception as e:
        why = f'{type(e).__name__}: {str(e)[:300]}'
        GATE = {}
    if why is not None:                                                       # tout échec ou incohérence FERME, et le fichier est réécrit fermé
        GATE.update({'porte_ouverte': False, 'closed_because': why})
        GATE.setdefault('pages_closing', ['(porte fermée par le carnet)'])
    json.dump(GATE, open(gp, 'w'), indent=1)                                  # publiée seulement ici, ouverte ou réécrite fermée
    shutil.rmtree(ref, ignore_errors=True)
    print('PORTE', 'OUVERTE' if GATE['porte_ouverte'] else 'FERMÉE', GATE['pages_closing'])
""")
md("## 7. Sorties : les dossiers de la chaîne pour le banc sur le Mac (une archive), et le rapport de la passe")
code("""
if remaining() <= 0:                                                          # la réserve (RESERVE_S) devait suffire ; sinon, échec et rapport d'échec
    fail(f'temps : {elapsed_h():.2f} h avant les archives finales')
out = W / f'p04m_passe{PASSE}_runs.tar.gz'
with tarfile.open(out, 'w:gz') as t:
    for f in ('e12', 'e22', 'e28', 'e32'):
        t.add(RUNS / f, arcname=f'runs/{f}')
    t.add(RUNS.parent / 'read_rest_all', arcname='read_rest_all')
if remaining() <= 0:
    out.unlink()
    fail(f'temps : {elapsed_h():.2f} h après les archives finales (> {LIMIT_H} h)')
report = {'passe': PASSE, 'incomplete': [], 'limit_hours': LIMIT_H, 'consumed': CONSUMED if PASSE == 2 else None, 'axel_commit': commit, 'pages': PAGES, 'machine': MACHINE, 'freeze': FREEZE, 'seconds_by_step': TIMES,
          'archives_sha256': ARCH, 'runs_archive_sha256': sha256(out), 'elapsed_hours': round(elapsed_h(), 3), 'clock': 'time.monotonic',
          'manifest_p04m': sha256(IN4 / 'MANIFEST_SHA256.txt'), 'truth_in_chain': False,
          'gate': None if GATE is None else {'porte_ouverte': GATE['porte_ouverte'], 'pages_closing': GATE['pages_closing'],
                                             'sha256': sha256(W / 'porte_pairage.json')}}
json.dump(report, open(W / f'p04m_passe{PASSE}_report.json', 'w'), indent=1)
print(json.dumps({k: report[k] for k in ('passe', 'elapsed_hours', 'runs_archive_sha256', 'gate')}, indent=1))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("kaggle_p04m.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("notebook written:", len(cells), "cells")
